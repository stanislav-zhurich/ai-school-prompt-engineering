import json
import time
import os
from datetime import datetime
from openai import AzureOpenAI

class PromptRunner:

    def __init__(self, model_name: str, client: AzureOpenAI, temperature: float = 0.0, max_tokens: int = 1000):
        self.client = client
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ─────────────────────────────────────────
    # Load prompts from JSON file
    # ─────────────────────────────────────────
    def load_prompts(self, json_file_path: str) -> list:
        """
        Load prompts from JSON file
        Returns list of prompt objects
        """
        with open(json_file_path, "r") as file:
            prompts = json.load(file)
        return prompts


    # ─────────────────────────────────────────
    # Fill placeholder in prompt
    # ─────────────────────────────────────────
    def fill_placeholder(self, prompt_text: str, resume: str) -> str:
        """
        Replace {resume} placeholder with actual resume text
        """
        return prompt_text.replace("{resume}", resume)


    # ─────────────────────────────────────────
    # Send single prompt to OpenAI API
    # ─────────────────────────────────────────
    def run_single_prompt(self, system_message: str, user_message: str) -> dict:
        """
        Send prompt to OpenAI and return response
        Returns dict with response text and token usage
        """
        # Build messages list
        messages = []

        # Add system message if exists
        if system_message and system_message != "TO BE FILLED BY AI RESPONSE":
            messages.append({
                "role": "system",
                "content": system_message
            })

        # Add user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        # Send to OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Extract response
        return {
            "response_text": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }


    # ─────────────────────────────────────────
    # Run all prompts from a JSON file
    # ─────────────────────────────────────────
    def run_all_prompts(self, json_file_path: str, resume: str) -> list:
        """
        Load all prompts from JSON file
        Run each one with resume as input
        Return list of results
        """
        # Load prompts
        prompts = self.load_prompts(json_file_path)

        results = []

        for prompt in prompts:

            # Skip pending prompts
            if prompt.get("status") == "pending":
                print(f"Skipping {prompt['id']} - status is pending")
                continue

            print(f"Running prompt: {prompt['id']} - {prompt['name']}")

            # Fill resume placeholder
            system_message = prompt.get("system_message", "")
            user_message = self.fill_placeholder(
                prompt.get("user_message", ""),
                resume
            )

            # Run prompt
            response = self.run_single_prompt(system_message, user_message)

            # Build result object
            result = {
                "id": prompt["id"],
                "version": prompt.get("version", ""),
                "name": prompt["name"],
                "type": prompt.get("type", ""),
                "quality": prompt.get("quality", ""),
                "need_to_score": prompt.get("need_to_score", True),
                "system_message": system_message,
                "user_message": user_message,
                "response_text": response["response_text"],
                "tokens_used": response["tokens_used"],
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"]
            }

            results.append(result)

           

            print(f"Done: {prompt['id']}")
            print(f"Tokens used: {response['tokens_used']}")
            print("-" * 50)

            # Small delay to avoid API rate limits
            time.sleep(1)


          # ← Collect and save results
        output_file = f"results/{json_file_path.split('/')[-1].split('.')[0]}_results.json"
        self.collect_results(
            results=results,
            source_file=json_file_path,
            output_file=output_file
        )      
        return output_file


    # ─────────────────────────────────────────
    # Special: Run meta prompt to get AI refined prompt
    # ─────────────────────────────────────────
    def run_meta_prompt(self, json_file_path: str) -> str:
        """
        Find meta_request_prompt in JSON file
        Send it to OpenAI
        Return AI refined prompt text
        """
        prompts = self.load_prompts(json_file_path)

        # Find the meta request prompt
        meta_prompt = None
        for prompt in prompts:
            if prompt["type"] == "meta_prompt":
                meta_prompt = prompt
                break

        if not meta_prompt:
            raise ValueError("No meta_prompt found in JSON file")

        print("Running meta prompt - asking AI to improve our prompt...")

        # Run meta prompt
        response = self.run_single_prompt(
            meta_prompt["system_message"],
            meta_prompt["user_message"]
        )

        print("Meta prompt done - AI returned refined prompt")
        print(f"Tokens used: {response['tokens_used']}")

        return response["response_text"]


    # ─────────────────────────────────────────
    # Special: Fill AI refined prompt into JSON
    # ─────────────────────────────────────────
    def fill_ai_refined_prompt(self, json_file_path: str, ai_response: str) -> None:
        """
        Parse AI response to extract system and user message
        Fill the pending ai_refined_prompt in JSON file
        Save updated JSON back to file
        """
        prompts = self.load_prompts(json_file_path)

        # Parse AI response to extract system and user message
        system_message, user_message = self.parse_ai_response(ai_response)

        # Find and update the pending ai_refined_prompt
        for prompt in prompts:
            if prompt["type"] == "ai_refined_prompt":
                prompt["system_message"] = system_message
                prompt["user_message"] = user_message
                prompt["status"] = "ready"
                print(f"Filled AI refined prompt: {prompt['id']}")
                break

        # Save updated JSON back to file
        with open(json_file_path, "w") as file:
            json.dump(prompts, file, indent=2)

        print("Saved updated prompts to JSON file")


    # ─────────────────────────────────────────
    # Helper: Parse AI response into system/user message
    # ─────────────────────────────────────────
    def parse_ai_response(self, ai_response: str) -> tuple:
        """
        AI returns response in format:
        SYSTEM MESSAGE:
        [system message content]

        USER MESSAGE:
        [user message content]

        This method extracts both parts
        Returns (system_message, user_message)
        """
        system_message = ""
        user_message = ""

        # Split by USER MESSAGE marker
        if "USER MESSAGE:" in ai_response:
            parts = ai_response.split("USER MESSAGE:")
            user_message = parts[1].strip()

            # Extract system message
            if "SYSTEM MESSAGE:" in parts[0]:
                system_part = parts[0].split("SYSTEM MESSAGE:")[1]
                system_message = system_part.strip()

        return system_message, user_message

    def collect_results(self, 
                        results: list, 
                        source_file: str, 
                        output_file: str) -> None:
        """
        Takes raw results list from run_all_prompts
        Adds metadata to each result
        Saves to output JSON file

        Args:
            results:     list of raw results from run_all_prompts
            source_file: which prompt JSON file was used
                         e.g. "02_prompt_optimization.json"
            output_file: where to save results
                         e.g. "results/02_prompt_optimization_results.json"
        """

        collected = []

        for result in results:

            # Build collected result with metadata
            collected_result = {

                # Original result data
                "id":               result["id"],
                "version":          result["version"],
                "name":             result["name"],
                "type":             result.get("type", ""),
                "quality":          result.get("quality", ""),
                "need_to_score":    result.get("need_to_score", True),

                # Prompt used
                "system_message":   result["system_message"],
                "user_message":     result["user_message"],

                # Response received
                "response_text":    result["response_text"],

                # Token usage
                "tokens_used":      result["tokens_used"],
                "prompt_tokens":    result["prompt_tokens"],
                "completion_tokens":result["completion_tokens"],

                # Metadata added by collector
                "source_file":      source_file,
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_used":       self.model,

                # Empty scores - will be filled by scorer.py later
                "scores": {
                    "prompt_adherence": 0,
                    "reasoning_shown":  0,
                    "reflection_shown": 0,
                    "format_followed":  0,
                    "specificity":      0,
                    "completeness":     0,
                    "total":            0,
                    "percentage":       0,
                    "comments":         ""
                }
            }

            collected.append(collected_result)

        # Save to output JSON file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as file:
            json.dump(collected, file, indent=2)

        print(f"Saved {len(collected)} results to {output_file}")