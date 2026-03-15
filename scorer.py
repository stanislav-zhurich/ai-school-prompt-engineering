import json
import time
from openai import AzureOpenAI


class Scorer:

    def __init__(self, model_name: str, client: AzureOpenAI, temperature: float = 0.0, max_tokens: int = 1000):
        self.client = client
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens


    # ─────────────────────────────────────────
    # Score single response using AI
    # ─────────────────────────────────────────
    def score_response(self, response_text: str) -> dict:
        """
        Send response to AI and ask it to rate quality
        Returns dict with scores for each criteria
        """

        scoring_prompt = f"""
            You are an expert prompt evaluator.
            Rate the following resume review response on each criteria below.

            RESPONSE TO EVALUATE:
            -----
            {response_text}
            -----

            Rate each criteria from 1 to 5:
            1 = Very Poor
            2 = Poor
            3 = Average
            4 = Good
            5 = Excellent

            CRITERIA:
            - clarity:       Is the feedback clear and easy to understand?
            - specificity:   Is the feedback specific or vague?
            - relevance:     Does it actually address the resume content?
            - completeness:  Did it cover all sections of the resume?
            - format:        Is the output well structured and organized?
            - reasoning:     Does it show thinking process and reasoning?

            Return your scores in this EXACT format and nothing else:
            clarity: [score]
            specificity: [score]
            relevance: [score]
            completeness: [score]
            format: [score]
            reasoning: [score]
            comments: [one sentence explaining the scores]
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": scoring_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        raw_scores = response.choices[0].message.content

        # Parse scores from response
        scores = self.parse_scores(raw_scores)

        return scores


    # ─────────────────────────────────────────
    # Parse AI scoring response into dict
    # ─────────────────────────────────────────
    def parse_scores(self, raw_scores: str) -> dict:
        """
        Parse raw AI scoring response into scores dict
        
        Input:
        clarity: 4
        specificity: 3
        ...

        Output:
        {
            "clarity": 4,
            "specificity": 3,
            ...
        }
        """
        scores = {
            "clarity":       0,
            "specificity":   0,
            "relevance":     0,
            "completeness":  0,
            "format":        0,
            "reasoning":     0,
            "total":         0,
            "percentage":    0,
            "comments":      ""
        }

        lines = raw_scores.strip().split("\n")

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key in scores and key != "comments":
                    try:
                        scores[key] = int(value)
                    except ValueError:
                        scores[key] = 0

                elif key == "comments":
                    scores["comments"] = value

        # Calculate total and percentage
        criteria = ["clarity", "specificity", "relevance",
                    "completeness", "format", "reasoning"]

        scores["total"] = sum(scores[c] for c in criteria)
        scores["percentage"] = round((scores["total"] / 30) * 100)

        return scores


    # ─────────────────────────────────────────
    # Score all results in a JSON file
    # ─────────────────────────────────────────
    def score_results_file(self, results_file: str) -> None:
        """
        Load results JSON file
        Score each result
        Save updated scores back to file

        Args:
            results_file: path to results JSON file
                          e.g. "results/02_prompt_optimization_results.json"
        """

        # Load results
        with open(results_file, "r") as file:
            results = json.load(file)

        print(f"Scoring {len(results)} results from {results_file}")
        print("-" * 50)

        # Score each result
        for result in results:

            print(f"Scoring: {result['id']} - {result['name']}")

            # Score the response
            scores = self.score_response(result["response_text"])

            # Update scores in result
            result["scores"] = scores

            print(f"Score: {scores['total']}/30 ({scores['percentage']}%)")
            print(f"Comments: {scores['comments']}")
            print("-" * 50)

            # Small delay between API calls
            time.sleep(1)

        # Save updated results back to file
        with open(results_file, "w") as file:
            json.dump(results, file, indent=2)

        print(f"Saved scored results to {results_file}")


    # ─────────────────────────────────────────
    # Score all result files at once
    # ─────────────────────────────────────────
    def score_all_files(self, result_files: list) -> None:
        """
        Score all result files one by one

        Args:
            result_files: list of result file paths
        """
        for file_path in result_files:
            print(f"\n{'='*50}")
            print(f"Processing: {file_path}")
            print(f"{'='*50}")
            self.score_results_file(file_path)