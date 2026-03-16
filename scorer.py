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
    def score_response(self, response_text: str, user_message: str) -> dict:
        """
        Now takes BOTH response AND the prompt that was used
        So it can evaluate if response followed prompt instructions
        """

        scoring_prompt = f"""
    You are an expert prompt quality evaluator.

    Your goal is to measure PROMPT QUALITY — how well the prompt
    guided the model to produce good output.

    IMPORTANT: Modern LLMs produce good responses even to vague prompts.
    A good response does NOT mean the prompt deserves credit.
    You are scoring the PROMPT, not the response.

    PROMPT THAT WAS USED:
    -----
    {user_message}
    -----

    RESPONSE TO EVALUATE:
    -----
    {response_text}
    -----

    SCORING SCALE:
    5 = Prompt EXPLICITLY requested this feature AND response delivered it fully
    4 = Prompt EXPLICITLY requested this feature AND response mostly delivered it
    3 = Prompt did NOT request this feature but response included it anyway (coincidence, no credit to prompt)
    2 = Prompt requested this feature but response barely followed it
    1 = Prompt did NOT request this AND response does not have it
        OR prompt requested it AND response completely ignored it

    DEFINITIONS — "EXPLICITLY requested" means the prompt contained:
    - Named steps or numbered instructions (for reasoning_shown)
    - A dedicated reflection or self-check step (for reflection_shown)
    - A defined output template with section names (for format_followed)
    - A list of sections to evaluate or a request for actionable suggestions (for specificity)
    - A requirement to cover all sections or be comprehensive (for completeness)
    - Clear, detailed instructions overall (for prompt_adherence)

    A prompt that only says "review this" or "tell me if it is good"
    has NOT explicitly requested any of the above.
    If the response happens to be specific and complete despite a vague prompt,
    score specificity = 3 and completeness = 3 (coincidence), NOT 4 or 5.

    ANCHORING EXAMPLES (different domain — for illustration only):

    Example A — Vague prompt: "Check this business plan and tell me if it will work"
    Correct scores:
      prompt_adherence: 2  (almost no instructions; response went far beyond what was asked)
      reasoning_shown:  1  (no reasoning steps requested and response shows none)
      reflection_shown: 1  (no reflection requested and response contains none)
      format_followed:  1  (no format specified; if response invents a format anyway = 3 at most)
      specificity:      3  (prompt never asked for specific criteria; response was specific by coincidence)
      completeness:     3  (prompt set almost no requirements; response covered many areas by coincidence)

    Example B — Well-structured prompt:
      "Analyze this business plan using these steps:
       Step 1 - Market analysis  Step 2 - Financial viability  Step 3 - Risk assessment
       After each step reflect: did I miss anything important?
       Output format: STRENGTHS / WEAKNESSES / RECOMMENDATIONS"
    Correct scores for a response that follows all instructions: 4-5 for all criteria.

    CRITERIA TO SCORE:
    - prompt_adherence:  Did the prompt provide clear, detailed instructions AND did the response follow them precisely?
    - reasoning_shown:   Did the prompt EXPLICITLY ask for numbered steps or chain-of-thought reasoning AND did the response show it?
    - reflection_shown:  Did the prompt EXPLICITLY ask for a self-reflection or self-check step AND did the response include it?
    - format_followed:   Did the prompt EXPLICITLY define an output template or section names AND did the response follow that exact format?
    - specificity:       Did the prompt EXPLICITLY request specific criteria, sections to check, or actionable suggestions AND did the response deliver them?
    - completeness:      Did the prompt EXPLICITLY list sections to cover or require comprehensive coverage AND did the response cover all of them?

    Return ONLY this format:
    prompt_adherence: [score]
    reasoning_shown: [score]
    reflection_shown: [score]
    format_followed: [score]
    specificity: [score]
    completeness: [score]
    comments: [one sentence explaining the overall score]
    """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": scoring_prompt}],
            temperature=0
        )

        raw_scores = response.choices[0].message.content
        scores = self.parse_scores(raw_scores)

        return scores


    # ─────────────────────────────────────────
    # Parse AI scoring response into dict
    # ─────────────────────────────────────────
    def parse_scores(self, raw_scores: str) -> dict:
        
        scores = {
            "prompt_adherence":  0,
            "reasoning_shown":   0,
            "reflection_shown":  0,
            "format_followed":   0,
            "specificity":       0,
            "completeness":      0,
            "total":             0,
            "percentage":        0,
            "comments":          ""
        }

        lines = raw_scores.strip().split("\n")

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key in scores and key not in ["total", "percentage", "comments"]:
                    try:
                        scores[key] = int(value)
                    except ValueError:
                        scores[key] = 0

                elif key == "comments":
                    scores["comments"] = value

        # Calculate total and percentage
        criteria = [
            "prompt_adherence",
            "reasoning_shown", 
            "reflection_shown",
            "format_followed",
            "specificity",
            "completeness"
        ]

        scores["total"] = sum(scores[c] for c in criteria)
        scores["percentage"] = round((scores["total"] / 30) * 100)

        return scores


    # ─────────────────────────────────────────
    # Score all results in a JSON file
    # ─────────────────────────────────────────
    def score_results_file(self, results_file: str) -> None:

        with open(results_file, "r") as file:
            results = json.load(file)

        for result in results:
            print(f"Scoring: {result['id']} - {result['name']}")

            # ← Pass BOTH response AND prompt used
            scores = self.score_response(
                response_text = result["response_text"],
                user_message  = result["user_message"]   # ← added this
            )

            result["scores"] = scores

            print(f"Score: {scores['total']}/30 ({scores['percentage']}%)")
            print("-" * 50)

            time.sleep(1)

        with open(results_file, "w") as file:
            json.dump(results, file, indent=2)


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