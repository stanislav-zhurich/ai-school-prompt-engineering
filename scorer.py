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
You are a senior prompt engineer evaluating PROMPT QUALITY.

Your task is to assess how well the PROMPT (not the response) was engineered.
A well-crafted prompt should be able to consistently guide any capable LLM
to produce high-quality output — the response is evidence, not the subject.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT UNDER EVALUATION:
{user_message}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL RESPONSE PRODUCED BY THAT PROMPT:
{response_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — THINK BEFORE SCORING
For each criterion below, write 1–2 sentences of evidence from the prompt
(not the response) before assigning a score.

CRITERION DEFINITIONS AND SCORING SCALE (0–3):

1. instruction_clarity
   Does the prompt give unambiguous, specific instructions?
   3 = Instructions are precise, leave no room for interpretation
   2 = Mostly clear but has at least one ambiguous phrase
   1 = Vague or could be interpreted in multiple ways
   0 = No real instructions; just a topic or question

2. output_specification
   Does the prompt define the expected output format, structure, or length?
   3 = Explicit template, section names, or format rules provided
   2 = Partial format guidance (e.g., "use bullet points" but no structure)
   1 = Implicit format expectations only
   0 = No output format specified at all

3. context_sufficiency
   Does the prompt supply enough background/constraints for the task?
   3 = All necessary context and constraints are present
   2 = Adequate context but missing at least one important constraint
   1 = Minimal context; model must make significant assumptions
   0 = No context beyond the bare task statement

4. reasoning_scaffolding
   Does the prompt explicitly structure or guide the reasoning process?
   3 = Numbered steps, explicit chain-of-thought, or staged analysis required
   2 = Hints at a process but does not enforce it (e.g., "consider each aspect")
   1 = Implicitly expects some reasoning but gives no structure
   0 = No reasoning guidance; model left to decide how to think

5. constraint_clarity
   Does the prompt set boundaries on what to include/exclude or how to behave?
   3 = Explicit constraints (scope, tone, what to avoid, persona, etc.)
   2 = Some constraints present but incomplete
   1 = Very loose constraints that could easily be ignored
   0 = No constraints specified

6. structural_determinism
   How much does the prompt's design constrain the model's degrees of freedom?
   (Note: this is NOT empirical reliability — it cannot be measured from a single run.
   It measures how much the prompt structure limits variation in output.)
   3 = Prompt is highly constraining — explicit steps, format, and scope leave little room for variation
   2 = Moderately constraining — some structure present but creative latitude remains
   1 = Loosely constraining — model has significant freedom to interpret the task
   0 = Unconstrained — prompt imposes no structure; outputs would be largely unpredictable

STEP 2 — OUTPUT (follow this format exactly, no extra lines)
instruction_clarity_reasoning: [your evidence from the prompt]
instruction_clarity: [0-3]
output_specification_reasoning: [your evidence from the prompt]
output_specification: [0-3]
context_sufficiency_reasoning: [your evidence from the prompt]
context_sufficiency: [0-3]
reasoning_scaffolding_reasoning: [your evidence from the prompt]
reasoning_scaffolding: [0-3]
constraint_clarity_reasoning: [your evidence from the prompt]
constraint_clarity: [0-3]
structural_determinism_reasoning: [your evidence from the prompt]
structural_determinism: [0-3]
comments: [one sentence summarizing the prompt's main strength and main weakness]
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

        criteria = [
            "instruction_clarity",
            "output_specification",
            "context_sufficiency",
            "reasoning_scaffolding",
            "constraint_clarity",
            "structural_determinism",
        ]

        scores = {c: 0 for c in criteria}
        scores.update({c + "_reasoning": "" for c in criteria})
        scores.update({"total": 0, "percentage": 0, "comments": ""})

        lines = raw_scores.strip().split("\n")

        for line in lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key in criteria:
                try:
                    scores[key] = int(value)
                except ValueError:
                    scores[key] = 0
            elif key.endswith("_reasoning") and key in scores:
                scores[key] = value
            elif key == "comments":
                scores["comments"] = value

        scores["total"] = sum(scores[c] for c in criteria)
        scores["percentage"] = round((scores["total"] / 18) * 100)

        return scores


    # ─────────────────────────────────────────
    # Score all results in a JSON file
    # ─────────────────────────────────────────
    def score_results_file(self, results_file: str) -> None:

        with open(results_file, "r") as file:
            results = json.load(file)

        for result in results:
            print(f"Scoring: {result['id']} - {result['name']}")

            if not result.get("need_to_score", True):
                print("Skipped (need_to_score = false)")
                print("-" * 50)
                continue

            scores = self.score_response(
                response_text = result["response_text"],
                user_message  = result["user_message"]
            )

            result["scores"] = scores

            print(f"Score: {scores['total']}/18 ({scores['percentage']}%)")
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