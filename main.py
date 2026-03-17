from prompt_runner import PromptRunner
from scorer import Scorer
from comparator import Comparator
from openai import AzureOpenAI
import os

DIAL_URL = "https://ai-proxy.lab.epam.com"
RESUME_MODEL_NAME = "meta.llama4-scout-17b-instruct-v1:0"
PROMPT_SCORING_MODEL_NAME = "gpt-4o-mini-2024-07-18"

API_VERSION = "2024-10-21"

def main():

    client = AzureOpenAI(
                api_key         = os.environ['OPENAI_API_KEY'],
                api_version     = API_VERSION,
                azure_endpoint  = DIAL_URL
            )

    runner = PromptRunner(RESUME_MODEL_NAME,  client)
    scorer = Scorer(PROMPT_SCORING_MODEL_NAME, client)
    comparator = Comparator()

    # Load resume
    resume = open("data/sample_resume.txt").read()

    # Run bad and optimized prompts
    optimized_prompts_results = runner.run_all_prompts(
        "prompts/01_prompt_optimization.json",
        resume
    )

    # Run ReAct self-reflection prompts
    react_selfreflection_prompts_results = runner.run_all_prompts(
        "prompts/02_react_selfreflection.json",
        resume
    )

    # Generate AI refined prompt
    ai_response = runner.run_meta_prompt(
        "prompts/03_meta_prompting.json"
    )

    # Fill AI refined prompt into JSON
    runner.fill_ai_refined_prompt(
        "prompts/03_meta_prompting.json",
        ai_response
    )

    # Run all meta prompting prompts including V4   
    meta_prompting_prompts_results = runner.run_all_prompts(
        "prompts/03_meta_prompting.json",
        resume
    )

    
    all_result_files = [optimized_prompts_results, react_selfreflection_prompts_results, meta_prompting_prompts_results]
    # Score all results
    scorer.score_all_files(all_result_files)

    comparator.generate_report(all_result_files)


if __name__ == "__main__":
    main()