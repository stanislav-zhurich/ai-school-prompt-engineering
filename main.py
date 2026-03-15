from prompt_runner import PromptRunner

runner = PromptRunner()
resume = open("data/sample_resume.txt").read()

# Run Part 2
optimized_prompts_results = runner.run_all_prompts(
    "prompts/01_prompt_optimization.json",
    resume
)

# Run Part 1
react_selfreflection_prompts_results = runner.run_all_prompts(
    "prompts/02_react_selfreflection.json",
    resume
)


# Run Part 3 Meta Prompting
# Step 1: Get AI refined prompt
ai_response = runner.run_meta_prompt(
    "prompts/03_meta_prompting.json"
)

# Step 2: Fill AI refined prompt into JSON
runner.fill_ai_refined_prompt(
    "prompts/03_meta_prompting.json",
    ai_response
)

# Step 3: Run all part 3 prompts including V4
meta_prompting_prompts_results = runner.run_all_prompts(
    "prompts/03_meta_prompting.json",
    resume
)