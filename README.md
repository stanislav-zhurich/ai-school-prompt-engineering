# AI Resume Review — Prompt Engineering Lab

A hands-on prompt engineering experiment that evaluates the same resume using prompts of increasing quality, scores each response with an AI evaluator, and generates a side-by-side comparison report.

---

## What the application does

The application answers the question: **does a better prompt produce a better response?**

It runs a single resume through five progressively more sophisticated prompts — from a one-liner to a fully structured ReAct reasoning loop — and measures the quality of each prompt on six engineering criteria. The results are written to JSON files and summarised in a human-readable Markdown report.

The three prompt techniques explored are:

| Part | Technique | What is tested |
|------|-----------|----------------|
| 1 | **Prompt Optimisation** | Bad → Improved → Optimised prompt |
| 2 | **ReAct + Self-Reflection** | Thought / Action / Observation loop with explicit self-check |
| 3 | **Meta Prompting** | Using AI as a prompt engineer to refine an existing prompt |

---

## Application structure

```
ai-school-prompt-engineering/
│
├── main.py                  # Entry point — orchestrates the full pipeline
├── prompt_runner.py         # Loads prompts, calls the LLM, saves results
├── scorer.py                # Evaluates each response using AI scoring
├── comparator.py            # Loads all results and generates the MD report
│
├── prompts/
│   ├── 01_prompt_optimization.json      # Bad / Improved / Optimised prompts
│   ├── 02_react_selfreflection.json     # ReAct self-reflection prompt
│   └── 03_meta_prompting.json           # Meta prompt + AI-refined prompt
│
├── data/
│   └── sample_resume.txt    # Resume used as input for all prompts
│
├── results/                 # Generated at runtime — do not edit manually
│   ├── 01_prompt_optimization_results.json
│   ├── 02_react_selfreflection_results.json
│   ├── 03_meta_prompting_results.json
│   └── comparison_report.md             # Final human-readable report
│
├── requirements.txt
└── README.md
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          main.py                            │
└────────┬──────────────────┬──────────────────┬─────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │  prompts/   │   │   data/     │   │  results/   │
  │  *.json     │   │  resume.txt │   │  *.json     │
  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
         │                 │                  │
         └────────┬─────────┘                 │
                  ▼                           │
         ┌────────────────┐                   │
         │ prompt_runner  │ ──── writes ──────▶
         │  (LLM calls)   │                   │
         └────────────────┘                   │
                                              │
         ┌────────────────┐                   │
         │    scorer      │ ◀──── reads ───────
         │  (AI scoring)  │ ──── updates ─────▶
         └────────────────┘                   │
                                              │
         ┌────────────────┐                   │
         │  comparator    │ ◀──── reads ───────
         │ (MD report)    │
         └───────┬────────┘
                 │ writes
                 ▼
        results/comparison_report.md
```

**Two models are used:**
- `RESUME_MODEL_NAME` — generates resume feedback (runs the prompts)
- `PROMPT_SCORING_MODEL_NAME` — evaluates prompt quality (runs the scorer)

Both are configured at the top of `main.py`.

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "your-api-key"

# macOS / Linux
export OPENAI_API_KEY="your-api-key"
```

### 3. (Optional) Change the resume

Replace the contents of `data/sample_resume.txt` with any resume in plain text format.

### 4. Run the pipeline

```bash
python main.py
```

This will:
1. Run all prompts against the resume and save raw responses to `results/`
2. Ask AI to generate an improved (meta-prompted) version of the best prompt
3. Score every prompt on six engineering criteria (0–3 per criterion, max 18 total)
4. Generate the comparison report

### 5. Run comparator only (re-generate report from existing results)

```python
from comparator import Comparator

Comparator().generate_report([
    "results/01_prompt_optimization_results.json",
    "results/02_react_selfreflection_results.json",
    "results/03_meta_prompting_results.json"
])
```

---

## How to find results

| Output | Location | Description |
|--------|----------|-------------|
| Raw LLM responses + scores | `results/01_prompt_optimization_results.json` | Prompts v1, v2, v3 |
| Raw LLM responses + scores | `results/02_react_selfreflection_results.json` | ReAct prompt |
| Raw LLM responses + scores | `results/03_meta_prompting_results.json` | Original, meta-request, AI-refined |
| **Comparison report** | **`results/comparison_report.md`** | **Side-by-side scores, criteria breakdown, and full prompt text** |

Open `results/comparison_report.md` in any Markdown viewer (VS Code preview, GitHub, etc.) for the final formatted report.

### Scoring criteria

Each **prompt** is scored from 0 to 3 on six engineering criteria (max total: 18).  
The scorer uses a chain-of-thought approach: it first writes explicit evidence from the prompt for each criterion, then assigns a score — reducing anchoring bias.

| Criterion | What is measured |
|-----------|-----------------|
| Instruction Clarity | Are the instructions precise and unambiguous about *how* to perform the task? |
| Output Specification | Does the prompt define the expected output format, structure, or length? |
| Context Sufficiency | Does the prompt supply enough background and constraints for the task? |
| Reasoning Scaffolding | Does the prompt explicitly structure or guide the reasoning process (e.g. numbered steps, chain-of-thought)? |
| Constraint Clarity | Does the prompt set explicit boundaries on scope, tone, persona, or what to avoid? |
| Reliability | Would this prompt consistently produce similar-quality output across multiple runs? |

**Score scale per criterion:**

| Score | Meaning |
|-------|---------|
| 3 | Criterion fully and explicitly addressed in the prompt |
| 2 | Partially addressed — present but incomplete |
| 1 | Barely addressed — implied but not enforced |
| 0 | Not addressed at all |

> The scorer evaluates the **prompt**, not the response.  
> A capable LLM may produce a good response even from a vague prompt — that does not earn the prompt a higher score.
