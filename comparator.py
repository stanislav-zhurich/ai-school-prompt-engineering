import json
from datetime import datetime


CRITERIA = [
    "instruction_clarity",
    "output_specification",
    "context_sufficiency",
    "reasoning_scaffolding",
    "constraint_clarity",
    "structural_determinism",
]

CRITERIA_LABELS = {
    "instruction_clarity":    "Instruction Clarity",
    "output_specification":   "Output Specification",
    "context_sufficiency":    "Context Sufficiency",
    "reasoning_scaffolding":  "Reasoning Scaffolding",
    "constraint_clarity":     "Constraint Clarity",
    "structural_determinism": "Structural Determinism",
}

MAX_SCORE_PER_CRITERION = 3
MAX_TOTAL = len(CRITERIA) * MAX_SCORE_PER_CRITERION   # 18


class Comparator:

    # ─────────────────────────────────────────
    # Load and filter results
    # ─────────────────────────────────────────
    def load_results(self, result_files: list) -> list:
        """
        Load all result JSON files.
        Returns flat list of all result entries.
        """
        all_results = []
        for file_path in result_files:
            with open(file_path, "r") as f:
                results = json.load(f)
            for result in results:
                result["_source_file"] = file_path
            all_results.extend(results)
        return all_results


    def filter_scored(self, results: list) -> list:
        """
        Keep only entries that should be scored and have non-zero scores.
        """
        return [
            r for r in results
            if r.get("need_to_score", True)
            and r.get("scores", {}).get("total", 0) > 0
        ]


    # ─────────────────────────────────────────
    # Markdown helpers
    # ─────────────────────────────────────────
    def _score_bar(self, score: int, max_score: int = MAX_SCORE_PER_CRITERION) -> str:
        """Render a compact text progress bar for a single criterion score."""
        filled = round((score / max_score) * 10)
        empty  = 10 - filled
        return f"{'█' * filled}{'░' * empty} {score}/{max_score}"


    def _total_bar(self, total: int) -> str:
        """Render a wider progress bar for the total score."""
        filled = round((total / MAX_TOTAL) * 20)
        empty  = 20 - filled
        return f"{'█' * filled}{'░' * empty} {total}/{MAX_TOTAL} ({round(total / MAX_TOTAL * 100)}%)"


    # ─────────────────────────────────────────
    # Report sections
    # ─────────────────────────────────────────
    def _section_combined_table(self, scored: list) -> str:
        """
        Single merged table: criteria as rows, prompts as columns.
        Includes per-criterion scores, totals, percentages and scorer comments.
        Prompts are ordered from lowest to highest score (left to right).
        """
        ordered = sorted(scored, key=lambda r: r["scores"]["total"])

        sep_col = "|".join("---" for _ in ordered)
        header  = "| Criterion | " + " | ".join(r["name"] for r in ordered) + " |"
        sep     = f"|-----------|{sep_col}|"

        lines = ["## 📊 Results\n", header, sep]

        for key in CRITERIA:
            label = CRITERIA_LABELS[key]
            vals  = " | ".join(str(r["scores"].get(key, 0)) for r in ordered)
            lines.append(f"| {label} | {vals} |")

        lines.append("| **Total** | " + " | ".join(f"**{r['scores']['total']}/{MAX_TOTAL}**" for r in ordered) + " |")
        lines.append("| **%** | " + " | ".join(f"**{r['scores']['percentage']}%**" for r in ordered) + " |")
        lines.append("| **Comments** | " + " | ".join(f"*{r['scores'].get('comments', '')}*" for r in ordered) + " |")
        lines.append("")

        return "\n".join(lines) + "\n"


    def _collapse_resume(self, user_message: str) -> str:
        """
        Replace the embedded resume block with a placeholder so the prompt
        instructions are readable without the candidate's data.

        Strategy: find where the post-resume instructions begin by scanning
        for known instruction-opener words. Everything between the opening
        instruction line and that marker is the resume and gets replaced.
        """
        instruction_markers = [
            "\nFollow",
            "\nPlease",
            "\nReturn",
            "\nStep 1",
            "\nProvide",
        ]

        instruction_start = None
        for marker in instruction_markers:
            pos = user_message.find(marker)
            if pos != -1 and (instruction_start is None or pos < instruction_start):
                instruction_start = pos

        opening = user_message.split("\n")[0]

        if instruction_start is not None:
            instructions = user_message[instruction_start:].lstrip("\n")
            return f"{opening}\n\n[... resume content ...]\n\n{instructions}"

        # No instruction markers found (e.g. bad prompt with no follow-up text)
        return f"{opening}\n\n[... resume content ...]"


    def _section_prompts(self, scored: list) -> str:
        """
        Show the system message, user message, and model response for every scored prompt.
        Resume content is collapsed to keep the section readable.
        """
        ordered = sorted(scored, key=lambda r: r["scores"]["total"])
        lines   = ["## 📝 Prompts Used\n"]

        for r in ordered:
            name           = r["name"]
            system_message = r.get("system_message", "").strip()
            user_message   = self._collapse_resume(r.get("user_message", "").strip())
            response_text  = r.get("response_text", "").strip()

            lines.append(f"### {name}\n")

            if system_message:
                lines.append("**System message:**\n")
                lines.append(f"```\n{system_message}\n```\n")
            else:
                lines.append("**System message:** *(none)*\n")

            lines.append("**User message:**\n")
            lines.append(f"```\n{user_message}\n```\n")

            if response_text:
                lines.append("**Model response:**\n")
                lines.append(f"{response_text}\n")

        return "\n".join(lines) + "\n"


    def _section_score_bars(self, scored: list) -> str:
        ranked = sorted(scored, key=lambda r: r["scores"]["total"], reverse=True)
        lines  = ["## 📈 Score Breakdown\n"]

        for r in ranked:
            s    = r["scores"]
            name = r["name"]
            ver  = r.get("version", "")
            lines.append(f"### {name} ({ver})\n")
            lines.append(f"**Total** `{self._total_bar(s['total'])}`\n")
            lines.append("| Criterion | Score |")
            lines.append("|-----------|-------|")
            for key in CRITERIA:
                label = CRITERIA_LABELS[key]
                val   = s.get(key, 0)
                lines.append(f"| {label} | `{self._score_bar(val)}` |")
            lines.append("")

        return "\n".join(lines) + "\n"


    def _section_improvements(self, scored: list) -> str:
        if len(scored) < 2:
            return ""

        ranked = sorted(scored, key=lambda r: r["scores"]["total"])
        worst  = ranked[0]
        best   = ranked[-1]

        lines = ["## 📉 → 📈 Improvement Summary\n"]
        lines.append(f"**Worst prompt:** {worst['name']} ({worst['scores']['total']}/{MAX_TOTAL} — {worst['scores']['percentage']}%)")
        lines.append(f"**Best prompt:**  {best['name']} ({best['scores']['total']}/{MAX_TOTAL} — {best['scores']['percentage']}%)\n")

        gain = best["scores"]["total"] - worst["scores"]["total"]
        lines.append(f"**Total gain:** +{gain} points ({best['scores']['percentage'] - worst['scores']['percentage']} percentage points)\n")

        lines.append("| Criterion | Worst → Best | Gain |")
        lines.append("|-----------|-------------|------|")
        for key in CRITERIA:
            label = CRITERIA_LABELS[key]
            w = worst["scores"].get(key, 0)
            b = best["scores"].get(key, 0)
            diff = b - w
            arrow = f"+{diff}" if diff >= 0 else str(diff)
            lines.append(f"| {label} | {w} → {b} | {arrow} |")

        lines.append("")
        return "\n".join(lines) + "\n"


    def _section_per_file(self, all_results: list, scored: list) -> str:
        scored_ids = {r["id"] for r in scored}
        files = {}
        for r in all_results:
            src = r.get("_source_file", "unknown")
            files.setdefault(src, []).append(r)

        lines = ["## 📁 Results by File\n"]
        for src, entries in files.items():
            lines.append(f"### `{src}`\n")
            lines.append("| Prompt | Scored | Total | % |")
            lines.append("|--------|--------|-------|---|")
            for e in entries:
                scored_flag = "✅" if e["id"] in scored_ids else "⏭ skipped"
                total = e.get("scores", {}).get("total", "-")
                pct   = e.get("scores", {}).get("percentage", "-")
                total_str = f"{total}/{MAX_TOTAL}" if isinstance(total, int) else "-"
                pct_str   = f"{pct}%" if isinstance(pct, int) else "-"
                lines.append(f"| {e['name']} | {scored_flag} | {total_str} | {pct_str} |")
            lines.append("")

        return "\n".join(lines) + "\n"


    # ─────────────────────────────────────────
    # Main: generate report
    # ─────────────────────────────────────────
    def generate_report(self, result_files: list, output_file: str = "results/comparison_report.md") -> str:
        """
        Load all result files, compare scores, and write a Markdown report.

        Args:
            result_files: list of result JSON file paths
            output_file:  path for the generated Markdown file

        Returns:
            path to the generated report
        """
        all_results = self.load_results(result_files)
        scored      = self.filter_scored(all_results)

        if not scored:
            print("No scored results found. Run the scorer first.")
            return output_file

        scored_sorted = sorted(scored, key=lambda r: r["scores"]["total"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model     = scored[0].get("model_used", "unknown")

        report = []
        report.append(f"# Prompt Quality Comparison Report\n")
        report.append(f"**Generated:** {timestamp}  ")
        report.append(f"**Model:** {model}  ")
        report.append(f"**Prompts compared:** {len(scored)}  ")
        report.append(f"**Scoring scale:** 0–3 per criterion, max {MAX_TOTAL} total\n")
        report.append("---\n")

        report.append(self._section_combined_table(scored))
        report.append(self._section_prompts(scored))

        report.append("---\n")
        report.append("*Report generated by comparator.py*\n")

        md_content = "\n".join(report)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"Report saved to: {output_file}")
        print(f"Prompts compared: {len(scored)}")
        for r in sorted(scored, key=lambda x: x["scores"]["total"], reverse=True):
            print(f"  {r['name']}: {r['scores']['total']}/{MAX_TOTAL} ({r['scores']['percentage']}%)")

        return output_file
