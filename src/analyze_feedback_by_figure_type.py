#!/usr/bin/env python3
"""Analyze Feedback Quality by Figure Type using LLM-as-Judge

Uses Claude to evaluate feedback quality (Match/Partial/Unmatched)
broken down by figure type × scenario.

Tests H4 for feedback: Does visual context help feedback quality more
for tables (structured) vs schematics (spatial)?

Figure Type Normalization (from eval_by_figure_type.py):
- table: content_type="table" OR figure_type contains "table"
- plot: figure_type contains "plot", "graph", or "chart"
- schematic: figure_type contains "schematic", "diagram", or "architecture"
- other: everything else (including N/A, empty, or unrecognized types)

Usage:
    python src/analyze_feedback_by_figure_type.py

    # Smoke test with 2 examples per cell
    python src/analyze_feedback_by_figure_type.py --max-per-cell 2
"""

import argparse
import json
import os
from pathlib import Path

import anthropic


def build_client() -> anthropic.Anthropic:
    """Build Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set."
        )
    return anthropic.Anthropic(api_key=api_key)


JUDGE_PROMPT = """You are evaluating the quality of AI-generated feedback on a student's answer about a scientific figure.

## Ground Truth Feedback (Reference)
{ground_truth}

## Predicted Feedback (To Evaluate)
{predicted}

## Task
Compare the predicted feedback to the ground truth and classify it as:

- **Match**: The predicted feedback captures the same key points as the ground truth. It correctly identifies what the student got wrong and provides accurate guidance.

- **Partial**: The predicted feedback is partially correct - it identifies some issues but misses others, or includes some inaccuracies alongside correct points.

- **Unmatched**: The predicted feedback is incorrect, misleading, or misses the main point entirely.

Respond with ONLY one word: Match, Partial, or Unmatched"""


def judge_feedback(client: anthropic.Anthropic, ground_truth: str, predicted: str) -> str:
    """Use Claude to judge feedback quality."""
    if not predicted or predicted == "N/A" or not ground_truth or ground_truth == "N/A":
        return "Unmatched"

    prompt = JUDGE_PROMPT.format(ground_truth=ground_truth, predicted=predicted)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.content[0].text.strip().lower()

    if "match" in result and "unmatched" not in result and "partial" not in result:
        return "Match"
    elif "partial" in result:
        return "Partial"
    else:
        return "Unmatched"


def analyze_feedback(results_path: str, client: anthropic.Anthropic, max_per_cell: int = None) -> dict:
    """Analyze feedback quality by figure type × scenario using LLM judge."""
    with open(results_path) as f:
        data = json.load(f)

    results = data['results']
    judgments = {}

    figure_types = ['table', 'plot', 'schematic', 'other']
    scenarios = ['text_only', 'caption_only', 'vision_only', 'multimodal']

    total_calls = 0
    for ft in figure_types:
        if ft not in results:
            continue
        for s in scenarios:
            n = len(results[ft][s])
            if max_per_cell:
                n = min(n, max_per_cell)
            total_calls += n

    print(f"Will make {total_calls} LLM judge calls\n")

    for figure_type in figure_types:
        if figure_type not in results:
            continue

        judgments[figure_type] = {}
        print(f"\n{'='*60}")
        print(f"FIGURE TYPE: {figure_type.upper()}")
        print(f"{'='*60}")

        for scenario in scenarios:
            examples = results[figure_type][scenario]
            if max_per_cell:
                examples = examples[:max_per_cell]

            print(f"\n  Scenario: {scenario} ({len(examples)} examples)")

            scenario_judgments = []
            for i, ex in enumerate(examples, 1):
                pred_feedback = ex['predicted'].get('feedback', '')
                ref_feedback = ex['ground_truth'].get('feedback', '')

                print(f"    [{i}/{len(examples)}] ", end="", flush=True)

                judgment = judge_feedback(client, ref_feedback, pred_feedback)
                scenario_judgments.append(judgment)
                print(f"{judgment}", flush=True)

            # Calculate stats
            match_count = sum(1 for j in scenario_judgments if j == "Match")
            partial_count = sum(1 for j in scenario_judgments if j == "Partial")
            unmatched_count = sum(1 for j in scenario_judgments if j == "Unmatched")
            total = len(scenario_judgments)

            judgments[figure_type][scenario] = {
                'match': match_count,
                'partial': partial_count,
                'unmatched': unmatched_count,
                'total': total,
                'match_pct': match_count / total * 100 if total > 0 else 0,
                'soft_match_pct': (match_count + partial_count) / total * 100 if total > 0 else 0,
            }

            print(f"    -> Match: {match_count}, Partial: {partial_count}, Unmatched: {unmatched_count}")

    return judgments


def print_results(judgments: dict) -> None:
    """Print results tables."""
    scenarios = ['text_only', 'caption_only', 'vision_only', 'multimodal']
    figure_types = ['table', 'plot', 'schematic', 'other']

    print("\n" + "=" * 80)
    print("FEEDBACK QUALITY BY FIGURE TYPE (LLM Judge)")
    print("=" * 80)

    # Match % Table
    print("\n### Match Rate (%)")
    print(f"{'Figure Type':<15}", end="")
    for s in scenarios:
        print(f"{s:>15}", end="")
    print(f"{'Δ':>10}")

    for ft in figure_types:
        if ft not in judgments:
            continue
        print(f"{ft:<15}", end="")
        for s in scenarios:
            pct = judgments[ft][s]['match_pct']
            print(f"{pct:>14.1f}%", end=" ")
        delta = judgments[ft]['multimodal']['match_pct'] - judgments[ft]['text_only']['match_pct']
        print(f"{delta:>+8.1f}pp")

    # Soft Match % Table (Match + Partial)
    print("\n### Soft Match Rate (%) - Match + Partial")
    print(f"{'Figure Type':<15}", end="")
    for s in scenarios:
        print(f"{s:>15}", end="")
    print(f"{'Δ':>10}")

    for ft in figure_types:
        if ft not in judgments:
            continue
        print(f"{ft:<15}", end="")
        for s in scenarios:
            pct = judgments[ft][s]['soft_match_pct']
            print(f"{pct:>14.1f}%", end=" ")
        delta = judgments[ft]['multimodal']['soft_match_pct'] - judgments[ft]['text_only']['soft_match_pct']
        print(f"{delta:>+8.1f}pp")

    # Context Benefit Summary
    print("\n" + "=" * 80)
    print("CONTEXT BENEFIT FOR FEEDBACK (Δ = multimodal - text_only)")
    print("=" * 80)

    print(f"\n{'Figure Type':<15} {'Δ Match':>12} {'Δ Soft Match':>14} {'Helps?':>10}")
    for ft in figure_types:
        if ft not in judgments:
            continue
        delta_match = judgments[ft]['multimodal']['match_pct'] - judgments[ft]['text_only']['match_pct']
        delta_soft = judgments[ft]['multimodal']['soft_match_pct'] - judgments[ft]['text_only']['soft_match_pct']
        helps = "YES" if delta_match > 0 else "NO"
        print(f"{ft:<15} {delta_match:>+11.1f}pp {delta_soft:>+13.1f}pp {helps:>10}")

    # H4 Test for Feedback
    print("\n" + "=" * 80)
    print("H4 HYPOTHESIS TEST (Feedback Quality)")
    print("=" * 80)

    if 'table' in judgments and 'schematic' in judgments:
        table_avg = sum(judgments['table'][s]['match_pct'] for s in scenarios) / len(scenarios)
        schematic_avg = sum(judgments['schematic'][s]['match_pct'] for s in scenarios) / len(scenarios)

        print(f"\nH4: Tables easier than Schematics? (Match %)")
        print(f"  Table avg:      {table_avg:.1f}%")
        print(f"  Schematic avg:  {schematic_avg:.1f}%")
        h4_pass = table_avg > schematic_avg
        print(f"  Result: {'PASS' if h4_pass else 'FAIL'}")


def save_report(judgments: dict, output_path: str) -> None:
    """Save markdown report."""
    scenarios = ['text_only', 'caption_only', 'vision_only', 'multimodal']
    figure_types = ['table', 'plot', 'schematic', 'other']

    lines = [
        "# H4 Hypothesis Test: Feedback Quality by Figure Type",
        "",
        "## Method",
        "",
        "Using Claude (claude-sonnet-4-20250514) as judge to evaluate feedback quality.",
        "Each feedback is classified as Match / Partial / Unmatched.",
        "",
        "## Match Rate by Figure Type × Scenario",
        "",
        "| Figure Type | text_only | caption_only | vision_only | multimodal | Avg | Δ |",
        "|-------------|-----------|--------------|-------------|------------|-----|---|",
    ]

    for ft in figure_types:
        if ft not in judgments:
            continue
        row = [f"| {ft}"]
        row_values = []
        for s in scenarios:
            j = judgments[ft][s]
            pct = j['match_pct']
            row.append(f"{pct:.1f}% ({j['match']}/{j['total']})")
            row_values.append(pct)
        avg = sum(row_values) / len(row_values) if row_values else 0
        row.append(f"{avg:.1f}%")
        delta = judgments[ft]['multimodal']['match_pct'] - judgments[ft]['text_only']['match_pct']
        row.append(f"{delta:+.1f}pp |")
        lines.append(" | ".join(row))

    lines.extend([
        "",
        "## Soft Match Rate (Match + Partial)",
        "",
        "| Figure Type | text_only | caption_only | vision_only | multimodal | Δ |",
        "|-------------|-----------|--------------|-------------|------------|---|",
    ])

    for ft in figure_types:
        if ft not in judgments:
            continue
        row = [f"| {ft}"]
        for s in scenarios:
            j = judgments[ft][s]
            pct = j['soft_match_pct']
            soft = j['match'] + j['partial']
            row.append(f"{pct:.1f}% ({soft}/{j['total']})")
        delta = judgments[ft]['multimodal']['soft_match_pct'] - judgments[ft]['text_only']['soft_match_pct']
        row.append(f"{delta:+.1f}pp |")
        lines.append(" | ".join(row))

    # Context benefit
    lines.extend([
        "",
        "## Context Benefit for Feedback (Δ = multimodal - text_only)",
        "",
        "| Figure Type | Δ Match | Δ Soft Match | Helps? |",
        "|-------------|---------|--------------|--------|",
    ])

    for ft in figure_types:
        if ft not in judgments:
            continue
        delta_match = judgments[ft]['multimodal']['match_pct'] - judgments[ft]['text_only']['match_pct']
        delta_soft = judgments[ft]['multimodal']['soft_match_pct'] - judgments[ft]['text_only']['soft_match_pct']
        helps = "Yes" if delta_match > 0 else "No"
        lines.append(f"| {ft} | {delta_match:+.1f}pp | {delta_soft:+.1f}pp | {helps} |")

    # H4 verdict
    if 'table' in judgments and 'schematic' in judgments:
        table_avg = sum(judgments['table'][s]['match_pct'] for s in scenarios) / len(scenarios)
        schematic_avg = sum(judgments['schematic'][s]['match_pct'] for s in scenarios) / len(scenarios)
        h4_pass = table_avg > schematic_avg

        lines.extend([
            "",
            "## H4 Hypothesis Test (Feedback)",
            "",
            "**H4:** Tables easier than Schematics?",
            "",
            f"- Table avg (Match %): {table_avg:.1f}%",
            f"- Schematic avg (Match %): {schematic_avg:.1f}%",
            f"- **Result: {'PASS' if h4_pass else 'FAIL'}**",
            "",
        ])

    # Summary
    lines.extend([
        "## Summary",
        "",
    ])

    # Check overall trend
    any_helps = any(
        judgments[ft]['multimodal']['match_pct'] > judgments[ft]['text_only']['match_pct']
        for ft in figure_types if ft in judgments
    )

    if any_helps:
        lines.append("Visual context **helps** feedback quality for some figure types.")
    else:
        lines.append("Visual context does **not** help feedback quality for any figure type.")

    Path(output_path).write_text("\n".join(lines))
    print(f"\nSaved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze feedback by figure type using LLM judge")
    parser.add_argument("--max-per-cell", type=int, default=None,
                        help="Max examples per (figure_type, scenario) cell for smoke test")
    args = parser.parse_args()

    results_path = "data/eval/figure_type_analysis/figure_type_results.json"
    output_dir = Path("data/eval/figure_type_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(output_dir / "feedback_by_figure_type.md")

    client = build_client()

    print("Analyzing feedback quality by figure type using LLM judge...\n")

    judgments = analyze_feedback(results_path, client, max_per_cell=args.max_per_cell)
    print_results(judgments)
    save_report(judgments, output_path)

    # Save raw judgments
    judgments_path = str(output_dir / "feedback_judgments.json")
    with open(judgments_path, 'w') as f:
        json.dump(judgments, f, indent=2)
    print(f"Saved raw judgments to {judgments_path}")


if __name__ == "__main__":
    main()
