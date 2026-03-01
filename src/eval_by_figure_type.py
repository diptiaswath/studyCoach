#!/usr/bin/env python3
"""Evaluation by Figure Type: Test H4 Hypothesis

Tests H4: "Tables should be easiest (explicit, localized information),
while architecture diagrams should be hardest (require understanding
spatial relationships between components)."

This script runs all 4 scenarios on figure-type subsets and calculates:
- Verdict Accuracy by figure type × scenario
- Context benefit (Δ = multimodal - text_only)
- H4 verdict (Tables > Schematics)

Figure Type Normalization:
- table: content_type="table" OR figure_type contains "table"
- plot: figure_type contains "plot", "graph", or "chart"
- schematic: figure_type contains "schematic", "diagram", or "architecture"
- other: everything else (including N/A, empty, or unrecognized types)

Usage:
    python src/eval_by_figure_type.py \
        --data data/test-A/SPIQA_testA_part1_output_latest.json \
        --images data/test-A/SPIQA_testA_Images \
        --output data/eval/figure_type_analysis

    # Smoke test with 3 examples per figure type:
    python src/eval_by_figure_type.py \
        --data data/test-A/SPIQA_testA_part1_output_latest.json \
        --images data/test-A/SPIQA_testA_Images \
        --output data/eval/figure_type_analysis \
        --max-per-type 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import parse_eval_output, to_data_url, SYSTEM_PROMPT

MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def build_client() -> openai.OpenAI:
    """Build Together.ai client."""
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TOGETHER_API_KEY environment variable is not set. "
            "Get a key at https://api.together.ai"
        )
    return openai.OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=api_key,
    )


def normalize_figure_type(figure_type: str, content_type: str) -> str:
    """Normalize figure_type to standard categories.

    Categories: table, plot, schematic, other
    """
    figure_type = (figure_type or "").strip().lower()
    content_type = (content_type or "").strip().lower()

    # Remove asterisks and extra whitespace
    figure_type = re.sub(r'\*+', '', figure_type).strip()

    # Tables: content_type=table OR figure_type contains "table"
    if content_type == "table" or "table" in figure_type:
        return "table"

    # Plots: figure_type contains "plot" or "graph" or "chart"
    if any(x in figure_type for x in ["plot", "graph", "chart"]):
        return "plot"

    # Schematics: figure_type contains "schematic" or "diagram" or "architecture"
    if any(x in figure_type for x in ["schematic", "diagram", "architecture"]):
        return "schematic"

    # N/A or empty - try to infer from content_type
    if figure_type in ("n/a", "") and content_type == "figure":
        return "other"

    return "other"


def load_all_examples(
    output_json_paths: List[str | Path],
    images_root: str | Path,
) -> List[Dict[str, Any]]:
    """Load ALL examples from augmented SPIQA output JSONs.

    Returns all examples with valid student answers, including figure_type.
    """
    images_root = Path(images_root)
    examples = []

    for path in output_json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for paper_key, paper in data.items():
            all_figures = paper.get("all_figures", {})
            for qa in paper.get("qa", []):
                student = (qa.get("student") or "").strip()
                if not student:
                    continue

                verdict = (qa.get("verdict") or "").strip()
                error_category = (qa.get("error_category") or "").strip()
                feedback = (qa.get("feedback") or "").strip()
                question = (qa.get("question") or "").strip()
                answer = (qa.get("answer") or "").strip()
                reference = (qa.get("reference") or "").strip()

                fig_details = all_figures.get(reference, {})
                caption = (fig_details.get("caption") or "").strip()
                raw_figure_type = (fig_details.get("figure_type") or "").strip()
                content_type = (fig_details.get("content_type") or "").strip()
                image_path = images_root / paper_key / reference

                # Normalize figure type
                figure_type = normalize_figure_type(raw_figure_type, content_type)

                examples.append({
                    "paper_id": paper_key,
                    "question": question,
                    "answer": answer,
                    "caption": caption,
                    "image_path": str(image_path),
                    "student": student,
                    "figure_type": figure_type,
                    "raw_figure_type": raw_figure_type,
                    "content_type": content_type,
                    "ground_truth": {
                        "verdict": verdict,
                        "error_category": error_category,
                        "feedback": feedback,
                    },
                })

    return examples


def group_by_figure_type(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group examples by figure type.

    Returns: {figure_type: [examples]}
    """
    by_figure_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for ex in examples:
        figure_type = ex["figure_type"]
        by_figure_type[figure_type].append(ex)

    return dict(by_figure_type)


def evaluate_text_only(client: openai.OpenAI, example: dict, timeout: int = 60) -> dict:
    """Scenario C1: text_only (Q + Student Answer)."""
    question = example["question"]
    student = example["student"]

    user_text = f"/no_think\nQuestion:\n{question}\nStudent Answer:\n{student}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ],
        timeout=timeout,
    )

    output_text = response.choices[0].message.content or ""
    return parse_eval_output(output_text), output_text


def evaluate_caption_only(client: openai.OpenAI, example: dict, timeout: int = 60) -> dict:
    """Scenario C2: caption_only (Q + Caption + Student Answer)."""
    caption = example["caption"]
    question = example["question"]
    student = example["student"]

    user_text = f"/no_think\nCaption:\n{caption}\nQuestion:\n{question}\nStudent Answer:\n{student}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ],
        timeout=timeout,
    )

    output_text = response.choices[0].message.content or ""
    return parse_eval_output(output_text), output_text


def evaluate_vision_only(client: openai.OpenAI, example: dict, timeout: int = 60) -> dict:
    """Scenario C3: vision_only (Q + Image + Student Answer)."""
    question = example["question"]
    student = example["student"]
    image_path = example["image_path"]

    data_url = to_data_url(image_path)
    text = f"/no_think\nQuestion:\n{question}\nStudent Answer:\n{student}"

    user_content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        timeout=timeout,
    )

    output_text = response.choices[0].message.content or ""
    return parse_eval_output(output_text), output_text


def evaluate_multimodal(client: openai.OpenAI, example: dict, timeout: int = 60) -> dict:
    """Scenario C4: multimodal (Q + Caption + Image + Student Answer)."""
    caption = example["caption"]
    question = example["question"]
    student = example["student"]
    image_path = example["image_path"]

    data_url = to_data_url(image_path)
    text = f"/no_think\nCaption:\n{caption}\nQuestion:\n{question}\nStudent Answer:\n{student}"

    user_content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        timeout=timeout,
    )

    output_text = response.choices[0].message.content or ""
    return parse_eval_output(output_text), output_text


SCENARIOS = {
    "text_only": evaluate_text_only,
    "caption_only": evaluate_caption_only,
    "vision_only": evaluate_vision_only,
    "multimodal": evaluate_multimodal,
}


def run_evaluation(
    client: openai.OpenAI,
    examples_by_type: Dict[str, List[Dict[str, Any]]],
    max_per_type: int | None = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Run all 4 scenarios on each figure type subset.

    Returns: {figure_type: {scenario: [results]}}
    """
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for figure_type, examples in examples_by_type.items():
        if max_per_type and len(examples) > max_per_type:
            examples = examples[:max_per_type]

        print(f"\n{'='*60}", flush=True)
        print(f"FIGURE TYPE: {figure_type.upper()} ({len(examples)} examples)", flush=True)
        print(f"{'='*60}", flush=True)

        results[figure_type] = {}

        for scenario_name, eval_func in SCENARIOS.items():
            print(f"\n  Scenario: {scenario_name}", flush=True)
            scenario_results = []

            for i, example in enumerate(examples, 1):
                print(f"    [{i}/{len(examples)}] paper={example['paper_id']} "
                      f"gt={example['ground_truth']['verdict']}", end="", flush=True)
                try:
                    predicted, raw_output = eval_func(client, example)
                    result = {
                        **example,
                        "predicted": predicted,
                        "raw_output": raw_output,
                    }
                    scenario_results.append(result)
                    print(f" -> {predicted['verdict']}", flush=True)
                except Exception as e:
                    print(f" ERROR: {e}", flush=True)
                    scenario_results.append({
                        **example,
                        "predicted": {"verdict": "", "error_category": "N/A", "feedback": "N/A"},
                        "raw_output": f"ERROR: {e}",
                    })

            results[figure_type][scenario_name] = scenario_results

    return results


def calculate_accuracy(results: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """Calculate verdict accuracy for a list of results.

    Returns: (correct_count, total_count, accuracy_percentage)
    """
    correct = sum(
        1 for r in results
        if r["predicted"]["verdict"].lower() == r["ground_truth"]["verdict"].lower()
    )
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0.0
    return correct, total, accuracy


def analyze_results(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """Analyze results and test H4 hypothesis.

    Returns analysis dict with:
    - accuracy_matrix: {figure_type: {scenario: accuracy}}
    - context_benefit: {figure_type: Δ}
    - h4: table > schematic?
    """
    accuracy_matrix: Dict[str, Dict[str, float]] = {}
    counts_matrix: Dict[str, Dict[str, Tuple[int, int]]] = {}

    for figure_type, scenarios in results.items():
        accuracy_matrix[figure_type] = {}
        counts_matrix[figure_type] = {}
        for scenario_name, scenario_results in scenarios.items():
            correct, total, accuracy = calculate_accuracy(scenario_results)
            accuracy_matrix[figure_type][scenario_name] = accuracy
            counts_matrix[figure_type][scenario_name] = (correct, total)

    # Calculate context benefit (Δ = multimodal - text_only)
    context_benefit: Dict[str, float] = {}
    for figure_type in accuracy_matrix:
        if "multimodal" in accuracy_matrix[figure_type] and "text_only" in accuracy_matrix[figure_type]:
            delta = accuracy_matrix[figure_type]["multimodal"] - accuracy_matrix[figure_type]["text_only"]
            context_benefit[figure_type] = delta

    # Test H4: table > schematic (averaged across scenarios)
    h4_passed = False
    table_avg = None
    schematic_avg = None
    if "table" in accuracy_matrix and "schematic" in accuracy_matrix:
        table_avg = sum(accuracy_matrix["table"].values()) / len(accuracy_matrix["table"])
        schematic_avg = sum(accuracy_matrix["schematic"].values()) / len(accuracy_matrix["schematic"])
        h4_passed = table_avg > schematic_avg

    return {
        "accuracy_matrix": accuracy_matrix,
        "counts_matrix": counts_matrix,
        "context_benefit": context_benefit,
        "h4": {
            "passed": h4_passed,
            "table_avg": table_avg,
            "schematic_avg": schematic_avg,
        },
    }


def print_summary(analysis: Dict[str, Any]) -> None:
    """Print summary tables to console."""
    print("\n" + "=" * 70)
    print("ACCURACY MATRIX (Figure Type × Scenario)")
    print("=" * 70)

    scenarios = ["text_only", "caption_only", "vision_only", "multimodal"]
    figure_types = ["table", "plot", "schematic", "other"]

    # Header
    print(f"{'Figure Type':<15}", end="")
    for s in scenarios:
        print(f"{s:>15}", end="")
    print(f"{'Avg':>10}")

    # Rows
    for ft in figure_types:
        if ft not in analysis["accuracy_matrix"]:
            continue
        print(f"{ft:<15}", end="")
        row_values = []
        for s in scenarios:
            acc = analysis["accuracy_matrix"][ft].get(s, 0)
            counts = analysis["counts_matrix"][ft].get(s, (0, 0))
            print(f"{acc:>12.1f}%", end="  ")
            row_values.append(acc)
        avg = sum(row_values) / len(row_values) if row_values else 0
        print(f"{avg:>8.1f}%")

    print("\n" + "=" * 70)
    print("CONTEXT BENEFIT (Δ = multimodal - text_only)")
    print("=" * 70)

    for ft in figure_types:
        if ft in analysis["context_benefit"]:
            delta = analysis["context_benefit"][ft]
            sign = "+" if delta >= 0 else ""
            print(f"  {ft:<15}: {sign}{delta:.1f}pp")

    print("\n" + "=" * 70)
    print("H4 HYPOTHESIS TEST")
    print("=" * 70)

    h4 = analysis["h4"]

    print(f"\nH4: Tables easier than Schematics?")
    print(f"  Table avg:      {h4['table_avg']:.1f}%" if h4['table_avg'] else "  Table avg:      N/A")
    print(f"  Schematic avg:  {h4['schematic_avg']:.1f}%" if h4['schematic_avg'] else "  Schematic avg:  N/A")
    print(f"  Result:         {'PASS' if h4['passed'] else 'FAIL'}")

    print(f"\n{'='*70}")


def save_markdown_report(
    analysis: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save markdown report."""
    report_path = output_dir / "figure_type_analysis.md"

    scenarios = ["text_only", "caption_only", "vision_only", "multimodal"]
    figure_types = ["table", "plot", "schematic", "other"]

    lines = [
        "# Figure Type Analysis: H4 Hypothesis Test",
        "",
        "## Hypothesis",
        "",
        "**H4:** Tables should be easiest (explicit, localized information),",
        "while architecture diagrams/schematics should be hardest (require",
        "understanding spatial relationships between components).",
        "",
        "## Accuracy Matrix",
        "",
        "| Figure Type | text_only | caption_only | vision_only | multimodal | Avg |",
        "|-------------|-----------|--------------|-------------|------------|-----|",
    ]

    for ft in figure_types:
        if ft not in analysis["accuracy_matrix"]:
            continue
        row = [f"| {ft}"]
        row_values = []
        for s in scenarios:
            acc = analysis["accuracy_matrix"][ft].get(s, 0)
            counts = analysis["counts_matrix"][ft].get(s, (0, 0))
            row.append(f"{acc:.1f}% ({counts[0]}/{counts[1]})")
            row_values.append(acc)
        avg = sum(row_values) / len(row_values) if row_values else 0
        row.append(f"{avg:.1f}%")
        row.append("|")
        lines.append(" | ".join(row))

    lines.extend([
        "",
        "## Context Benefit (Δ = multimodal - text_only)",
        "",
        "| Figure Type | Δ (pp) | Interpretation |",
        "|-------------|--------|----------------|",
    ])

    for ft in figure_types:
        if ft in analysis["context_benefit"]:
            delta = analysis["context_benefit"][ft]
            sign = "+" if delta >= 0 else ""
            effect = "helps" if delta > 0 else "hurts" if delta < 0 else "neutral"
            lines.append(f"| {ft} | {sign}{delta:.1f} | Visual context **{effect}** |")

    h4 = analysis["h4"]

    lines.extend([
        "",
        "## H4 Hypothesis Test",
        "",
        "**H4:** Tables > Schematics?",
        "",
        f"- Table avg: {h4['table_avg']:.1f}%" if h4['table_avg'] else "- Table avg: N/A",
        f"- Schematic avg: {h4['schematic_avg']:.1f}%" if h4['schematic_avg'] else "- Schematic avg: N/A",
        f"- **Result: {'PASS' if h4['passed'] else 'FAIL'}**",
        "",
        "---",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate by figure type to test H4 hypothesis"
    )
    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="Path(s) to *_output.json augmented dataset files",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Root directory containing per-paper image subdirectories",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=None,
        help="Maximum examples per figure type (for smoke testing)",
    )
    parser.add_argument(
        "--skip-types",
        nargs="+",
        default=[],
        help="Figure types to skip (e.g., --skip-types schematic table)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing results JSON to resume from",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = build_client()

    # Step 1: Load all examples
    print("Step 1: Loading all examples...", flush=True)
    all_examples = load_all_examples(args.data, args.images)
    print(f"  Loaded {len(all_examples)} total examples", flush=True)

    # Step 2: Group by figure type
    print("\nStep 2: Grouping by figure type...", flush=True)
    examples_by_type = group_by_figure_type(all_examples)
    total = sum(len(v) for v in examples_by_type.values())
    print(f"  Total examples: {total}", flush=True)
    for ft, exs in sorted(examples_by_type.items()):
        pct = len(exs) / total * 100 if total > 0 else 0
        print(f"    {ft}: {len(exs)} ({pct:.0f}%)", flush=True)

    # Skip specified figure types
    if args.skip_types:
        print(f"\nSkipping figure types: {args.skip_types}", flush=True)
        examples_by_type = {k: v for k, v in examples_by_type.items() if k not in args.skip_types}

    # Load existing results if resuming
    existing_results = {}
    if args.resume and Path(args.resume).exists():
        print(f"\nLoading existing results from {args.resume}...", flush=True)
        with open(args.resume) as f:
            existing_data = json.load(f)
            existing_results = existing_data.get("results", {})
        print(f"  Found results for: {list(existing_results.keys())}", flush=True)

    # Steps 3-4: Run evaluation
    print("\nSteps 3-4: Running 4 scenarios per figure type...", flush=True)
    results = run_evaluation(client, examples_by_type, max_per_type=args.max_per_type)

    # Merge with existing results
    if existing_results:
        for ft, scenarios in existing_results.items():
            if ft not in results:
                results[ft] = scenarios
                print(f"  Using existing results for {ft}", flush=True)

    # Step 5-6: Calculate accuracy and context benefit
    print("\nSteps 5-6: Calculating accuracy and context benefit...")
    analysis = analyze_results(results)

    # Step 7: Test H4 and print summary
    print("\nStep 7: Testing H4 hypothesis...")
    print_summary(analysis)

    # Step 8: Save results
    print("\nStep 8: Saving results...")

    # Save full results JSON
    results_path = output_dir / "figure_type_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        # Convert results to serializable format
        serializable = {}
        for ft, scenarios in results.items():
            serializable[ft] = {}
            for s, res_list in scenarios.items():
                serializable[ft][s] = res_list
        json.dump({
            "results": serializable,
            "analysis": analysis,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved full results to {results_path}")

    # Save markdown report
    save_markdown_report(analysis, output_dir)


if __name__ == "__main__":
    main()
