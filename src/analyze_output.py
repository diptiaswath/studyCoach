#!/usr/bin/env python3
"""Analyze SPIQA+ augmented output JSON files.

Usage:
  # Single file analysis
  python src/analyze_output.py data/test-A/SPIQA_testA_part1_output_latest.json

  # Compare two files (before vs after)
  python src/analyze_output.py data/test-A/SPIQA_testA_part1_output.json data/test-A/SPIQA_testA_part1_output_latest.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_file(data: dict) -> dict:
    """Extract statistics from augmented SPIQA+ JSON."""
    verdicts = Counter()
    error_categories = Counter()
    total_qa = 0

    for paper_key, paper in data.items():
        qa_list = paper.get("qa", [])
        for qa in qa_list:
            total_qa += 1
            verdict = qa.get("verdict", "").lower().strip()
            error_cat = qa.get("error_category", "").lower().strip()

            verdicts[verdict] += 1
            error_categories[error_cat] += 1

    # Calculate percentages for non-correct errors
    non_correct_total = total_qa - verdicts.get("correct", 0)
    error_pcts = {}
    for cat in ["factual", "omission", "conceptual"]:
        count = error_categories.get(cat, 0)
        pct = (count / non_correct_total * 100) if non_correct_total > 0 else 0
        error_pcts[cat] = (count, pct)

    return {
        "total_qa": total_qa,
        "verdicts": dict(verdicts),
        "error_categories": dict(error_categories),
        "error_pcts": error_pcts,
        "non_correct_total": non_correct_total
    }


def print_single_analysis(stats: dict, filename: str):
    """Print analysis for a single file."""
    print(f"\n{'='*60}")
    print(f"Analysis: {filename}")
    print(f"{'='*60}\n")

    print(f"Total QA pairs: {stats['total_qa']}\n")

    print("Verdict Distribution:")
    print("-" * 30)
    for verdict in ["correct", "partially correct", "incorrect"]:
        count = stats["verdicts"].get(verdict, 0)
        pct = count / stats["total_qa"] * 100 if stats["total_qa"] > 0 else 0
        print(f"  {verdict:<20} {count:>4} ({pct:>5.1f}%)")

    print(f"\nError Category Distribution (n={stats['non_correct_total']} non-correct):")
    print("-" * 30)
    for cat in ["factual", "omission", "conceptual"]:
        count, pct = stats["error_pcts"].get(cat, (0, 0))
        print(f"  {cat:<20} {count:>4} ({pct:>5.1f}%)")

    print()


def print_comparison(stats1: dict, stats2: dict, name1: str, name2: str):
    """Print side-by-side comparison of two files."""
    print(f"\n{'='*70}")
    print("COMPARISON: Before vs After")
    print(f"{'='*70}\n")

    print(f"  Before: {name1}")
    print(f"  After:  {name2}\n")

    # Verdict comparison
    print("Verdict Distribution:")
    print("-" * 50)
    print(f"  {'Verdict':<20} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for verdict in ["correct", "partially correct", "incorrect"]:
        c1 = stats1["verdicts"].get(verdict, 0)
        c2 = stats2["verdicts"].get(verdict, 0)
        diff = c2 - c1
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"  {verdict:<20} {c1:>10} {c2:>10} {diff_str:>10}")

    # Error category comparison
    print(f"\nError Category Distribution:")
    print("-" * 50)
    print(f"  {'Category':<20} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for cat in ["factual", "omission", "conceptual"]:
        c1, p1 = stats1["error_pcts"].get(cat, (0, 0))
        c2, p2 = stats2["error_pcts"].get(cat, (0, 0))
        diff = c2 - c1
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"  {cat:<20} {c1:>4} ({p1:>4.0f}%) {c2:>4} ({p2:>4.0f}%) {diff_str:>10}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    factual_before = stats1["error_pcts"].get("factual", (0, 0))
    factual_after = stats2["error_pcts"].get("factual", (0, 0))
    conceptual_before = stats1["error_pcts"].get("conceptual", (0, 0))
    conceptual_after = stats2["error_pcts"].get("conceptual", (0, 0))

    conceptual_increase = conceptual_after[0] - conceptual_before[0]
    conceptual_pct_change = (conceptual_after[0] / conceptual_before[0] * 100 - 100) if conceptual_before[0] > 0 else 0

    print(f"  - Factual errors:    {factual_before[1]:.0f}% -> {factual_after[1]:.0f}%")
    print(f"  - Conceptual errors: {conceptual_before[1]:.0f}% -> {conceptual_after[1]:.0f}% (+{conceptual_increase}, {conceptual_pct_change:+.0f}% increase)")
    print()

    # Determine if improvement
    if factual_after[1] < factual_before[1] and conceptual_after[0] > conceptual_before[0]:
        print("  Verdict: IMPROVEMENT - More balanced error distribution")
        print("           Reduced factual bias, increased conceptual coverage")
    elif factual_after[1] == factual_before[1] and conceptual_after[0] == conceptual_before[0]:
        print("  Verdict: NO CHANGE - Error distribution unchanged")
    else:
        print("  Verdict: MIXED - Review changes manually")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SPIQA+ augmented output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file analysis
  python src/analyze_output.py data/test-A/SPIQA_testA_part1_output_latest.json

  # Compare two files
  python src/analyze_output.py data/test-A/SPIQA_testA_part1_output.json data/test-A/SPIQA_testA_part1_output_latest.json
        """
    )
    parser.add_argument("files", nargs="+", help="One or two JSON files to analyze")
    args = parser.parse_args()

    if len(args.files) == 1:
        # Single file analysis
        path = Path(args.files[0])
        data = load_json(path)
        stats = analyze_file(data)
        print_single_analysis(stats, path.name)

    elif len(args.files) == 2:
        # Comparison mode
        path1, path2 = Path(args.files[0]), Path(args.files[1])
        data1, data2 = load_json(path1), load_json(path2)
        stats1, stats2 = analyze_file(data1), analyze_file(data2)

        print_single_analysis(stats1, path1.name)
        print_single_analysis(stats2, path2.name)
        print_comparison(stats1, stats2, path1.name, path2.name)

    else:
        parser.error("Please provide 1 or 2 JSON files")


if __name__ == "__main__":
    main()
