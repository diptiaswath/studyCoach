#!/usr/bin/env python3
"""Dataset Quality Validation Script

Performs human validation of SPIQA+ synthetic student answers and feedback.
Aligns with Phase 3 of the dataset construction pipeline:
  - Human spot-check (~50-100 samples)
  - Analyze error distribution
  - Quality gate (≥90% agreement)

Usage:
  python src/validate_dataset.py --data data/test-A/SPIQA_testA_part1_output_latest.json --samples 20
  python src/validate_dataset.py --data data/test-A/SPIQA_testA_part1_output_latest.json --samples 20 --output results/validation_results.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load dataset and flatten QA pairs."""
    data = json.load(open(data_path))

    all_qa = []
    for paper_id, paper in data.items():
        for qa in paper['qa']:
            qa['paper_id'] = paper_id
            all_qa.append(qa)

    return all_qa


def analyze_distribution(all_qa: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze error type and verdict distribution."""
    by_error = {}
    by_verdict = {}

    for qa in all_qa:
        cat = qa.get('error_category', 'unknown').lower()
        verdict = qa.get('verdict', 'unknown').lower()

        by_error[cat] = by_error.get(cat, 0) + 1
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1

    return {'by_error': by_error, 'by_verdict': by_verdict}


def stratified_sample(all_qa: List[Dict[str, Any]], n_per_category: int = 5, seed: int = 42) -> Dict[str, List[Dict]]:
    """Get stratified sample across error categories."""
    random.seed(seed)

    samples = {
        'factual': [],
        'conceptual': [],
        'omission': [],
        'correct': []
    }

    # Shuffle to get random samples
    shuffled = all_qa.copy()
    random.shuffle(shuffled)

    for qa in shuffled:
        cat = qa.get('error_category', '').lower()
        verdict = qa.get('verdict', '').lower()

        if cat == 'factual' and len(samples['factual']) < n_per_category:
            samples['factual'].append(qa)
        elif cat == 'conceptual' and len(samples['conceptual']) < n_per_category:
            samples['conceptual'].append(qa)
        elif cat == 'omission' and len(samples['omission']) < n_per_category:
            samples['omission'].append(qa)
        elif cat == 'n/a' and verdict == 'correct' and len(samples['correct']) < n_per_category:
            samples['correct'].append(qa)

    return samples


def display_sample(idx: int, qa: Dict[str, Any]) -> None:
    """Display a single sample for validation."""
    print(f"\n{'='*80}")
    print(f"SAMPLE {idx}")
    print(f"{'='*80}")
    print(f"Paper: {qa['paper_id']}")
    print(f"Reference: {qa['reference']}")
    print(f"Verdict: {qa['verdict']}")
    print(f"Error Category: {qa['error_category']}")
    print(f"\nQUESTION:\n{qa['question']}")
    print(f"\nGROUND TRUTH ANSWER:\n{qa['answer']}")
    print(f"\nSTUDENT ANSWER:\n{qa['student']}")
    print(f"\nFEEDBACK:\n{qa['feedback']}")


def get_validation_input(idx: int) -> Dict[str, bool]:
    """Get human validation input for a sample."""
    print(f"\n--- VALIDATION (Sample {idx}) ---")

    while True:
        verdict_ok = input("Verdict correct? (y/n/s=skip): ").strip().lower()
        if verdict_ok in ['y', 'n', 's']:
            break
        print("Please enter y, n, or s")

    if verdict_ok == 's':
        return {'skipped': True}

    while True:
        category_ok = input("Error category correct? (y/n): ").strip().lower()
        if category_ok in ['y', 'n']:
            break
        print("Please enter y or n")

    while True:
        feedback_ok = input("Feedback accurate & useful? (y/n): ").strip().lower()
        if feedback_ok in ['y', 'n']:
            break
        print("Please enter y or n")

    notes = input("Notes (optional, press Enter to skip): ").strip()

    return {
        'skipped': False,
        'verdict_ok': verdict_ok == 'y',
        'category_ok': category_ok == 'y',
        'feedback_ok': feedback_ok == 'y',
        'notes': notes
    }


def calculate_agreement(results: List[Dict]) -> Dict[str, float]:
    """Calculate agreement rates."""
    valid_results = [r for r in results if not r.get('skipped', False)]

    if not valid_results:
        return {'verdict': 0, 'category': 0, 'feedback': 0, 'overall': 0}

    n = len(valid_results)
    verdict_agree = sum(1 for r in valid_results if r['verdict_ok'])
    category_agree = sum(1 for r in valid_results if r['category_ok'])
    feedback_agree = sum(1 for r in valid_results if r['feedback_ok'])
    overall_agree = sum(1 for r in valid_results if r['verdict_ok'] and r['category_ok'] and r['feedback_ok'])

    return {
        'verdict': verdict_agree / n * 100,
        'category': category_agree / n * 100,
        'feedback': feedback_agree / n * 100,
        'overall': overall_agree / n * 100,
        'n_validated': n,
        'n_skipped': len(results) - n
    }


def run_validation(data_path: str, n_samples: int = 20, seed: int = 42) -> Dict[str, Any]:
    """Run full validation workflow."""
    # Load data
    print(f"Loading data from {data_path}...")
    all_qa = load_dataset(data_path)
    print(f"Total QA pairs: {len(all_qa)}")

    # Analyze distribution
    print("\n" + "="*80)
    print("ERROR DISTRIBUTION")
    print("="*80)
    dist = analyze_distribution(all_qa)

    print("\nBy error category:")
    for cat, count in sorted(dist['by_error'].items()):
        print(f"  {cat}: {count}")

    print("\nBy verdict:")
    for verdict, count in sorted(dist['by_verdict'].items()):
        print(f"  {verdict}: {count}")

    # Stratified sample
    n_per_cat = n_samples // 4
    samples = stratified_sample(all_qa, n_per_category=n_per_cat, seed=seed)

    total_samples = sum(len(v) for v in samples.values())
    print(f"\nSampled {total_samples} examples ({n_per_cat} per category)")

    # Validation loop
    print("\n" + "="*80)
    print("STARTING VALIDATION")
    print("="*80)
    print("For each sample, you'll be asked to validate:")
    print("  1. Is the verdict correct?")
    print("  2. Is the error category correct?")
    print("  3. Is the feedback accurate and useful?")
    print("\nPress Enter to begin...")
    input()

    results = []
    idx = 1

    for category in ['factual', 'conceptual', 'omission', 'correct']:
        print(f"\n{'#'*80}")
        print(f"# CATEGORY: {category.upper()} ({len(samples[category])} samples)")
        print(f"{'#'*80}")

        for qa in samples[category]:
            display_sample(idx, qa)
            validation = get_validation_input(idx)
            validation['sample_idx'] = idx
            validation['category'] = category
            validation['paper_id'] = qa['paper_id']
            validation['reference'] = qa['reference']
            results.append(validation)
            idx += 1

    # Calculate agreement
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    agreement = calculate_agreement(results)

    print(f"\nSamples validated: {agreement['n_validated']}")
    print(f"Samples skipped: {agreement['n_skipped']}")
    print(f"\nAgreement rates:")
    print(f"  Verdict:        {agreement['verdict']:.1f}%")
    print(f"  Error Category: {agreement['category']:.1f}%")
    print(f"  Feedback:       {agreement['feedback']:.1f}%")
    print(f"  Overall:        {agreement['overall']:.1f}%")

    # Quality gate
    QUALITY_THRESHOLD = 90.0
    passed = agreement['overall'] >= QUALITY_THRESHOLD

    print(f"\n{'='*80}")
    print(f"QUALITY GATE: {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"{'='*80}")
    print(f"Threshold: ≥{QUALITY_THRESHOLD}%")
    print(f"Actual:    {agreement['overall']:.1f}%")

    return {
        'distribution': dist,
        'results': results,
        'agreement': agreement,
        'quality_gate_passed': passed
    }


def main():
    parser = argparse.ArgumentParser(description='Validate SPIQA+ dataset quality')
    parser.add_argument('--data', required=True, help='Path to dataset JSON file')
    parser.add_argument('--samples', type=int, default=20, help='Number of samples to validate (default: 20)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling (default: 42)')
    parser.add_argument('--output', help='Path to save validation results JSON')

    args = parser.parse_args()

    results = run_validation(args.data, n_samples=args.samples, seed=args.seed)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
