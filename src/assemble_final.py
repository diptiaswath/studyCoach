#!/usr/bin/env python3
"""Steps 9-10: Quality Gate & Final Assembly

Step 9: Validates that LLM agreement >= 90% on validation set
Step 10: Assembles final SPIQA+ dataset from validated examples

Usage:
  python src/assemble_final.py \\
    --incongruent data/spiqa_incongruent_samples.json \\
    --validation data/validation_results.json \\
    --output data/SPIQA_plus_final.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def check_quality_gate(validation_results: Dict[str, Any]) -> bool:
    """
    Check if quality gate criteria are met (≥90% agreement).
    
    Args:
        validation_results: Validation results from validator.py
    
    Returns:
        True if quality gate passed, False otherwise
    """
    agreement_rate = validation_results['metadata']['overall_agreement_rate']
    passed = agreement_rate >= 0.90
    
    print(f"\n{'='*60}")
    print(f"QUALITY GATE CHECK")
    print(f"{'='*60}")
    print(f"Overall Agreement Rate: {agreement_rate:.1%}")
    print(f"Threshold: 90.0%")
    print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"{'='*60}\n")
    
    if not passed:
        print("⚠️  Quality gate FAILED. Recommendations:")
        print("   1. Review failed examples (see validation_results.json)")
        print("   2. Improve seed exemplars (Step 4)")
        print("   3. Regenerate samples (Step 6)")
        print("   4. Re-validate (Step 7)")
    
    return passed


def assemble_final_dataset(
    incongruent_path: Path,
    validation_path: Path,
    output_path: Path,
    enforce_quality_gate: bool = True
) -> Dict[str, Any]:
    """
    Assemble final SPIQA+ dataset from validated examples.
    
    Args:
        incongruent_path: Path to spiqa_incongruent_samples.json
        validation_path: Path to validation_results.json
        output_path: Path to write SPIQA_plus_final.json
        enforce_quality_gate: If True, only include examples that passed validation
    
    Returns:
        Final SPIQA+ dataset
    """
    # Load data
    with open(incongruent_path, 'r', encoding='utf-8') as f:
        incongruent_data = json.load(f)
    
    with open(validation_path, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
    
    # Check quality gate
    if enforce_quality_gate:
        passed = check_quality_gate(validation_data)
        if not passed:
            print("❌ Aborting assembly due to quality gate failure")
            return {}
    
    # Create mapping of example_id -> validation result
    validation_map = {
        result['example_id']: result
        for result in validation_data.get('results', [])
    }
    
    # Assemble final examples
    final_examples = []
    error_distribution = defaultdict(int)
    figure_distribution = defaultdict(int)
    included_count = 0
    excluded_count = 0
    
    for example in incongruent_data.get('examples', []):
        example_id = example['id']
        validation_result = validation_map.get(example_id)
        
        # Check if validation passed
        if enforce_quality_gate and validation_result:
            if not validation_result.get('overall_agreement', False):
                excluded_count += 1
                continue
        
        # Add validation metadata to example
        if validation_result:
            example['validation'] = {
                'validator_verdict': validation_result.get('verdict'),
                'validator_error_category': validation_result.get('error_category'),
                'validator_reasoning': validation_result.get('reasoning'),
                'agreement': validation_result.get('overall_agreement', False),
                'error': validation_result.get('error')
            }
        else:
            example['validation'] = {
                'validator_verdict': 'N/A',
                'validator_error_category': 'N/A',
                'validator_reasoning': '',
                'agreement': False,
                'error': 'No validation result'
            }
        
        final_examples.append(example)
        
        # Track distributions
        error_cat = example['model_response'].get('error_category', 'N/A')
        fig_type = example.get('figure_category', 'unknown')
        error_distribution[error_cat] += 1
        figure_distribution[fig_type] += 1
        included_count += 1
    
    # Create final dataset with metadata
    final_dataset = {
        'metadata': {
            'source': 'SPIQA Test-A',
            'dataset_type': 'SPIQA+',
            'total_examples': len(final_examples),
            'examples_included': included_count,
            'examples_excluded': excluded_count,
            'validator_model': validation_data['metadata'].get('validator_model', 'N/A'),
            'overall_agreement_rate': validation_data['metadata'].get('overall_agreement_rate', 0),
            'quality_gate_passed': validation_data['metadata'].get('quality_gate_passed', False),
            'error_distribution': dict(error_distribution),
            'figure_distribution': dict(figure_distribution)
        },
        'examples': final_examples
    }
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"FINAL SPIQA+ DATASET ASSEMBLED")
    print(f"{'='*60}")
    print(f"Total Examples: {len(final_examples)}")
    print(f"Examples Included: {included_count}")
    print(f"Examples Excluded: {excluded_count}")
    print(f"\n📊 Error Distribution:")
    for error_type, count in sorted(error_distribution.items()):
        percentage = (count / len(final_examples) * 100) if final_examples else 0
        print(f"   {error_type}: {count} ({percentage:.1f}%)")
    print(f"\n📊 Figure Distribution:")
    for fig_type, count in sorted(figure_distribution.items()):
        percentage = (count / len(final_examples) * 100) if final_examples else 0
        print(f"   {fig_type}: {count} ({percentage:.1f}%)")
    print(f"\n📁 Output: {output_path}")
    print(f"{'='*60}\n")
    
    return final_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble final SPIQA+ dataset")
    parser.add_argument("--incongruent", default="data/spiqa_incongruent_samples.json",
                        help="Path to spiqa_incongruent_samples.json")
    parser.add_argument("--validation", default="data/validation_results.json",
                        help="Path to validation_results.json")
    parser.add_argument("--output", default="data/SPIQA_plus_final.json",
                        help="Path to write final SPIQA+ dataset")
    parser.add_argument("--skip-quality-gate", action="store_true",
                        help="Skip quality gate check (include all examples)")
    args = parser.parse_args()
    
    incongruent_path = Path(args.incongruent)
    validation_path = Path(args.validation)
    output_path = Path(args.output)
    
    if not incongruent_path.exists():
        print(f"❌ Error: {incongruent_path} not found")
        exit(1)
    
    if not validation_path.exists():
        print(f"❌ Error: {validation_path} not found")
        exit(1)
    
    assemble_final_dataset(
        incongruent_path,
        validation_path,
        output_path,
        enforce_quality_gate=not args.skip_quality_gate
    )
