#!/usr/bin/env python3
"""Generate Congruent Samples

Creates "correct" user answer samples from SPIQA ground truth answers.
For each QA pair in Test-A, creates a 4-tuple:
  (context, question, user_answer="ground_truth_answer", model_response="N/A")

Output: data/spiqa_congruent_samples.json

Usage:
  python src/generate_congruent_samples.py data/SPIQA_testA_part1.json

NOTE: Uses same normalize_figure_category function as icl.py for consistency.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Import shared function from icl.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from icl import normalize_figure_category


def generate_congruent_samples(json_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Generate congruent samples from SPIQA Test-A JSON.
    
    Args:
        json_path: Path to SPIQA_testA_part*.json
        output_path: Path to write spiqa_congruent_samples.json
    
    Returns:
        Dictionary with metadata and congruent examples
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    congruent_samples = []
    error_distribution = {'omission': 0, 'factual': 0, 'conceptual': 0}
    figure_distribution = {'plot': 0, 'figure': 0, 'table': 0}
    
    example_id = 0
    for paper_key in sorted(data.keys()):
        paper = data[paper_key]
        all_figures = paper.get('all_figures', {})
        qa_list = paper.get('qa', [])
        
        for qa_idx, qa in enumerate(qa_list, 1):
            question = (qa.get('question', '') or '').strip()
            figure_ref = (qa.get('reference', '') or '').strip()
            correct_answer = (qa.get('answer', '') or '').strip()
            
            # Get figure details
            figure_details = all_figures.get(figure_ref, None)
            if not figure_details or not figure_ref:
                continue  # Skip if no figure reference
            
            caption = figure_details.get('caption', '').strip()
            content_type = figure_details.get('content_type', '')
            figure_type = figure_details.get('figure_type', '')
            figure_category = normalize_figure_category(figure_type, content_type)
            
            # Create 4-tuple
            example = {
                'id': f"{paper_key}_qa_{qa_idx}_congruent",
                'paper_id': paper_key,
                'qa_index': qa_idx,
                'figure_reference': figure_ref,
                'figure_category': figure_category,
                'context': {
                    'caption': caption,
                    'image_path': figure_ref
                },
                'question': question,
                'user_answer': correct_answer,  # Ground truth as "correct" answer
                'model_response': {
                    'verdict': 'Correct',
                    'error_category': 'N/A',
                    'feedback': 'N/A'
                }
            }
            
            congruent_samples.append(example)
            figure_distribution[figure_category] = figure_distribution.get(figure_category, 0) + 1
            example_id += 1
    
    # Create output with metadata
    output_data = {
        'metadata': {
            'source': 'SPIQA Test-A',
            'dataset_type': 'congruent_samples',
            'total_examples': len(congruent_samples),
            'figure_distribution': figure_distribution,
        },
        'examples': congruent_samples
    }
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f" Generated {len(congruent_samples)} congruent samples")
    print(f" Figure distribution:")
    for fig_type, count in figure_distribution.items():
        print(f"   {fig_type}: {count}")
    print(f" Output: {output_path}")
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate congruent samples from SPIQA Test-A")
    parser.add_argument("json_path", help="Path to SPIQA_testA JSON file")
    parser.add_argument("--output", default="data/spiqa_congruent_samples.json",
                        help="Path to write output JSON")
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    output_path = Path(args.output)
    
    if not json_path.exists():
        print(f"  Error: {json_path} not found")
        exit(1)
    
    generate_congruent_samples(json_path, output_path)
