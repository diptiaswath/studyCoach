#!/usr/bin/env python3
"""Step 7: LLM Validator

Validates synthetic error examples using a second LLM (e.g., Gemini, Claude, or GPT-3.5).
Compares validator's classification with original classification to check agreement.

Output: data/validation_results.json

Usage:
  python src/validator.py data/spiqa_incongruent_samples.json \\
    --validator-model gemini-1.5-pro \\
    --output data/validation_results.json
"""

import json
import argparse
import base64
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI


client = OpenAI()

VALIDATOR_PROMPT = """You are a validator for an educational dataset. You are given:
1. A figure caption
2. A student's answer to a question about the figure
3. The correct answer

Your task: Determine if the student's answer is incorrect, and if so, what type of error it contains.

Respond in this exact format:
Verdict = [Correct|Partially Correct|Incorrect]
Error Category = [Omission|Factual|Conceptual|N/A]
Reasoning = [1-2 sentences explaining your classification]
"""


def to_data_url(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def parse_validator_output(text: str) -> Dict[str, str]:
    """Parse validator model output."""
    result = {
        'verdict': 'N/A',
        'error_category': 'N/A',
        'reasoning': ''
    }
    
    for line in text.split('\n'):
        if 'Verdict' in line and '=' in line:
            result['verdict'] = line.split('=')[1].strip()
        elif 'Error Category' in line and '=' in line:
            result['error_category'] = line.split('=')[1].strip()
        elif 'Reasoning' in line and '=' in line:
            result['reasoning'] = line.split('=', 1)[1].strip()
    
    return result


def validate_example(
    image_path: str,
    caption: str,
    question: str,
    student_answer: str,
    correct_answer: str,
    original_verdict: str,
    original_error_category: str,
    model: str = "gpt-4-turbo"
) -> Dict[str, Any]:
    """
    Validate a single example using LLM.
    
    Args:
        image_path: Path to figure image
        caption: Figure caption
        question: Question text
        student_answer: Student's answer to validate
        correct_answer: Ground truth answer
        original_verdict: Original classification
        original_error_category: Original error category
        model: LLM model to use for validation
    
    Returns:
        Validation result with agreement flag
    """
    if not Path(image_path).exists():
        return {
            'verdict': 'N/A',
            'error_category': 'N/A',
            'reasoning': 'Image not found',
            'original_verdict': original_verdict,
            'original_error_category': original_error_category,
            'agreement': False,
            'error': 'Image not found'
        }
    
    prompt = f"""{VALIDATOR_PROMPT}

Caption: {caption}

Question: {question}

Student's Answer: {student_answer}

Correct Answer: {correct_answer}
"""
    
    try:
        image_url = to_data_url(image_path)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        validator_output = parse_validator_output(response.choices[0].message.content)
        
        # Check agreement
        verdict_agreement = validator_output['verdict'].lower() == original_verdict.lower()
        error_agreement = validator_output['error_category'].lower() == original_error_category.lower()
        
        return {
            'verdict': validator_output['verdict'],
            'error_category': validator_output['error_category'],
            'reasoning': validator_output['reasoning'],
            'original_verdict': original_verdict,
            'original_error_category': original_error_category,
            'verdict_agreement': verdict_agreement,
            'error_category_agreement': error_agreement,
            'overall_agreement': verdict_agreement and error_agreement,
            'error': None
        }
    
    except Exception as e:
        return {
            'verdict': 'N/A',
            'error_category': 'N/A',
            'reasoning': '',
            'original_verdict': original_verdict,
            'original_error_category': original_error_category,
            'agreement': False,
            'error': str(e)
        }


def validate_samples(
    incongruent_json_path: Path,
    images_root: Path,
    output_path: Path,
    validator_model: str = "gpt-4-turbo",
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Validate incongruent samples.
    
    Args:
        incongruent_json_path: Path to spiqa_incongruent_samples.json
        images_root: Root directory for images
        output_path: Path to write validation results
        validator_model: LLM model to use
        max_samples: Max number of samples to validate (None = all)
    """
    with open(incongruent_json_path, 'r', encoding='utf-8') as f:
        incongruent_data = json.load(f)
    
    examples = incongruent_data.get('examples', [])
    if max_samples:
        examples = examples[:max_samples]
    
    validation_results = []
    agreement_count = 0
    verdict_agreements = 0
    error_agreements = 0
    
    for idx, example in enumerate(examples, 1):
        paper_id = example['paper_id']
        figure_ref = example['figure_reference']
        image_path = images_root / paper_id / figure_ref
        
        result = validate_example(
            image_path=str(image_path),
            caption=example['context']['caption'],
            question=example['question'],
            student_answer=example['user_answer'],
            correct_answer=example.get('correct_answer', ''),
            original_verdict=example['model_response']['verdict'],
            original_error_category=example['model_response']['error_category'],
            model=validator_model
        )
        
        result['example_id'] = example['id']
        validation_results.append(result)
        
        if result.get('overall_agreement'):
            agreement_count += 1
        if result.get('verdict_agreement'):
            verdict_agreements += 1
        if result.get('error_category_agreement'):
            error_agreements += 1
        
        if idx % 10 == 0:
            print(f"  Validated {idx}/{len(examples)}...")
    
    # Summary statistics
    agreement_rate = agreement_count / len(examples) if examples else 0
    verdict_rate = verdict_agreements / len(examples) if examples else 0
    error_rate = error_agreements / len(examples) if examples else 0
    
    output_data = {
        'metadata': {
            'validator_model': validator_model,
            'total_validated': len(examples),
            'overall_agreement_rate': round(agreement_rate, 3),
            'verdict_agreement_rate': round(verdict_rate, 3),
            'error_category_agreement_rate': round(error_rate, 3),
            'quality_gate_passed': agreement_rate >= 0.90
        },
        'results': validation_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Validation complete")
    print(f"📊 Results:")
    print(f"   Overall Agreement: {agreement_rate:.1%}")
    print(f"   Verdict Agreement: {verdict_rate:.1%}")
    print(f"   Error Category Agreement: {error_rate:.1%}")
    print(f"   Quality Gate (≥90%): {'✅ PASSED' if agreement_rate >= 0.90 else '❌ FAILED'}")
    print(f"📁 Output: {output_path}")
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate synthetic examples with second LLM")
    parser.add_argument("incongruent_json", help="Path to spiqa_incongruent_samples.json")
    parser.add_argument("--images-root", default="data/test-A/SPIQA_testA_Images",
                        help="Root directory for images")
    parser.add_argument("--output", default="data/validation_results.json",
                        help="Path to write validation results")
    parser.add_argument("--validator-model", default="gpt-4-turbo",
                        help="LLM model to use for validation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to validate (default: all)")
    args = parser.parse_args()
    
    incongruent_path = Path(args.incongruent_json)
    images_root = Path(args.images_root)
    output_path = Path(args.output)
    
    if not incongruent_path.exists():
        print(f"❌ Error: {incongruent_path} not found")
        exit(1)
    
    if not images_root.exists():
        print(f"❌ Error: {images_root} not found")
        exit(1)
    
    print(f"Validating samples with {args.validator_model}...")
    validate_samples(
        incongruent_path,
        images_root,
        output_path,
        validator_model=args.validator_model,
        max_samples=args.max_samples
    )
