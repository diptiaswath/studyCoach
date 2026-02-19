# Phase 2 & 3 Workflow: Generating and Validating SPIQA+

This document describes how to execute Steps 4-10 to generate and validate the SPIQA+ dataset.

## Overview

```
SPIQA Test-A (666 QAs)
        ↓
   [Step 4] Hand-curate 20-30 seed exemplars
        ↓
   [Step 5] Generate congruent samples (666 correct answers)
        ↓
   [Step 6] Generate incongruent samples (666-1332 wrong answers)
        ↓
   [Step 7] LLM Validator (verify error presence)
        ↓
   [Step 8] Human Spot-Check (~50-100 samples)
        ↓
   [Step 9] Quality Gate (≥90% agreement)
        ↓
   [Step 10] Assemble Final SPIQA+ Dataset (~200-400 examples)
```

---

## Step-by-Step Execution

### Step 4: Hand-Curate ICL Seed Set (2-3 hours manual)

**Goal**: Generate 20-30 high-quality exemplars (5-8 per error category)

**Process**:

1. **Run seed.py multiple times to generate candidates**:
   ```bash
   # Currently generates 1 OMISSION example
   python src/seed.py
   ```

2. **Uncomment additional chart examples in seed.py** (lines 197-210):
   ```python
   # Uncomment to generate FACTUAL error
   chart_incorrect_factual = generate_seed_example(...)
   
   # Uncomment to generate CONCEPTUAL error
   chart_incorrect_conceptual = generate_seed_example(...)
   ```

3. **For each figure type (plot, figure, table), generate 5-8 examples**:
   - Define new exemplar tuples in seed.py `__main__` block
   - Run seed.py and collect good outputs
   - Copy best examples to `data/seed_exemplars.json`

4. **Organize into seed_exemplars.json**:
   ```json
   {
     "plot": {
       "omission": [
         {
           "user_prompt": "Caption: ...\nQuestion: ...",
           "assistant_response": "Student: ...\n\nAgent: Verdict = ...",
           "image_path": "test-A/SPIQA_testA_Images/.../figure.png"
         },
         ...
       ],
       "factual": [...],
       "conceptual": [...]
     },
     "figure": {...},
     "table": {...}
   }
   ```

**Deliverable**: `data/seed_exemplars.json` (20-30 curated examples)

---

### Step 5: Generate Congruent Samples (0.5 hours automated)

**Goal**: Create "correct" user answer samples from SPIQA ground truth

**Command**:
```bash
python src/generate_congruent_samples.py data/SPIQA_testA_part1.json \
  --output data/spiqa_congruent_samples.json
```

**What it does**:
- Reads SPIQA Test-A JSON (666 QAs)
- For each QA, uses ground truth answer as "correct" user answer
- Creates 4-tuple: (context, question, user_answer, model_response="N/A")
- Outputs ~666 congruent examples

**Deliverable**: `data/spiqa_congruent_samples.json` (~666 examples)

---

### Step 6: ICL-Based Synthetic Generation (4-6 hours, API calls)

**Goal**: Generate 1-2 incongruent (wrong) answers per QA using seed exemplars

**Workflow**:

1. **Modify icl.py to use seed_exemplars.json** instead of hardcoded exemplars:
   ```python
   # Load seed exemplars
   with open('data/seed_exemplars.json', 'r') as f:
       exemplars = json.load(f)
   
   # Run inference on all Test-A QAs
   updated_json = run_inference(
       'data/SPIQA_testA_part1.json',
       'data/test-A/SPIQA_testA_Images',
       exemplars=exemplars
   )
   ```

2. **Run on each SPIQA_testA_part file**:
   ```bash
   python src/icl.py data/SPIQA_testA_part1.json data/test-A/SPIQA_testA_Images \
     --output data/spiqa_incongruent_samples_part1.json
   
   python src/icl.py data/SPIQA_testA_part2.json data/test-A/SPIQA_testA_Images \
     --output data/spiqa_incongruent_samples_part2.json
   
   # ... repeat for parts 3-4
   ```

3. **Merge all parts into single JSON**:
   ```python
   # (Create merge script)
   merged = {
       'metadata': {...},
       'examples': part1_examples + part2_examples + part3_examples + part4_examples
   }
   ```

**Deliverable**: `data/spiqa_incongruent_samples.json` (~666-1332 examples)

---

### Step 7: LLM Validator (2-3 hours, API calls)

**Goal**: Verify error presence & category using second LLM

**Command**:
```bash
python src/validator.py data/spiqa_incongruent_samples.json \
  --images-root data/test-A/SPIQA_testA_Images \
  --output data/validation_results.json \
  --validator-model gpt-4-turbo \
  --max-samples 100  # Start with 100 for testing
```

**What it does**:
- For each example, calls validator model with: image + caption + student_answer + correct_answer
- Extracts: verdict, error_category
- Compares with original classification
- Outputs agreement metrics

**Expected output**:
```json
{
  "metadata": {
    "overall_agreement_rate": 0.92,
    "verdict_agreement_rate": 0.94,
    "error_category_agreement_rate": 0.91,
    "quality_gate_passed": true
  },
  "results": [
    {
      "example_id": "...",
      "verdict": "Incorrect",
      "error_category": "Omission",
      "original_verdict": "Incorrect",
      "original_error_category": "Omission",
      "overall_agreement": true
    },
    ...
  ]
}
```

**Deliverable**: `data/validation_results.json`

---

### Step 8: Human Spot-Check (3-4 hours manual)

**Goal**: Manual review of 50-100 stratified samples

**Sampling strategy**:
```python
# Stratify by error_category × figure_type
strata = [
  ('omission', 'plot'),
  ('omission', 'figure'),
  ('omission', 'table'),
  ('factual', 'plot'),
  ('factual', 'figure'),
  ('factual', 'table'),
  ('conceptual', 'plot'),
  ('conceptual', 'figure'),
  ('conceptual', 'table'),
]

# Sample 5-10 per stratum
for category, fig_type in strata:
    matching = [ex for ex in examples if ex['error_category'] == category and ex['figure_category'] == fig_type]
    sample = random.sample(matching, min(10, len(matching)))
```

**Spot-check criteria**:
- [ ] Is the student answer plausible?
- [ ] Is the error category accurate?
- [ ] Is the feedback constructive?
- [ ] Does the feedback explain the error?

**Output format** (`data/human_spotcheck.json`):
```json
{
  "example_id": {
    "reviewer": "Dipti",
    "human_verdict": "Incorrect",
    "human_error_category": "Omission",
    "quality_score": 4,  // 1-5 scale
    "comments": "Good example, feedback could be shorter"
  },
  ...
}
```

**Deliverable**: `data/human_spotcheck.json` (~50-100 examples)

---

### Step 9: Quality Gate (0.5 hours automated)

**Goal**: Ensure ≥90% agreement on validation set

**Check 1: LLM Validator Agreement**
```bash
# If validation_results.json shows overall_agreement_rate >= 0.90:
✅ PASSED
# Otherwise:
❌ FAILED - Revise exemplars and regenerate
```

**Check 2: Human Agreement (optional, if conducting spot-check)**
```python
human_agreement = sum(
    1 for review in human_spotcheck.values()
    if review['human_error_category'] == example['error_category']
) / len(human_spotcheck)

if human_agreement >= 0.90:
    print("✅ Quality gate PASSED")
else:
    print("❌ Quality gate FAILED")
```

**If FAILED**:
1. Identify failing exemplars in validator results
2. Review and improve seed exemplars (Step 4)
3. Regenerate (Step 6)
4. Re-validate (Step 7)

---

### Step 10: Assemble Final SPIQA+ Dataset (1 hour automated)

**Goal**: Create final 200-400 validated 4-tuples

**Command**:
```bash
python src/assemble_final.py \
  --incongruent data/spiqa_incongruent_samples.json \
  --validation data/validation_results.json \
  --output data/SPIQA_plus_final.json
```

**What it does**:
- Loads incongruent samples + validation results
- Checks quality gate (only if `--skip-quality-gate` NOT set)
- Filters examples based on validator agreement
- Assembles final dataset with metadata
- Computes error & figure distributions

**Output** (`data/SPIQA_plus_final.json`):
```json
{
  "metadata": {
    "source": "SPIQA Test-A",
    "dataset_type": "SPIQA+",
    "total_examples": 350,
    "examples_included": 350,
    "examples_excluded": 283,
    "overall_agreement_rate": 0.92,
    "quality_gate_passed": true,
    "error_distribution": {
      "Omission": 120,
      "Factual": 110,
      "Conceptual": 120
    },
    "figure_distribution": {
      "plot": 140,
      "figure": 120,
      "table": 90
    }
  },
  "examples": [
    {
      "id": "1603.00286v5_qa_1_error_1",
      "paper_id": "1603.00286v5",
      "qa_index": 1,
      "figure_category": "plot",
      "context": {
        "caption": "...",
        "image_path": "1603.00286v5-Figure3-1.png"
      },
      "question": "...",
      "user_answer": "...",
      "model_response": {
        "verdict": "Incorrect",
        "error_category": "Omission",
        "feedback": "..."
      },
      "validation": {
        "validator_verdict": "Incorrect",
        "validator_error_category": "Omission",
        "agreement": true
      }
    },
    ...
  ]
}
```

**Deliverable**: `data/SPIQA_plus_final.json` (~350 validated examples)

---

## Quick Start Commands

```bash
# Step 5: Generate congruent samples
python src/generate_congruent_samples.py data/SPIQA_testA_part1.json \
  --output data/spiqa_congruent_samples.json

# Step 7: Validate incongruent samples (test with 100 examples first)
python src/validator.py data/spiqa_incongruent_samples.json \
  --output data/validation_results.json \
  --max-samples 100

# Step 10: Assemble final dataset
python src/assemble_final.py \
  --incongruent data/spiqa_incongruent_samples.json \
  --validation data/validation_results.json \
  --output data/SPIQA_plus_final.json
```

---

## Timeline

| Step | Task | Effort | Status |
|------|------|--------|--------|
| 4 | Hand-curate 20-30 exemplars | 2-3 hrs (manual) | ⏳ TODO |
| 5 | Generate congruent samples | 0.5 hrs (automated) | ⏳ TODO |
| 6 | Generate incongruent samples | 4-6 hrs (API calls) | ⏳ TODO |
| 7 | LLM Validation | 2-3 hrs (API calls) | ⏳ TODO |
| 8 | Human spot-check | 3-4 hrs (manual) | ⏳ TODO |
| 9 | Quality gate | 0.5 hrs (automated) | ⏳ TODO |
| 10 | Assemble final | 1 hr (automated) | ⏳ TODO |
| **Total** | | **~13-18 hrs** | |

---

## Cost Estimates

**OpenAI API Costs**:
- Seed generation (Step 4): ~30 calls × $0.05 = **$1.50**
- Incongruent generation (Step 6): ~900 calls × $0.05 = **$45.00**
- LLM validation (Step 7): ~900 calls × $0.03 = **$27.00**
- **Total**: **~$73-75**

---

## Troubleshooting

**Issue**: Image not found errors in validator
```
→ Check that images_root path is correct and images exist
```

**Issue**: LLM agreement < 90%
```
→ Review failing examples in validation_results.json
→ Improve seed exemplars (Step 4)
→ Regenerate and re-validate
```

**Issue**: Output file already exists
```
→ Rename existing file or pass different --output path
```

---

## Next Steps

1. **Complete Step 4**: Manually select and curate 20-30 exemplars using seed.py
2. **Run Steps 5-7**: Execute automated generation and validation
3. **Conduct Step 8**: Manual spot-check of 50-100 samples
4. **Complete Steps 9-10**: Quality gate and assembly

See `PHASE2_PHASE3_IMPLEMENTATION_PLAN.md` for detailed design docs.
