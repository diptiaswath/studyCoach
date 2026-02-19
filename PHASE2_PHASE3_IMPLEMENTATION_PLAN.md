# Phase 2 & 3 Implementation Plan: SPIQA+ Dataset Generation & Validation

## Overview
This document outlines the implementation steps for generating 200-400 synthetic student error examples (SPIQA+) from the SPIQA Test-A dataset (666 QAs across 118 papers).

---

## Phase 2: Generate

### Step 4: Hand-Curate ICL Seed Set (20-30 examples)
**Goal**: Create high-quality exemplars (5-8 per error category) across figure types

**Figure Types to Cover**:
- `plot` (bar charts, line graphs)
- `figure` (schematics, diagrams)
- `table` (structured data)

**Error Categories**:
- Omission (missing key details)
- Factual (misreading visual elements)
- Conceptual (wrong conclusion from correct data)

**Process**:
1. Use `src/seed.py` to generate 30+ raw examples (currently generates 1)
2. Manually review and select best 20-30 examples
3. Store curated set in `data/seed_exemplars.json`
4. Organize by figure_category + error_category

**Deliverable**: `data/seed_exemplars.json`
```json
{
  "plot": {
    "omission": [exemplar1, exemplar2, ...],
    "factual": [...],
    "conceptual": [...]
  },
  "figure": {...},
  "table": {...}
}
```

---

### Step 5: Generate Congruent Samples
**Goal**: Create "correct" user answer samples from SPIQA ground truth

**Process**:
1. Read `data/SPIQA_testA_part*.json`
2. For each QA, use ground truth answer as "correct" user answer
3. Create 4-tuple: (context, question, user_answer="correct", model_response=None)
4. Save to `data/spiqa_congruent_samples.json`

**Deliverable**: ~666 congruent 4-tuples (one per QA)

---

### Step 6: ICL-Based Synthetic Generation
**Goal**: Generate 1-2 incongruent (wrong) answers per QA using LLM + seed exemplars

**Process**:
1. Load seed exemplars from Step 4
2. For each QA in Test-A:
   - Select 2-3 exemplars matching figure_category
   - Build prompt with exemplars + caption + question
   - Call OpenAI API to generate 1-2 wrong answers
   - Parse output (student_answer, verdict, error_category, feedback)
   - Append to `data/spiqa_incongruent_samples.json`
3. Track error distribution (Omission/Factual/Conceptual)

**Deliverable**: ~666-1332 incongruent 4-tuples

**Pseudocode**:
```python
for paper_key in sorted(test_a_json.keys()):
  for qa in paper_key["qa"]:
    figure_category = normalize_figure_type(qa["reference"])
    exemplars = seed_set[figure_category]  # 2-3 examples
    
    for num_errors in [1, 2]:  # Generate 1-2 wrong answers
      response = openai.chat(
        system=SYSTEM_PROMPT,
        exemplars=exemplars,
        user_query=qa["question"] + qa["reference_image"]
      )
      parsed = parse_response(response)
      incongruent_samples.append({
        "paper": paper_key,
        "qa_index": qa_idx,
        "figure_category": figure_category,
        "student_answer": parsed.student_answer,
        "verdict": parsed.verdict,
        "error_category": parsed.error_category,
        "feedback": parsed.feedback
      })
```

---

## Phase 3: Validate

### Step 7: LLM Validator
**Goal**: Verify error presence and category using a second LLM (e.g., Gemini, different GPT model)

**Process**:
1. For each synthetic example:
   - Send to validator model with:
     - Original image + caption + question
     - Generated student answer
     - Prompt: "Is this answer incorrect? What error category?"
   - Extract: verdict_val, error_category_val
2. Compare with original classification
3. Log disagreements for manual review

**Deliverable**: `data/validation_results.json`
```json
{
  "example_id": {
    "original_verdict": "Incorrect",
    "validator_verdict": "Incorrect",
    "original_error": "Omission",
    "validator_error": "Omission",
    "agreement": true
  },
  ...
}
```

---

### Step 8: Human Spot-Check
**Goal**: Manual review of ~50-100 stratified samples

**Sampling Strategy**:
- Stratify by: error_category (Omission/Factual/Conceptual) × figure_type (plot/figure/table)
- Select 5-10 examples per stratum
- Total: 45-90 examples

**Spot-Check Criteria**:
- Is the student answer plausible? (Would a real student make this mistake?)
- Is the error category accurate?
- Is the feedback constructive and concise?
- Does the feedback correctly explain the error?

**Deliverable**: `data/human_spotcheck.json`
```json
{
  "example_id": {
    "human_verdict": "Incorrect",
    "human_error_category": "Omission",
    "quality_score": 4/5,
    "comments": "Good example, feedback could be more concise"
  },
  ...
}
```

---

### Step 9: Quality Gate
**Goal**: Ensure LLM validator & original classification meet 90% agreement threshold

**Criteria**:
1. LLM validator agrees with original classification ≥90% on spot-check subset
2. Human judges agree with original classification ≥90% on spot-check subset
3. If either fails: revise exemplars (Step 4) and regenerate (Step 6)

**Process**:
```python
agreement_count = 0
for example in spot_check_subset:
  if example.validator_error == example.original_error:
    agreement_count += 1

agreement_rate = agreement_count / len(spot_check_subset)
if agreement_rate >= 0.90:
  print("✅ Quality gate PASSED")
else:
  print("❌ Quality gate FAILED - revise exemplars and regenerate")
```

---

### Step 10: Final SPIQA+ Dataset
**Goal**: Assemble 200-400 validated 4-tuples with annotations

**Output Structure**:
```json
{
  "metadata": {
    "source": "SPIQA Test-A",
    "total_examples": 350,
    "error_distribution": {
      "omission": 120,
      "factual": 110,
      "conceptual": 120
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
      "user_answer": "...",  // Synthetic (could be correct or incorrect)
      "model_response": {
        "verdict": "Incorrect",
        "error_category": "Omission",
        "feedback": "..."
      },
      "validation": {
        "validator_agreement": true,
        "human_reviewed": false,
        "quality_score": null
      }
    },
    ...
  ]
}
```

**Deliverable**: `data/SPIQA_plus_final.json` (~350 examples)

---

## Implementation Timeline & Dependencies

| Step | Status | Dependencies | Effort |
|------|--------|--------------|--------|
| 4 | TODO | seed.py enhanced | 2-3 hrs (manual curation) |
| 5 | TODO | Step 4 | 0.5 hrs (scripted) |
| 6 | TODO | Steps 4-5 | 4-6 hrs (API calls) |
| 7 | TODO | Step 6 | 2-3 hrs (API calls) |
| 8 | TODO | Step 7 | 3-4 hrs (manual) |
| 9 | TODO | Steps 7-8 | 0.5 hrs (scripted) |
| 10 | TODO | Steps 6-9 | 1 hr (scripted) |

**Total**: ~13-18 hours

---

## File Organization

```
studyCoach/
├── data/
│   ├── SPIQA_testA_part*.json          (Original SPIQA)
│   ├── seed_exemplars.json             (Step 4 - 20-30 curated examples)
│   ├── spiqa_congruent_samples.json    (Step 5 - 666 correct answers)
│   ├── spiqa_incongruent_samples.json  (Step 6 - 666-1332 wrong answers)
│   ├── validation_results.json         (Step 7 - LLM validation)
│   ├── human_spotcheck.json            (Step 8 - Human review)
│   └── SPIQA_plus_final.json           (Step 10 - Final dataset)
├── src/
│   ├── seed.py                         (Enhanced for Step 4)
│   ├── icl.py                          (Enhanced for Steps 5-6)
│   ├── validator.py                    (NEW - Step 7)
│   ├── spotcheck_ui.py                 (NEW - Step 8)
│   ├── quality_gate.py                 (NEW - Step 9)
│   └── assemble_final.py               (NEW - Step 10)
└── PHASE2_PHASE3_IMPLEMENTATION_PLAN.md (This file)
```

---

## Next Steps

1. **Immediate**: Enhance `seed.py` to generate 20-30 examples interactively
2. **Week 1**: Run Step 5 & 6 (generate all 666-1332 incongruent samples)
3. **Week 2**: Run Step 7 (LLM validation)
4. **Week 2**: Run Step 8 (Human spot-check, 3-4 hours manual work)
5. **Week 2**: Run Steps 9-10 (Quality gate & assembly)

---

## Cost Estimates

**API Calls**:
- Seed generation (Step 4): ~30 calls × $0.05 = $1.50
- Incongruent generation (Step 6): ~900 calls × $0.05 = $45
- LLM validation (Step 7): ~900 calls × $0.03 = $27
- **Total**: ~$73-75

---

## Questions for Clarification

1. Should we store congruent + incongruent separately or merge into one dataset?
2. Who will conduct the human spot-check?
3. Should we support multiple validators or use a single second model?
4. Do we need a web UI for spot-checking or is a JSON file + Python script sufficient?
5. Should we version the intermediate datasets (e.g., `SPIQA_plus_v1.json`) or overwrite?

