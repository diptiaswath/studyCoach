# studyCoach
CMU MultiModal ML Course Project

Contributors: [Budhika, Jiashan, Khubi, Dipti]
## Core Scripts

### seed.py - Exemplar Generation
**Report Reference:** Phase 2, Step 4 — Hand-Curate ICL Seed Set

**Purpose:** Generate high-quality exemplar examples for in-context learning (ICL) bootstrapping. Supports manual curation of 20–30 high-quality 4-tuple examples (5–8 per error category across figure types).

**Input:**
- Image file (figure, plot, or table from academic paper)
- Caption text
- Question text
- Correct answer
- Desired error type (Omission, Factual, or Conceptual)

**Output:**
- Structured exemplar tuple: (user_prompt, assistant_response, error_classification)
- User prompt format: Caption + Question
- Assistant response format: Student answer + Agent verdict/error category/feedback

**Process:**
1. Takes a single image and caption
2. Calls OpenAI to generate a synthetic student error
3. Returns formatted exemplar for use in ICL prompts
4. Used manually in Step 4 to hand-curate 20-30 seed examples across error types and figure types

**Usage:**
```bash
python src/seed.py
```

**Report Quote:**
> "Manually create 20–30 high-quality 4-tuple examples (5–8 per error category across figure types)."

---

### icl.py - Inference Engine
**Report Reference:** Phase 2, Step 6 — ICL-Based Synthetic Generation

**Purpose:** Generate synthetic student responses using in-context learning across entire SPIQA Test-A dataset. Uses 2–3 seed exemplars per figure type to generate 1–2 incongruent user answers + structured model responses per QA.

**Input:**
- SPIQA JSON file (nested: paper_id → all_figures → qa list; 118 papers, 666 QA pairs)
- Images root directory (contains paper folders with figure images)
- Exemplar set (hardcoded or loaded from JSON; 2–3 per figure type)

**Output:**
- Updated JSON with inference results added to each QA item
- New fields added to each QA:
  - `student`: Generated wrong student answer (synthetic)
  - `verdict`: Incorrect/Partially Correct/Correct
  - `error_category`: Omission/Factual/Conceptual/N/A
  - `feedback`: Study coach explanation
  - `correct_answer`: Ground truth answer (for validation)

**Process:**
1. Iterates all papers in sorted order (deterministic processing)
2. For each QA item:
   - Extracts question, figure reference, caption from SPIQA metadata
   - Loads 2–3 exemplars matching figure type (plot/figure/table)
   - Builds multimodal prompt: system instructions + exemplars + current question with base64-encoded image
   - Calls OpenAI with instructions parameter (not system role)
   - Parses structured output using regex (Verdict, Error Category, Feedback)
   - Updates JSON in-place
3. Tracks error distribution (Factual/Omission/Conceptual counts)
4. Stops early when all error types have been generated

**Usage:**
```bash
python src/icl.py data/SPIQA_testA_part1.json data/test-A/SPIQA_testA_Images \
  --output data/SPIQA_testA_part1_output.json
```

**Key Details:**
- Processes papers in **sorted order** (deterministic for reproducibility)
- Only processes QAs with available exemplars for that figure type
- Currently only 'plot' exemplars exist; 'figure' and 'table' exemplars needed
- Uses dynamic error balancing in prompt template (substitutes current FACTUAL/OMISSION/CONCEPTUAL counts)
- Each exemplar includes base64-encoded image for multimodal input

**Report Quote:**
> "Prompt an LLM with 2–3 seed examples to generate 1–2 incongruent user answers + structured model responses per QA."

---

## 4-Tuple Structure

Following the StudyCoach report specification, each generated example is a **4-tuple** with:

| Field | Source | Description |
|-------|--------|-------------|
| Context | SPIQA | Visual and textual context from the paper (figure + caption) |
| Question | SPIQA | The figure-grounded question |
| User Answer | Generated (synthetic) | A student's explanation (correct/partial/incorrect) |
| Model Response | Generated (synthetic) | Verdict + error category + coaching feedback |

**Structured Model Response:**

| Component | Values | Purpose |
|-----------|--------|---------|
| Verdict | Correct / Partially Correct / Incorrect | Tests detection capability |
| Error Category | Factual / Omission / Conceptual | Tests error classification |
| Feedback | Free-text | Qualitative evaluation & study coach explanation |

---

## Error Classification System

Following the report's framework, three core error types guide both prompt engineering and output parsing:

- **Omission:** Missing key details in the student answer (e.g., "The methods are similar" vs full metric comparison)
- **Factual:** Misreading visual elements (wrong axis interpretation, legend confusion, incorrect values)
- **Conceptual:** Wrong conclusion from correct data (e.g., confusing lower loss with better performance)

Error distribution is tracked globally via counters and balanced dynamically in ICL prompts.

---

## Data Flow & Phases

### Phase 2: Generate (Steps 4-6)

```
Step 4: seed.py
  ↓
  Manually curate 20-30 exemplars (5-8 per error category)
  Save to data/seed_exemplars.json
  ↓
Step 5: generate_congruent_samples.py (separate pipeline)
  ↓
  Extract 666 "correct answer" examples
  ↓
Step 6: icl.py (main inference)
  ↓
  Load exemplars from seed_exemplars.json
  Generate 1-2 incongruent answers per QA
  Output: SPIQA Test-A with synthetic errors added
```

### Phase 3: Validate (Steps 7-10)

```
Step 7: validator.py
  ↓
  Re-validate each synthetic example with different LLM
  Compute agreement metrics
  ↓
Step 8: Human Spot-Check
  ↓
  Manual review of 50-100 stratified samples
  ↓
Step 9-10: assemble_final.py
  ↓
  Apply quality gate (≥90% LLM-human agreement)
  Output: Final SPIQA+ dataset (200-400 validated 4-tuples)
```