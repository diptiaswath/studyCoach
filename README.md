# studyCoach
CMU MultiModal ML Course Project

Contributors: [Budhika, Jiashan, Khubi, Dipti]
## Core Scripts

### seed.py - Exemplar Generation
**Report Reference:** Phase 2, Step 4 — Hand-Curate ICL Seed Set

**Report Requirement:**
> "Manually create 20–30 high-quality 4-tuple examples (5–8 per error category across figure types)."

**Purpose:** Generate high-quality exemplar examples for in-context learning (ICL) bootstrapping. Seed examples teach the LLM how to generate different types of student errors (Omission, Factual, Conceptual) on scientific paper figure questions. Once manually curated, these exemplars are used by icl.py during Phase 2, Step 6 for the main inference pipeline.

**Required Inputs (8 Parameters)**

| # | Parameter | Type | Example | Source |
|---|-----------|------|---------|--------|
| 1 | `image_path` | str | `'data/test-A/SPIQA_testA_Images/1702.08694v3/1702.08694v3-Figure3-1.png'` | SPIQA image file |
| 2 | `caption` | str | `'Figure 3: Results on real data...'` | SPIQA figure metadata |
| 3 | `question` | str | `'How does C-Tarone compare to binarization?'` | SPIQA qa.question |
| 4 | `answer` | str | `'The C-Tarone method has higher precision...'` | SPIQA qa.answer |
| 5 | `verdict` | str | `'incorrect'` or `'partially correct'` | From verdicts dict |
| 6 | `error_category` | str | `'omission'`, `'factual'`, or `'conceptual'` | From error_categories dict |
| 7 | `verdict_explanation` | str | `'an answer which gets none of the required key insights correct'` | verdicts[verdict] |
| 8 | `error_category_explanation` | str | `'an error due to omitting key details in the answer'` | error_categories[error_category] |

**Example Function Call**
```python
chart_incorrect_omission = generate_seed_example(
    image_path=chart1,                                    # ← Input 1
    caption=chart1_caption,                               # ← Input 2
    question=chart1_question,                             # ← Input 3
    answer=chart1_answer,                                 # ← Input 4
    verdict='incorrect',                                  # ← Input 5
    error_category='omission',                            # ← Input 6
    verdict_explanation=verdicts['incorrect'],            # ← Input 7
    error_category_explanation=error_categories['omission']  # ← Input 8
)
```

**Output Structure**

| Field | Format | Example |
|-------|--------|---------|
| `verdict` | Single word | `Incorrect` |
| `error_category` | Single word | `Omission` |
| `student_answer` | Multi-sentence | `The C-Tarone method is generally similar to the binarization method...` |
| `feedback` | Multi-sentence coaching | `Your answer omits key details... According to the figure, C-Tarone consistently has higher precision...` |

**Process**
1. Provide image, caption, question, and ground truth answer
2. Specify desired error type (verdict + error_category) 
3. Call `generate_seed_example()` with all 8 required inputs
4. LLM generates a synthetic wrong student answer + coaching feedback
5. Manually review output for quality
6. Hardcode high-quality exemplars into icl.py (currently lines 330–368)

**Curation Guidelines**
- Target: 20–30 total exemplars across all figure types
- Distribution: 5–8 per error category (Omission, Factual, Conceptual)
- Figure types: Include exemplars for 'plot', 'figure', and 'table'
- Quality: Prefer subtle, realistic errors over obvious mistakes
- Manual review: All generated exemplars must be reviewed before use

**Usage**
```bash
python src/seed.py
```

**Output Destination**
Generated exemplars should be saved to `data/seed_exemplars.json` for Step 6 (icl.py) to load dynamically during the main inference pipeline.

---

### icl.py - Inference Engine
**Report Reference:** Phase 2, Step 6 — ICL-Based Synthetic Generation

**Report Requirement:**
> "Prompt an LLM with 2–3 seed examples to generate 1–2 incongruent user answers + structured model responses per QA."

**Purpose:** Generate synthetic student responses using in-context learning across entire SPIQA Test-A dataset. Takes curated seed exemplars from Step 4 and uses them to teach the LLM how to generate realistic student errors on scientific paper figure questions. Produces 1–2 synthetic incongruent (wrong) answers per QA with structured model responses (verdict + error category + feedback).

**Required Inputs (4 Parameters)**

| # | Parameter | Type | Example | Source | Notes |
|---|-----------|------|---------|--------|-------|
| 1 | `json_path` | str/Path | `'data/SPIQA_testA_part1.json'` | SPIQA Test-A dataset | Nested JSON: paper_id → all_figures → qa list |
| 2 | `images_root` | str/Path | `'data/test-A/SPIQA_testA_Images'` | Image directory | Contains paper folders with figure images |
| 3 | `exemplars` | dict | `{'plot': [...], 'figure': [...], 'table': [...]}` | seed.py output | 2–3 exemplars per figure type; keys: 'plot', 'figure', 'table' |
| 4 | `output_path` | str/Path | `'data/SPIQA_testA_part1_output.json'` | Output file | Updated JSON with inference results |

**Example Command**
```bash
python src/icl.py data/SPIQA_testA_part1.json data/test-A/SPIQA_testA_Images \
  --output data/SPIQA_testA_part1_output.json
```

**Exemplar Format (Input 3)**
```python
exemplars = {
    'plot': [
        (user_prompt_1, assistant_response_1, 'data/test-A/SPIQA_testA_Images/.../image1.png'),
        (user_prompt_2, assistant_response_2, 'data/test-A/SPIQA_testA_Images/.../image2.png'),
        (user_prompt_3, assistant_response_3, 'data/test-A/SPIQA_testA_Images/.../image3.png')
    ],
    'figure': [...],
    'table': [...]
}
```

**Output Structure**

Each QA item in the output JSON gains 5 new fields:

| Field | Type | Example | Purpose |
|-------|------|---------|---------|
| `student` | str | `'The C-Tarone method is generally similar...'` | Generated wrong student answer (synthetic) |
| `verdict` | str | `'Incorrect'` | Student answer classification: Correct / Partially Correct / Incorrect |
| `error_category` | str | `'Omission'` | Error type: Omission / Factual / Conceptual / N/A |
| `feedback` | str | `'Your answer omits key details... According to the figure...'` | Study coach explanation of the error (2–4 sentences) |
| `correct_answer` | str | `'The C-Tarone method has higher precision...'` | Ground truth answer (from SPIQA qa.answer) — used for validation |

**Example Output QA Item**
```json
{
  "question": "How does the C-Tarone method compare to the binarization method?",
  "answer": "The C-Tarone method has higher precision and F-measure...",
  "reference": "1702.08694v3-Figure3-1.png",
  "student": "The C-Tarone method is generally similar to the binarization method across all datasets.",
  "verdict": "Incorrect",
  "error_category": "Omission",
  "feedback": "Your answer omits key details regarding the differences between the two methods. According to the figure, C-Tarone consistently has higher precision and F-measure than binarization.",
  "correct_answer": "The C-Tarone method has higher precision and F-measure..."
}
```

**Process**

1. **Load Data** — Reads SPIQA JSON (118 papers, 666 QA pairs)
2. **Iterate in Sorted Order** — Processes papers alphabetically (deterministic for reproducibility)
3. **For Each QA:**
   - Extract question, figure reference, caption from SPIQA metadata
   - Look up figure details: content_type, figure_type
   - Normalize to figure_category ('plot', 'figure', or 'table')
   - Load 2–3 seed exemplars matching that figure category
   - If no exemplars exist for that figure type, skip QA
4. **Build Multimodal Prompt:**
   - System instructions (with dynamic error counts: $FACTUAL, $OMISSION, $CONCEPTUAL)
   - 2–3 exemplar messages (each with user prompt + base64-encoded image + assistant response)
   - Final user message (caption + question + current image as base64)
5. **Call OpenAI API:**
   - Model: gpt-5.1
   - Endpoint: `client.responses.create()` with `instructions` parameter
   - Include metadata (paper_id, qa_index) for debugging
6. **Parse Output:**
   - Use regex to extract: Student answer, Verdict, Error Category, Feedback
   - Clean whitespace and handle edge cases
   - If verdict is "Correct", auto-set error_category and feedback to "N/A"
7. **Update JSON:**
   - Add parsed fields to QA item
   - Track error distribution (increment FACTUAL/OMISSION/CONCEPTUAL counters)
8. **Exit Condition:**
   - Stop early when all three error types (FACTUAL > 0, OMISSION > 0, CONCEPTUAL > 0) have been generated at least once
9. **Write Output:**
   - Save updated JSON to output_path
   - Print final error distribution counts

**Critical Implementation Details**

**Deterministic Ordering:**
```python
for paper_key in sorted(data.keys()):           # Must use sorted()
    for qa_idx, qa in enumerate(qa_list, 1):    # Sequential QA index
```
Processing order is critical—outputs must match this exact iteration or parsing will misalign between icl.py and validator.py.

**Figure Category Normalization:**
```python
figure_category = figure_content_type if "N/A" in figure_type else figure_type
if figure_category not in exemplars:
    continue  # Skip if no exemplars for this figure type
```
Only QAs with corresponding exemplar categories are processed.

**Base64 Image Encoding:**
Each exemplar and the current question image are converted to base64 with MIME type:
```
data:image/jpeg;base64,{base64_data}
```

**Key Details:**
- Processes papers in **sorted order** (deterministic for reproducibility across runs)
- Only processes QAs with available exemplars for that figure type
- Currently only **'plot' exemplars exist** (lines 330–368); 'figure' and 'table' exemplars needed for complete coverage
- Uses **dynamic error balancing** in prompt template (substitutes current FACTUAL/OMISSION/CONCEPTUAL counters to encourage diverse error types)
- Each exemplar includes **base64-encoded image** for multimodal input to OpenAI
- Stores `correct_answer` from SPIQA to enable validator.py to verify outputs
- Early exit optimization: stops processing once all error types have been generated at least once

**Performance Metrics**
- Dataset size: 118 papers, 666 QA pairs in SPIQA Test-A
- Processing: ~1–2 API calls per QA (depends on exemplar availability)
- Estimated cost: $73–75 for full Test-A dataset (gpt-5.1 at ~$2/1M input tokens)
- Estimated time: 13–18 hours depending on API rate limits

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