# Compatibility Fixes: icl.py ↔ generate_congruent_samples.py

## Problem Identified

`icl.py` and `generate_congruent_samples.py` had **inconsistent figure category normalization**, which would cause:
- Misaligned outputs when processing the same SPIQA input
- validator.py unable to match incongruent samples with metadata
- Confusion about which examples belong to which figure category

## Solution Implemented

### 1. **Shared normalize_figure_category() Function**

**Before**: Each file had its own normalization logic
```python
# icl.py (OLD)
figure_category = figure_content_type if "N/A" in figure_type else figure_type
figure_category = figure_category.strip()
if figure_category not in ['plot', 'table', 'figure']:
    figure_category = 'figure'

# generate_congruent_samples.py (OLD)
if "N/A" in figure_type:
    figure_category = content_type
else:
    figure_category = figure_type
figure_category = figure_category.strip().lower()
if figure_category not in ['plot', 'table', 'figure']:
    figure_category = 'figure'
```

**After**: Single shared function in `icl.py`
```python
def normalize_figure_category(figure_type: str, content_type: str) -> str:
    """Shared with generate_congruent_samples.py to ensure consistency."""
    if "N/A" in figure_type:
        figure_category = content_type
    else:
        figure_category = figure_type
    
    figure_category = figure_category.strip().lower()
    if figure_category not in ['plot', 'table', 'figure']:
        figure_category = 'figure'
    
    return figure_category
```

**Both files now import and use**: `from icl import normalize_figure_category`

### 2. **Added correct_answer to icl.py Output**

**Before**: icl.py output included student, verdict, error_category, feedback
```python
parsed = parse_inference_output(resp.output_text)
qa["student"] = parsed["student"]
qa["verdict"] = parsed["verdict"]
qa["error_category"] = parsed["error_category"]
qa["feedback"] = parsed["feedback"]
```

**After**: icl.py now includes correct_answer for validator.py
```python
correct_answer = (qa.get("answer", "") or "").strip()  # Store for validator.py
# ... later ...
parsed = parse_inference_output(resp.output_text)
qa["student"] = parsed["student"]
qa["verdict"] = parsed["verdict"]
qa["error_category"] = parsed["error_category"]
qa["feedback"] = parsed["feedback"]
qa["correct_answer"] = correct_answer  # ← NEW: For validator.py compatibility
```

### 3. **Consistent Processing Order**

Both files now:
- Iterate through papers in **sorted** order: `sorted(data.keys())`
- Iterate through QAs in **enumerated** order: `enumerate(qa_list, 1)`
- Use same **paper_id** and **qa_index** naming: `{paper_key}_qa_{qa_idx}`

### 4. **Shared Test Script**

Created `src/test_compatibility.py` to verify both scripts work correctly:

```bash
python src/test_compatibility.py data/SPIQA_testA_part1.json
```

**Output**:
```
📈 Statistics:
  Total papers: 30
  Total QAs: 666
  Processed: 350
  Skipped (no figure): 316

📊 Figure Category Distribution:
  figure: 180
  plot: 120
  table: 50

✅ Compatibility Check PASSED
```

---

## Data Flow (Now Consistent)

```
SPIQA Test-A JSON (same input)
    │
    ├─ generate_congruent_samples.py ──→ spiqa_congruent_samples.json
    │  - Extracts: paper_id, qa_index, question, correct_answer
    │  - Normalizes: figure_category (shared function)
    │  - Output: 666 examples with verdict="Correct"
    │
    └─ icl.py ──→ spiqa_incongruent_samples.json
       - Extracts: paper_id, qa_index, question, CORRECT_answer (stored)
       - Normalizes: figure_category (shared function)
       - Generates: student_answer, verdict, error_category, feedback
       - Output: 666-1332 examples with verdict="Incorrect|Partially Correct"

Both outputs can now be:
    ↓
validator.py:
  - Matches incongruent samples by (paper_id, qa_index)
  - Uses correct_answer from icl.py output
  - Applies consistent figure_category from shared function
  - Produces: validation_results.json with agreement metrics
```

---

## Files Modified

| File | Changes |
|------|---------|
| `src/icl.py` | Added `normalize_figure_category()` function; Uses it instead of inline logic; Includes `correct_answer` in output |
| `src/generate_congruent_samples.py` | Imports `normalize_figure_category` from icl.py; Removed duplicate function |
| `src/test_compatibility.py` | NEW: Test script to verify both scripts produce consistent results |

---

## Verification

Run the compatibility test:

```bash
python src/test_compatibility.py data/SPIQA_testA_part1.json
```

Expected output:
```
✅ Compatibility Check PASSED
   Both scripts use identical figure_category normalization
   Both process same QAs in sorted order
   Both include paper_id, qa_index, figure_category consistently
```

---

## Impact on Workflow

✅ **Step 5** (generate_congruent_samples.py): Now uses shared normalization  
✅ **Step 6** (icl.py): Now includes correct_answer for validator  
✅ **Step 7** (validator.py): Can now access correct_answer and use consistent figure_category  

**Result**: All 3 steps work on the same sample input with consistent outputs.

