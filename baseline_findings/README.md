# Multimodal Incongruence Detection — Evaluation Scripts

Evaluates whether an LLM can assess a student's answer to a scientific figure question
and assign a verdict (Correct / Partially Correct / Incorrect) and error category
(Omission / Factual / Conceptual).

---

## Prerequisites

### 1. Install dependencies

```bash
pip install openai
```

### 2. Get API keys

**Together.ai** — used for all 4 scenarios (~$0.01–$0.05 total for 50 examples)
1. Sign up at https://api.together.ai
2. Go to Settings → API Keys → Create API Key
3. Export: `export TOGETHER_API_KEY=your_key`

> Groq was used in an earlier version for the text-only scenario but has been replaced
> by Together.ai so all scenarios use the same model for fair comparison.

### 3. Set environment variables

Either export directly:

```bash
export TOGETHER_API_KEY=your_key_here
```

Or add to a `.env.local` file and source it:

```bash
source .env.local
```

---

## Running the Evaluation

All scripts should be run from the `studyCoach/` directory.

### The 4 scenarios

| Script | Caption | Image |
|---|---|---|
| `eval_text_only.py` | ❌ | ❌ |
| `eval_caption_only.py` | ✅ | ❌ |
| `eval_vision_only.py` | ❌ | ✅ |
| `eval_multimodal.py` | ✅ | ✅ |

### Full run (50 examples per scenario)

```bash
# Scenario 1: Text-only (no caption, no image)
python src/eval_text_only.py \
  --data data/test-A/SPIQA_testA_part1_output_latest.json \
  --images data/test-A/SPIQA_testA_Images \
  --output data/eval/text_only_results.json

# Scenario 2: Caption-only
python src/eval_caption_only.py \
  --data data/test-A/SPIQA_testA_part1_output_latest.json \
  --images data/test-A/SPIQA_testA_Images \
  --output data/eval/caption_only_results.json

# Scenario 3: Vision-only (no caption)
python src/eval_vision_only.py \
  --data data/test-A/SPIQA_testA_part1_output_latest.json \
  --images data/test-A/SPIQA_testA_Images \
  --output data/eval/vision_only_results.json

# Scenario 4: Multimodal (caption + image)
python src/eval_multimodal.py \
  --data data/test-A/SPIQA_testA_part1_output_latest.json \
  --images data/test-A/SPIQA_testA_Images \
  --output data/eval/multimodal_results.json
```

### Without reference answer (add `--no-answer` flag)

Each script accepts a `--no-answer` flag that omits the ground truth reference answer
from the prompt. This tests whether the model can assess the student purely from the
figure/caption, without an answer key.

```bash
python src/eval_text_only.py \
  --data data/test-A/SPIQA_testA_part1_output_latest.json \
  --images data/test-A/SPIQA_testA_Images \
  --output data/eval/text_only_no_answer_results.json \
  --no-answer
```

The same flag works on all 4 scripts.

### Smoke test (3 examples)

Add `--max 3` to any script to test on just 3 examples before a full run:

```bash
python src/eval_multimodal.py \
  --data data/test-A/SPIQA_testA_part1_output_latest.json \
  --images data/test-A/SPIQA_testA_Images \
  --output data/eval/multimodal_results.json \
  --max 3
```

---

## Output Format

Each script writes a JSON array to the specified `--output` path. Each entry:

```json
{
  "paper_id": "1802.07351v2",
  "question": "What trend does the figure show?",
  "answer": "The reference correct answer...",
  "caption": "Figure 3: ...",
  "image_path": "data/test-A/SPIQA_testA_Images/1802.07351v2/...",
  "student": "The synthetic student answer...",
  "ground_truth": {
    "verdict": "Incorrect",
    "error_category": "Factual",
    "feedback": "..."
  },
  "predicted": {
    "verdict": "Incorrect",
    "error_category": "Factual",
    "feedback": "..."
  },
  "raw_output": "Verdict = Incorrect\nError Category = Factual\nFeedback = ..."
}
```

---

## Project Structure

```
studyCoach/
  src/
    eval_utils.py           # Shared: data loading, sampling, parsing, saving
    eval_text_only.py       # Scenario 1: no caption, no image
    eval_caption_only.py    # Scenario 2: caption only
    eval_vision_only.py     # Scenario 3: image only
    eval_multimodal.py      # Scenario 4: caption + image
    icl.py                  # Generates synthetic student answers (upstream)
  prompts/
    incongruence_eval_v1.txt  # Initial prompt (no verdict defs)
    incongruence_eval_v2.txt  # + verdict definitions
    incongruence_eval_v3.txt  # + ICL exemplars (current)
  data/
    test-A/
      SPIQA_testA_part1_output_latest.json  # Augmented dataset (174 QAs)
      SPIQA_testA_Images/                   # Per-paper image directories
    eval/
      text_only_results.json
      caption_only_results.json
      vision_only_results.json
      multimodal_results.json
      text_only_no_answer_results.json
      caption_only_no_answer_results.json
      vision_only_no_answer_results.json
      multimodal_no_answer_results.json
  baseline_findings/
    README.md       # This file
    FINDINGS.md     # Full results and analysis
```

---

## Model & Sampling Details

- **Model:** `Qwen/Qwen3-VL-8B-Instruct` (Together.ai serverless endpoint)
- **Sampling:** Equal-stratum across (verdict × error category), seed=42
- **All 4 scenarios use the same 50 examples** — results are directly comparable
- **Prompt:** `prompts/incongruence_eval_v3.txt` — loaded at runtime from disk
- To swap the prompt, edit the filename in `eval_utils.py:SYSTEM_PROMPT`
- To swap the model, edit `MODEL` at the top of any eval script
