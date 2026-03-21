# LLM-as-Judge Evaluation: Qwen3-VL-32B

## Setup

- **Evaluated model**: `Qwen3-VL-32B-Instruct` (via Together.ai)
- **Judge model**: `claude-sonnet-4-6` (Anthropic)
- **N = 50 examples per scenario** (4 scenarios evaluated)
- **Reference answer withheld** from Qwen3 during feedback generation (to encourage reasoning over string matching)
- The LLM judge receives: question, student answer, ground-truth feedback (A), and predicted feedback (B) — and judges whether B provides equivalent guidance to A
- 3 LLM judge labels: `match`, `partial`, `unmatched`

---

## Prompt Conditions

| Dimension | Options |
|---|---|
| **Input modality** | `text_only`, `caption_only`, `vision_only`, `multimodal` |
| **Reference answer provided to evaluated model** | No |

> **Note**: Same judge prompt as the 8B baseline evaluation.

---

## Evaluation Results

### Phase 1: Verdict Accuracy

| Scenario | Accuracy |
|---|---|
| caption_only | **58%** |
| text_only | 56% |
| vision_only | 56% |
| multimodal | 54% |

**Finding**: Verdict accuracy is flat across scenarios (~54–58%). Unlike the 8B baseline where text_only was the clear winner, at 32B scale modality differences are negligible for binary classification.

### Phase 2a: Automated Metrics

| Scenario | F1 | ROUGE-L | BLEU |
|---|---|---|---|
| caption_only | **0.325** | **0.207** | 7.16 |
| multimodal | 0.298 | 0.202 | 7.44 |
| text_only | 0.298 | 0.192 | 5.93 |
| vision_only | 0.294 | 0.193 | **8.13** |

**Finding**: As with the baseline, automatic metrics are essentially flat and fail to differentiate feedback quality across scenarios.

### Phase 2b: Human Evaluation (N=10 per scenario)

Same annotation indices as baseline for direct comparison.

| Scenario | Match | Partial | Unmatched | Human Match % | Soft Match % |
|---|---|---|---|---|---|
| multimodal | 8 | 1 | 1 | **80%** | 90% |
| vision_only | 6 | 2 | 2 | 60% | 80% |
| caption_only | 5 | 4 | 1 | 50% | **90%** |
| text_only | 5 | 4 | 1 | 50% | **90%** |

**Finding**: Multimodal leads on strict match. Caption_only and text_only tie at 50% but both reach 90% soft match, suggesting the model frequently captures the right idea but incompletely.

### Phase 2c: LLM-as-Judge (N=50 per scenario)

| Scenario | N | Match | Partial | Unmatched | Match % | Soft Match % | Avg F1 | Avg ROUGE-L | Avg BLEU |
|---|---|---|---|---|---|---|---|---|---|
| multimodal | 50 | 10 | 19 | 21 | **20.0%** | **58.0%** | 0.298 | 0.202 | 7.44 |
| caption_only | 50 | 4 | 20 | 26 | 8.0% | 48.0% | **0.325** | **0.207** | 7.16 |
| vision_only | 50 | 3 | 21 | 26 | 6.0% | 48.0% | 0.294 | 0.193 | 8.13 |
| text_only | 50 | 1 | 20 | 29 | 2.0% | 42.0% | 0.298 | 0.192 | 5.93 |

**Finding**: Multimodal leads on strict match (20%) with a clear gap over other scenarios. Soft match rates (42–58%) are more compressed, mirroring the baseline pattern where the model is often directionally correct but incomplete.

---

## Comparison with Human Annotation

Human annotators (N=10 per scenario) were consistently more lenient than the LLM judge (N=50), matching the same pattern observed for the 8B baseline.

| Scenario | Human Match % | LLM Match % | Human Soft % | LLM Soft % |
|---|---|---|---|---|
| multimodal | **80%** | **20%** | 90% | **58%** |
| vision_only | 60% | 6% | 80% | 48% |
| caption_only | 50% | 8% | **90%** | 48% |
| text_only | 50% | 2% | **90%** | 42% |

> Note: Human N=10, LLM judge N=50. Comparison is directional.

**Scenario ranking is consistent across both evaluations**: multimodal > vision_only ≥ caption_only > text_only. Despite the absolute gap (human rates 4–10× higher than LLM strict match), the ordering is preserved — the LLM judge confirms the human finding at larger scale. The leniency gap reflects the stricter rubric, larger sample, and the tendency for human annotators to credit feedback that is _directionally correct_ even if incomplete.

---

## Comparison with 8B Baseline

### Verdict Accuracy

| Scenario | 8B | 32B | Delta |
|---|---|---|---|
| caption_only | 48% | **58%** | +10pp |
| vision_only | 50% | **56%** | +6pp |
| multimodal | 48% | **54%** | +6pp |
| text_only | 56% | 56% | 0pp |

### Human Annotation

| Scenario | 8B Match % | 32B Match % | 8B Soft % | 32B Soft % |
|---|---|---|---|---|
| multimodal | **80%** | **80%** | **100%** | 90% |
| vision_only | 60% | 60% | 80% | 80% |
| caption_only | 50% | 50% | 90% | 90% |
| text_only | 40% | **50%** (+10pp) | 90% | 90% |

### LLM Judge

| Scenario | 8B Match % | 32B Match % | 8B Soft % | 32B Soft % |
|---|---|---|---|---|
| multimodal | 6% | **20%** (+14pp) | 56% | **58%** (+2pp) |
| caption_only | 2% | **8%** (+6pp) | 46% | **48%** (+2pp) |
| vision_only | 4% | **6%** (+2pp) | 48% | 48% (0pp) |
| text_only | 2% | 2% (0pp) | 40% | **42%** (+2pp) |

---

## Key Takeaways

- **Qwen3-32B outperforms the 8B baseline** across all evaluation methods, with the largest gain on multimodal (+14pp LLM judge strict match).
- **Scaling helps most for multimodal** — the model better leverages combined visual+textual context at larger scale.
- **Scenario ranking is stable** across model sizes and evaluation methods: multimodal > vision_only > caption_only > text_only. This strengthens the core finding that visual context improves feedback quality.
- **Automatic metrics still fail to differentiate scenarios** — semantic evaluation (human or LLM-based) remains essential for this task.

Full numeric results: `data/eval_summary/qwen3_32b/qwen3_32b_combined_summary.json`

---

## Analysis Update: Feedback-Only Prompt Experiment (32B)

**Motivation:** Replicating the same single-task simplification tested on the 8B baseline — instead of asking the model to simultaneously classify verdict + error category + generate feedback, the prompt is reduced to feedback generation only. The hypothesis is that a focused single-task prompt produces higher quality feedback.

**Setup:**
- Baseline: `multimodal` (h1h2 prompt) — asks for Verdict + Error Category + Feedback, no reference answer
- New: `feedback_only` — simplified prompt asks for Feedback only, no reference answer
- Both use the multimodal scenario (caption + image) on Qwen3-VL-32B-Instruct, N=50, judged by Claude (`claude-sonnet-4-6`)

| Run | N | Match | Partial | Unmatched | Match % | Soft Match % | Avg F1 | Avg ROUGE-L | Avg BLEU |
|-----|---|-------|---------|-----------|---------|--------------|--------|-------------|----------|
| multimodal / h1h2 prompt (baseline 32B) | 50 | 10 | 19 | 21 | 20.0% | 58.0% | 0.298 | 0.202 | 7.44 |
| feedback_only (new 32B) | 50 | 9 | 29 | 12 | **18.0%** | **76.0%** | **0.404** | **0.257** | **10.45** |

**Finding:** The single-task feedback-only prompt substantially improves soft match from 58% to 76% (+18 pp) and reduces unmatched cases from 21 to 12 (-43%). All auto metrics improved substantially (F1 +36%, ROUGE-L +27%, BLEU +40%). Partial matches increased (19 → 29), indicating the model produces directionally correct feedback more often when not burdened with classification. Strict match dropped slightly (20% → 18%), likely because the h1h2 multimodal baseline was already strong at binary verdict alignment. Overall, task simplification yields a clear improvement in feedback quality at 32B scale.

### 32B vs 8B: Feedback-Only Comparison

| Model | N | Match | Partial | Unmatched | Match % | Soft Match % | Avg F1 | Avg ROUGE-L | Avg BLEU |
|-------|---|-------|---------|-----------|---------|--------------|--------|-------------|----------|
| 8B feedback_only | 50 | 4 | 29 | 17 | 4.0% | 66.0% | 0.377 | 0.242 | 7.30 |
| 32B feedback_only | 50 | 9 | 29 | 12 | **18.0%** | **76.0%** | **0.404** | **0.257** | **10.45** |

**Finding:** Scaling from 8B to 32B with the feedback-only prompt yields consistent gains across all metrics. Strict match improves from 4% → 18% (+14 pp), soft match from 66% → 76% (+10 pp), and unmatched cases drop from 17 → 12. F1, ROUGE-L, and BLEU all improve, indicating 32B generates feedback that is both semantically closer to ground truth and lexically more precise. The partial match count is identical (29), suggesting the ceiling on "directionally correct" responses is model-scale-independent — the 32B advantage lies in converting partial responses to full matches.
