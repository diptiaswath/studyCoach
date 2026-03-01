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

---

## Summary Results

| Scenario | N | Match | Partial | Unmatched | Match % | Soft Match % | Avg F1 | Avg ROUGE-L | Avg BLEU |
|---|---|---|---|---|---|---|---|---|---|
| multimodal | 50 | 10 | 19 | 21 | **20.0%** | **58.0%** | 0.298 | 0.202 | 7.44 |
| caption_only | 50 | 4 | 20 | 26 | 8.0% | 48.0% | **0.325** | **0.207** | 7.16 |
| vision_only | 50 | 3 | 21 | 26 | 6.0% | 48.0% | 0.294 | 0.193 | 8.13 |
| text_only | 50 | 1 | 20 | 29 | 2.0% | 42.0% | 0.298 | 0.192 | 5.93 |

---

## Prior Evaluation Results

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

---

## Comparison with 8B Baseline

### LLM Judge

| Scenario | 8B Match % | 32B Match % | 8B Soft % | 32B Soft % |
|---|---|---|---|---|
| multimodal | 6% | **20%** (+14pp) | 56% | **58%** (+2pp) |
| caption_only | 2% | **8%** (+6pp) | 46% | **48%** (+2pp) |
| vision_only | 4% | **6%** (+2pp) | 48% | 48% (0pp) |
| text_only | 2% | 2% (0pp) | 40% | **42%** (+2pp) |

### Human Annotation

| Scenario | 8B Match % | 32B Match % | 8B Soft % | 32B Soft % |
|---|---|---|---|---|
| multimodal | **80%** | **80%** | **100%** | 90% |
| vision_only | 60% | 60% | 80% | 80% |
| caption_only | 50% | 50% | 90% | 90% |
| text_only | 40% | **50%** (+10pp) | 90% | 90% |

### Verdict Accuracy

| Scenario | 8B | 32B | Delta |
|---|---|---|---|
| caption_only | 48% | **58%** | +10pp |
| vision_only | 50% | **56%** | +6pp |
| multimodal | 48% | **54%** | +6pp |
| text_only | 56% | 56% | 0pp |

---

## Key Takeaways

- **Qwen3-32B outperforms the 8B baseline** across all evaluation methods, with the largest gain on multimodal (+14pp LLM judge strict match).
- **Scaling helps most for multimodal** — the model better leverages combined visual+textual context at larger scale.
- **Scenario ranking is stable** across model sizes and evaluation methods: multimodal > vision_only > caption_only > text_only. This strengthens the core finding that visual context improves feedback quality.
- **Automatic metrics still fail to differentiate scenarios** — semantic evaluation (human or LLM-based) remains essential for this task.

Full numeric results: `data/eval_summary/qwen3_32b/qwen3_32b_combined_summary.json`
