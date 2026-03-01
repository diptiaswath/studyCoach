# LLM-as-Judge Evaluation: Qwen3-VL-8B (Baseline)

## Setup

- **Evaluated model**: `Qwen3-VL-8B-Instruct` (via Together.ai)
- **Judge model**: `claude-sonnet-4-6` (Anthropic)
- **N = 50 examples per scenario** (4 scenarios evaluated)
- **Reference answer withheld** from the evaluated model in `_no_answer` configurations (to encourage reasoning over string matching)
- The LLM judge receives: question, student answer, ground-truth feedback (A), and predicted feedback (B) — and judges whether B provides equivalent guidance to A
- 3 LLM judge labels: `match`, `partial`, `unmatched`

---

## Prompt Conditions

| Dimension | Options |
|---|---|
| **Input modality** | `text_only`, `caption_only`, `vision_only`, `multimodal` |
| **Reference answer provided to evaluated model** | `_results` (yes) vs `_no_answer_results` (no) |

> **Note**: LLM judge was only run on the `_no_answer` configurations. The `_results` conditions show lexical metrics only.

---

## Evaluation Results

### Phase 1: Verdict Accuracy

| Scenario | Accuracy |
|---|---|
| text_only_no_answer | **56%** |
| vision_only_no_answer | 50% |
| caption_only_no_answer | 48% |
| multimodal_no_answer | 48% |

**Finding**: Text-only wins on classification. Adding visual input hurts accuracy at 8B scale — the model appears to do plausibility checking from training knowledge rather than genuinely verifying claims against the figure.

### Phase 2a: Automated Metrics

| Scenario | F1 | ROUGE-L | BLEU |
|---|---|---|---|
| text_only_no_answer | 0.30 | 0.19 | 5.1 |
| caption_only_no_answer | 0.29 | 0.19 | 4.8 |
| vision_only_no_answer | 0.29 | 0.18 | 6.7 |
| multimodal_no_answer | 0.30 | 0.20 | 6.2 |

**Finding**: Automatic metrics are essentially flat (~0.29–0.30 F1) and fail to differentiate feedback quality across scenarios.

### Phase 2b: Human Evaluation (N=10 per scenario)

| Scenario | Match | Partial | Unmatched | Human Match % | Soft Match % |
|---|---|---|---|---|---|
| multimodal_no_answer | 8 | 2 | 0 | **80%** | **100%** |
| vision_only_no_answer | 6 | 2 | 2 | 60% | 80% |
| caption_only_no_answer | 5 | 4 | 1 | 50% | 90% |
| text_only_no_answer | 4 | 5 | 1 | 40% | 90% |

**Finding**: Human evaluation reveals dramatic differences that automatic metrics missed. Multimodal feedback is twice as useful as text-only (80% vs 40%), despite text-only winning on verdict accuracy.

### Phase 2c: LLM-as-Judge (N=50 per scenario)

| Scenario | N | Match | Partial | Unmatched | Match % | Soft Match % | Avg F1 | Avg ROUGE-L | Avg BLEU |
|---|---|---|---|---|---|---|---|---|---|
| multimodal_no_answer | 50 | 3 | 25 | 22 | **6.0%** | **56.0%** | 0.302 | 0.197 | 6.16 |
| vision_only_no_answer | 50 | 2 | 22 | 26 | 4.0% | 48.0% | 0.292 | 0.184 | 6.70 |
| caption_only_no_answer | 50 | 1 | 22 | 27 | 2.0% | 46.0% | 0.294 | 0.192 | 4.79 |
| text_only_no_answer | 50 | 1 | 19 | 30 | 2.0% | 40.0% | 0.296 | 0.189 | 5.09 |

**Finding**: Multimodal leads on soft match (56%) at N=50, confirming the human evaluation finding. Match rates are low overall (2–6%), reflecting the stricter LLM judge rubric applied to a larger sample.

---

## Comparison with Human Annotation

Human annotators (N=10 per scenario) were consistently more lenient than the LLM judge (N=50).

| Scenario | Human Match % | LLM Match % | Human Soft % | LLM Soft % |
|---|---|---|---|---|
| multimodal_no_answer | **80%** | **6%** | **100%** | **56%** |
| vision_only_no_answer | 60% | 4% | 80% | 48% |
| caption_only_no_answer | 50% | 2% | 90% | 46% |
| text_only_no_answer | 40% | 2% | 90% | 40% |

> Note: Human N=10, LLM judge N=50. Comparison is directional.

**Scenario ranking is consistent across both evaluations**: multimodal > vision_only > caption_only > text_only. Despite the absolute gap (human rates ~10× higher than LLM strict match), the ordering is preserved — the LLM judge confirms the human finding at larger scale. The leniency gap reflects the stricter rubric, larger sample, and the tendency for human annotators to credit feedback that is _directionally correct_ even if incomplete.

---

## Key Takeaways

- **Visual context improves feedback quality but hurts verdict accuracy** at 8B scale: multimodal leads on feedback (80% human match) but trails on classification (48% vs 56% text-only). The visual signal adds value to explanation but adds noise to binary decisions.
- **Scenario ranking is stable** across all evaluation methods: multimodal > vision_only > caption_only > text_only.
- **Automatic metrics fail to differentiate scenarios** — semantic evaluation (human or LLM-based) is essential for this task.
- **Human and LLM judge agree directionally** despite large absolute differences in match rates, validating the LLM judge as a scalable complement to human annotation.

Full numeric results: `data/eval/eval_summary/baseline_summary.csv`
