# Human Eval Summary (No Answer Condition)

Results when reference answer is withheld from the model.

## What's Being Measured

| Metric | Question it answers |
|--------|---------------------|
| Verdict Accuracy | Did model predict correct/partial/incorrect correctly? |
| Human Match | Does model's feedback align with ground truth feedback? |

**Human labels:**
- **Match** = Predicted feedback says the same thing as ground truth
- **Partial** = Feedback is somewhat useful / partially aligned
- **Unmatched** = Feedback misses the point or is wrong

---

## 1. Auto Metrics (50 samples each)

| Scenario | F1 | ROUGE-L | BLEU |
|----------|---:|--------:|-----:|
| text_only_no_answer | 0.30 | 0.19 | 5.1 |
| caption_only_no_answer | 0.29 | 0.19 | 4.8 |
| vision_only_no_answer | 0.29 | 0.18 | 6.7 |
| multimodal_no_answer | 0.30 | 0.20 | 6.2 |

---

## 2. Human Annotations (10 samples each)

| Scenario | Match | Partial | Unmatched | Match % | Soft Match % |
|----------|------:|--------:|----------:|--------:|-------------:|
| multimodal_no_answer | 8 | 2 | 0 | **80%** | 100% |
| vision_only_no_answer | 6 | 2 | 2 | 60% | 80% |
| caption_only_no_answer | 5 | 4 | 1 | 50% | 90% |
| text_only_no_answer | 4 | 5 | 1 | 40% | 90% |

*Soft Match = Match + Partial (feedback at least partially useful)*

---

## Key Takeaways

| Finding | Implication |
|---------|-------------|
| multimodal_no_answer has 80% human match | Visual input helps feedback quality when answer withheld |
| text_only_no_answer has lowest human match (40%) | Text-only struggles to give good explanations |
| Auto metrics don't correlate well with human judgment | F1/ROUGE/BLEU aren't great proxies for feedback quality |
| Soft match is high across all (80-100%) | Most feedback is at least partially useful |

---

## Key Insight: Verdict Accuracy ≠ Feedback Quality

| Scenario | Verdict Accuracy | Human Match |
|----------|------------------|-------------|
| text_only_no_answer | **56%** (best) | 40% |
| multimodal_no_answer | 48% | **80%** (best) |

- **text_only** wins on verdict classification but gives poor explanations
- **multimodal** loses on verdict but gives better feedback when it explains

**Takeaway:** Visual input helps the model *explain* better, even if it doesn't help classify better at 8B scale.
