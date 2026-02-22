# Human Evaluation vs Automatic Metrics

## Setup

- 8 model configurations
- **N = 10 annotated examples per configuration**
- 3 human labels:
  - match (1)
  - partial (2)
  - unmatched (0)

We compare human judgments against automatic metrics (F1, ROUGE-L, BLEU).

---

## Summary Results (Strict + Soft Match)

| Scenario               | N   | Match | Partial | Unmatched | Match % | Soft Match % | Avg F1 | Avg ROUGE-L | Avg BLEU |
| ---------------------- | --- | ----- | ------- | --------- | ------- | ------------ | ------ | ----------- | -------- |
| caption_only_no_answer | 10  | 5     | 4       | 1         | 50%     | 90%          | 0.3008 | 0.1919      | 6.60     |
| caption_only           | 10  | 4     | 6       | 0         | 40%     | 100%         | 0.2963 | 0.1756      | 5.57     |
| multimodal_no_answer   | 10  | 8     | 2       | 0         | 80%     | 100%         | 0.3311 | 0.2166      | 9.58     |
| multimodal             | 10  | 4     | 5       | 1         | 40%     | 90%          | 0.3023 | 0.1866      | 6.70     |
| text_only_no_answer    | 10  | 4     | 5       | 1         | 40%     | 90%          | 0.2542 | 0.1717      | 6.02     |
| text_only              | 10  | 2     | 4       | 4         | 20%     | 60%          | 0.2482 | 0.1488      | 2.91     |
| vision_only_no_answer  | 10  | 6     | 2       | 2         | 60%     | 80%          | 0.2597 | 0.1536      | 7.11     |
| vision_only            | 10  | 3     | 6       | 1         | 30%     | 90%          | 0.2822 | 0.1803      | 5.35     |

Full numeric results:
`data/eval_summary/human_vs_metrics_summary.csv`

---

## Key Observations

- Multimodal_no_answer achieves the highest strict match rate (80%).
- Removing the reference answer increases modality differences.
- Automatic metrics correlate loosely but not perfectly with human judgments.
- With small N (10 per model), results are directional rather than definitive.

---
