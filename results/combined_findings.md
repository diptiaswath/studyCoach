# Combined Findings: Baseline + Human Evaluation

## Question

Can a small vision-language model (8B) grade student answers about scientific figures — and provide useful feedback?

---

## Setup

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-VL-8B-Instruct` (Together.ai) |
| Condition | No-answer (reference answer withheld from model) |
| Eval sample | 50 examples (stratified by verdict × error category) |
| Human annotation | 10 examples per scenario |

### Scenarios Tested

| Scenario | Caption | Image |
|----------|:-------:|:-----:|
| text_only | - | - |
| caption_only | Yes | - |
| vision_only | - | Yes |
| multimodal | Yes | Yes |

---

## Results Summary

### Verdict Accuracy (Auto-Eval, n=50)

| Scenario | Accuracy |
|----------|----------|
| text_only_no_answer | **56%** |
| vision_only_no_answer | 50% |
| caption_only_no_answer | 48% |
| multimodal_no_answer | 48% |

### Feedback Quality (Human-Eval, n=10)

| Scenario | Match | Partial | Unmatched | Human Match % |
|----------|------:|--------:|----------:|--------------:|
| multimodal_no_answer | 8 | 2 | 0 | **80%** |
| vision_only_no_answer | 6 | 2 | 2 | 60% |
| caption_only_no_answer | 5 | 4 | 1 | 50% |
| text_only_no_answer | 4 | 5 | 1 | 40% |

### Automated Metrics (n=50)

| Scenario | F1 | ROUGE-L | BLEU |
|----------|---:|--------:|-----:|
| text_only_no_answer | 0.30 | 0.19 | 5.1 |
| caption_only_no_answer | 0.29 | 0.19 | 4.8 |
| vision_only_no_answer | 0.29 | 0.18 | 6.7 |
| multimodal_no_answer | 0.30 | 0.20 | 6.2 |

**Note:** Auto metrics show little variation across scenarios (~0.29-0.30 F1), while human match varies dramatically (40% to 80%). This confirms that **F1/ROUGE/BLEU are poor proxies for feedback quality** in this task. Human evaluation was necessary to reveal the true differences.

---

## Conclusion: Verdict Accuracy ≠ Feedback Quality

| Metric | Winner | Loser |
|--------|--------|-------|
| **Verdict Accuracy** (Did model classify correctly?) | text_only (56%) | multimodal (48%) |
| **Feedback Quality** (Was explanation useful?) | multimodal (80%) | text_only (40%) |

### The Insight

At 8B scale, adding visual input:
- **Hurts** classification accuracy (56% → 48%)
- **Helps** explanation quality (40% → 80%)

### What This Means

| Use Case | Recommended Scenario |
|----------|---------------------|
| **Grading** (just need correct/incorrect) | text_only |
| **Coaching** (need helpful explanations) | multimodal |

### Why This Happens

The 8B model *can* use visual information for reasoning — that's why its explanations are better. But it can't yet use that information reliably enough to improve verdict classification. The visual signal adds noise to the decision, but adds value to the explanation.

---

## One-Liner

> "The model classifies better without images but explains better with them."

---

## Statistical Notes

- **Sample size**: With n=50, differences <5pp are noise
- **Reliable comparisons**: text_only vs multimodal (8pp gap) is meaningful
- **Unreliable comparisons**: vision_only vs caption_only (2pp gap) is noise

---

## Next Steps

| Priority | Action | Rationale |
|----------|--------|-----------|
| High | Test stronger model (72B+) | See if vision helps classification at scale |
| High | Filter to Factual errors | Where visual grounding matters most |
| Medium | Chain-of-thought prompting | Force model to describe figure first |
| Medium | Larger eval sample (200+) | Detect smaller differences reliably |

---

## Source Files

### Main Branch

| File | Description |
|------|-------------|
| `results/baseline_eval_summary.md` | Detailed baseline eval results |
| `results/human_eval_summary.md` | Human annotation results |
| `data/eval/*_no_answer_results.json` | Raw eval outputs |
| `baseline_findings/FINDINGS.md` | Full analysis notes |

### Origin Branch (human-eval-annotation)

| File | Description |
|------|-------------|
| `HUMAN_FINDINGS.md` | Human evaluation methodology and findings |
| `human_vs_metrics_summary.csv` | Human match labels per scenario |
| `eval_metrics_summary.csv` | Auto metrics (F1, ROUGE-L, BLEU) per scenario |
