# Combined Findings: Baseline + Human Evaluation

## Question

Can a small vision-language model (8B) grade student answers about scientific figures — and provide useful feedback?

This question has two parts:
1. **Verdict Classification**: Can the model correctly identify if an answer is Correct, Partially Correct, or Incorrect?
2. **Coaching**: Can the model provide feedback (explanation) that would actually help a student learn?

We ran two evaluations to answer each part.

---

## Evaluation Approach

### Phase 1: Baseline Evaluation

| Aspect | Details |
|--------|---------|
| **Question answered** | Did the model classify the verdict correctly? |
| **Method** | Automated comparison of predicted vs ground truth verdict |
| **Sample size** | 50 examples per scenario |
| **Output** | Verdict accuracy (%) |

### Phase 2: Human Evaluation

| Aspect | Details |
|--------|---------|
| **Question answered** | Was the model's feedback useful to a student? |
| **Method** | Auto metrics (F1, ROUGE-L, BLEU) + human annotation |
| **Sample size** | 50 (auto metrics) + 10 (human labels) per scenario |
| **Output** | Match / Partial / Unmatched labels |

**Why both?** A model could get the verdict wrong but still give useful feedback, or get the verdict right but give a useless explanation. We needed both evaluations to see the full picture.

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

### Why No-Answer Condition?

We also evaluated scenarios where the reference answer was provided to the model. Including it reduced variation across modalities and lowered alignment with human judgment — likely because the task shifted from reasoning-based evaluation to textual similarity matching. We focused on the no-answer condition as it better reflects a realistic coaching scenario where the model must reason from the figure itself.

---

## Results

### Phase 1: Verdict Accuracy (Baseline Eval)

*Question: Did the model classify correctly?*

| Scenario | Accuracy |
|----------|----------|
| text_only_no_answer | **56%** |
| vision_only_no_answer | 50% |
| caption_only_no_answer | 48% |
| multimodal_no_answer | 48% |

**Finding:** Text-only wins on classification. Adding visual input hurts accuracy at 8B scale.

### Phase 2a: Automated Metrics

*Question: Does predicted feedback match ground truth feedback?*

| Scenario | F1 | ROUGE-L | BLEU |
|----------|---:|--------:|-----:|
| text_only_no_answer | 0.30 | 0.19 | 5.1 |
| caption_only_no_answer | 0.29 | 0.19 | 4.8 |
| vision_only_no_answer | 0.29 | 0.18 | 6.7 |
| multimodal_no_answer | 0.30 | 0.20 | 6.2 |

**Finding:** Auto metrics show almost no difference across scenarios (~0.29-0.30 F1). If we stopped here, we'd conclude all scenarios produce similar feedback quality.

### Phase 2b: Human Evaluation

*Question: Is the feedback actually useful to a student?*

| Scenario | Match | Partial | Unmatched | Human Match % |
|----------|------:|--------:|----------:|--------------:|
| multimodal_no_answer | 8 | 2 | 0 | **80%** |
| vision_only_no_answer | 6 | 2 | 2 | 60% |
| caption_only_no_answer | 5 | 4 | 1 | 50% |
| text_only_no_answer | 4 | 5 | 1 | 40% |

**Finding:** Human evaluation reveals dramatic differences that auto metrics missed. Multimodal feedback is twice as useful as text-only (80% vs 40%).

*Soft Match = Match + Partial (feedback that is at least partially useful). Multimodal achieves 100% soft match (8+2) vs text-only at 90% (4+5).*

**Key insight:** F1/ROUGE/BLEU are poor proxies for feedback quality. Human evaluation was necessary to reveal the true differences.

---

## Conclusion: Verdict Accuracy ≠ Feedback Quality

| Metric | Winner | Loser |
|--------|--------|-------|
| **Verdict Accuracy** (Phase 1) | text_only (56%) | multimodal (48%) |
| **Feedback Quality** (Phase 2b) | multimodal (80%) | text_only (40%) |

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
- **Human eval sample size**: With n=10 per scenario, each example shifts the match rate by 10 percentage points. The observed 40pp spread (40% to 80%) may narrow as we scale annotation. However, the directional finding — multimodal explains better than text-only — is consistent with the soft match rates (100% for multimodal vs 90% for text-only).

pp = percentage points

### Dataset Composition Note

The SPIQA+ dataset error distribution:

| Error Type | % of Dataset |
|------------|--------------|
| Factual    | 48%          |
| Conceptual | 38%          |
| Omission   | 14%          |

**Why this matters for visual grounding:**

| Error Type | Example | Visual Grounding Needed |
|------------|---------|------------------------|
| **Factual** | Student says "BERT achieves 89%" but figure shows 84.6% | **High** — must see figure to verify the number |
| **Conceptual** | Student says "loss decreases throughout" but figure shows it plateaus | **Medium** — must see pattern being misinterpreted |
| **Omission** | Student mentions only one result but figure shows four | **Low** — can sometimes detect incompleteness from text alone |

**Impact on findings:**

Factual errors are exactly where visual input should help most — the model must see the figure to verify claims. With only 48% factual errors in the dataset, multimodal has fewer opportunities to demonstrate its classification advantage.

The 8pp gap favoring text-only (56% vs 48%) might **shrink** if we tested on factual errors only — those are the cases where seeing the figure is essential. This supports the "Filter to Factual errors" next step below.

---

## Next Steps

| Priority | Action | Rationale |
|----------|--------|-----------|
| High | Test stronger model (72B+) | See if vision helps classification at scale |
| High | Filter to Factual errors | Where visual grounding matters most |
| Medium | Chain-of-thought prompting | Force model to describe figure first |
| Medium | Larger eval sample (200+) | Detect smaller differences reliably |
| Medium | Add paper context (C2/C3 conditions) | Test if broader context helps reasoning-dependent errors more than factual errors |
| Medium | Analyze by figure type | Test if tables are easier than plots/schematics (structured vs spatial reasoning) |

---

## Source Files

| File | Description |
|------|-------------|
| `results/baseline_eval_summary.md` | Detailed baseline eval results |
| `results/human_eval_summary.md` | Human annotation results |
| `data/eval/*_no_answer_results.json` | Raw eval outputs |
| `baseline_findings/FINDINGS.md` | Full analysis notes |
| `HUMAN_FINDINGS.md` | Human evaluation methodology and findings |
| `human_vs_metrics_summary.csv` | Human match labels per scenario |
| `eval_metrics_summary.csv` | Auto metrics (F1, ROUGE-L, BLEU) per scenario |
