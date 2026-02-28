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

### Why Text-Only "Wins" (But For The Wrong Reasons)

In **text_only (C1)**, the model receives only the question and student answer — no figure, no caption, no reference answer. So the model evaluates based on:

1. **Pre-trained knowledge** — does the student's claim sound plausible given what the model learned during training?
2. **Internal consistency** — does the answer contradict itself or make logical errors?
3. **Language patterns** — does the phrasing suggest confidence/hedging/confusion?

**The tradeoff:**
- Text-only classifies better but is doing **plausibility checking**, not **verification against the figure**
- It can't actually confirm "BERT achieves 84.6%" — it's just judging if the claim sounds reasonable
- Multimodal classifies worse but **explains better** because it actually uses the figure

**This is why scaling matters:** We want the model to verify claims against the actual figure, not just guess based on training knowledge.

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

**H3 Tested This — Results Were Surprising:**

We predicted factual errors would benefit most from visual context (high grounding needed). The opposite occurred:

| Error Type | Verdict Δ | Feedback Δ | Verdict Effect | Feedback Effect |
|------------|-----------|------------|----------------|-----------------|
| Factual | -21.2pp | +21.2pp | Hurts most | Helps least |
| Conceptual | -14.6pp | +31.7pp | Hurts medium | Helps medium |
| Omission | -6.7pp | +33.3pp | Hurts least | Helps most |

**Interpretation:**

- **Verdict accuracy:** Visual context hurts factual errors *most* (-21.2pp), not least. The 8B model cannot reliably ground specific values from figures — visual input adds noise rather than helping verification.

- **Feedback quality:** Visual context helps omission errors *most* (+33.3pp). The model can use the figure to explain what was missing, even if it can't classify correctly.

- **The paradox:** High visual grounding *needed* ≠ high visual grounding *achieved*. At 8B scale, the model struggles most with precisely the errors that require detailed visual verification.

---

## H3 Hypothesis Test: Verdict Accuracy by Error Type

We ran a full evaluation on all 108 error examples (excluding correct answers) to test H3.

**H3:** Factual errors detected more reliably than conceptual errors, and visual context helps conceptual errors more than factual errors.

### Accuracy by Error Type × Scenario

| Error Type | text_only | caption_only | vision_only | multimodal | Avg |
|------------|-----------|--------------|-------------|------------|-----|
| factual (n=52) | 51.9% | 40.4% | 34.6% | 30.8% | 39.4% |
| conceptual (n=41) | 63.4% | 61.0% | 43.9% | 48.8% | 54.3% |
| omission (n=15) | 53.3% | 33.3% | 46.7% | 46.7% | 45.0% |

### Context Benefit (Δ = multimodal - text_only)

| Error Type | Δ (pp) | Interpretation |
|------------|--------|----------------|
| factual | -21.2 | Visual context **hurts** by 21.2pp |
| conceptual | -14.6 | Visual context **hurts** by 14.6pp |
| omission | -6.7 | Visual context **hurts** by 6.7pp |

**Key finding:** All Δ values are negative — visual context hurts verdict accuracy for ALL error types at 8B scale.

### H3 Results

| Part | Hypothesis | Result |
|------|------------|--------|
| 1 | Factual > Conceptual? | **FAIL** (39.4% < 54.3%) |
| 2 | Context helps conceptual more? | **FAIL** (hurts both; -14.6pp vs -21.2pp) |

**H3 Overall: FAIL**

### Interpretation

1. **Part 1 failed:** Factual errors are NOT detected more reliably — conceptual errors are easier to classify (54.3% vs 39.4%). The model struggles to ground specific values from figures.

2. **Part 2 failed:** Visual context does NOT help conceptual errors — it hurts them by 14.6pp. It just hurts factual errors even more (-21.2pp).

3. **Text-only performs best** for all error types — visual input is a distraction for verdict classification at 8B scale.

---

## H3 Hypothesis Test: Feedback Quality by Error Type

We also evaluated feedback quality using Claude as LLM judge (Match/Partial/Unmatched).

### Match Rate by Error Type × Scenario

| Error Type | text_only | caption_only | vision_only | multimodal | Avg | Δ |
|------------|-----------|--------------|-------------|------------|-----|---|
| factual (n=52) | 13.5% | 25.0% | 28.8% | 34.6% | 25.5% | **+21.2pp** |
| conceptual (n=41) | 26.8% | 51.2% | 36.6% | 58.5% | 43.3% | **+31.7pp** |
| omission (n=15) | 0.0% | 0.0% | 26.7% | 33.3% | 15.0% | **+33.3pp** |

### Context Benefit for Feedback (Δ = multimodal - text_only)

| Error Type | Δ Match | Interpretation |
|------------|---------|----------------|
| factual | +21.2pp | Visual context **helps** |
| conceptual | +31.7pp | Visual context **helps** |
| omission | +33.3pp | Visual context **helps** |

**Key finding:** All Δ values are positive — visual context helps feedback quality for ALL error types.

### H3 Results (Feedback)

| Part | Hypothesis | Result |
|------|------------|--------|
| 1 | Factual > Conceptual? | **FAIL** (25.5% < 43.3%) |
| 2 | Context helps conceptual more than factual? | **PASS** (+31.7pp > +21.2pp) |

---

## H3 Summary: Opposite Effects for Verdict vs Feedback

| Metric | Factual Δ | Conceptual Δ | Omission Δ | Visual Context Effect |
|--------|-----------|--------------|------------|----------------------|
| **Verdict Accuracy** | -21.2pp | -14.6pp | -6.7pp | **Hurts all** |
| **Feedback Quality** | +21.2pp | +31.7pp | +33.3pp | **Helps all** |

### Key Insight

At 8B scale, visual input has **opposite effects** on verdict vs feedback:
- **Verdict:** Visual context hurts classification for all error types (model gets distracted)
- **Feedback:** Visual context helps explanation for all error types (model uses figure to explain)

This confirms the baseline finding: **"The model classifies better without images but explains better with them."**

### H3 Overall Verdict

| H3 Part | Verdict Accuracy | Feedback Quality |
|---------|------------------|------------------|
| Part 1: Factual > Conceptual? | FAIL | FAIL |
| Part 2: Context helps conceptual more? | FAIL | **PASS** |

---

## Next Steps

| Priority | Action | Rationale |
|----------|--------|-----------|
| High | Test stronger model (72B+) | See if vision helps classification at scale |
| High | Filter to Factual errors | Where visual grounding matters most |
| Medium | Chain-of-thought prompting | Force model to describe figure first |
| Medium | Larger eval sample (200+) | Detect smaller differences reliably |
| Optional | Add paper context (C5 condition) | Test if relevant paragraphs help reasoning-dependent errors more than factual errors |
| Medium | Analyze by figure type | Test if tables are easier than plots/schematics (structured vs spatial reasoning) |

### Why Test Stronger Model (72B+)?

At 8B scale, visual input hurts classification (56% → 48%) but helps explanation (40% → 80%). This suggests the model *can* use visual information for reasoning but can't yet use it reliably for verdict classification.

**The hypothesis:** Larger models have better visual grounding capabilities. A 72B+ model (e.g., Qwen2.5-VL-72B, GPT-4o) might show visual input helping *both* metrics — closing the classification gap while maintaining the explanation advantage.

**What we'd look for:**
- Multimodal accuracy ≥ text-only accuracy (currently 8pp behind)
- Maintained or improved feedback quality (currently 80% human match)

### Why Filter to Factual Errors?

*See Dataset Composition Note above.* Factual errors (48% of dataset) require high visual grounding — the model must read actual values from the figure.

**Update (H3 tested):** We tested this hypothesis and found the opposite — factual errors are actually *harder* to detect (39.4% avg) than conceptual errors (54.3% avg). Visual context hurts factual errors most severely (-21.2pp). This suggests the 8B model cannot reliably ground specific values from figures, making factual verification particularly difficult.

### Why Chain-of-Thought Prompting?

The surprising finding was that visual input *hurts* classification but *helps* explanation. One theory: the 8B model doesn't properly attend to visual details during classification — it gets distracted by the image without systematically extracting information from it.

**Current approach (direct verdict):**
```
Prompt: Here's the figure, student answer, question. Give verdict.
Model: Verdict = Incorrect
```

**Chain-of-thought approach (describe first, then judge):**
```
Prompt: First describe what you see in the figure. Then evaluate the student's answer.
Model:
  Step 1: The figure shows a bar chart with BERT at 84.6%, RoBERTa at 87.2%...
  Step 2: The student claims "BERT achieves 89%" — this contradicts the figure.
  Verdict = Incorrect
```

**The bet:** If the model must explicitly describe the figure before judging, it might use visual information more effectively for classification — not just explanation. This could close the gap between text-only (56%) and multimodal (48%) accuracy.

### Why Larger Eval Sample (200+)?

With n=50, differences <5pp are noise (~7pp confidence intervals). The current 8pp gap (text-only 56% vs multimodal 48%) is borderline significant.

**With n=200+:**
- Confidence intervals shrink to ~3-4pp
- Can reliably detect 5pp differences
- Can compare vision_only vs caption_only (currently 2pp gap = noise)

This is especially important for testing H3/H4 hypotheses with a competitive baseline — we need statistical power to detect meaningful differences.

### Why Add Paper Context (C5 Condition)?

The current 4 scenarios (C1-C4) vary caption and image, but don't test paper context. Adding relevant paragraphs creates a new condition:

| Condition | Input | Status |
|-----------|-------|--------|
| C1 | text_only (Q + Student Answer) | Tested |
| C2 | caption_only (Q + Caption + Student Answer) | Tested |
| C3 | vision_only (Q + Image + Student Answer) | Tested |
| C4 | multimodal (Q + Caption + Image + Student Answer) | Tested |
| **C5** | **multimodal + relevant paragraphs** | **Optional** |

**Future hypothesis:** Paper context may help conceptual/omission errors more than factual errors, since these require understanding experimental setup beyond what the figure shows.

**Motivation from H3 findings:**
- Visual context helped feedback quality most for **omission errors (+33.3pp)** — the model can explain what's missing when it sees the figure
- Paper context might further improve this by providing the full scope of what should be included
- Visual context hurt verdict accuracy most for **factual errors (-21.2pp)** — paper context might help ground specific values that the 8B model struggles to read from figures

### Why Analyze by Figure Type?

The dataset contains different figure types with different reasoning demands:

| Figure Type | % of Dataset | Reasoning Type |
|-------------|--------------|----------------|
| Tables | ~44% | Structured lookup — explicit rows/columns |
| Plots | ~28% | Trend reading — spatial patterns |
| Schematics | ~17% | Architecture understanding — distributed relationships |
| Other | ~11% | Mixed |

**Hypothesis (H4):** Tables should be easiest (explicit, localized information), while architecture diagrams should be hardest (require understanding spatial relationships between components).

**Why this matters:** If accuracy varies significantly by figure type, Study Coach could route different figure types to different models or prompting strategies.

---

## Source Files

| File | Description |
|------|-------------|
| `results/baseline_eval_summary.md` | Detailed baseline eval results |
| `results/human_eval_summary.md` | Human annotation results |
| `data/eval/*_no_answer_results.json` | Raw eval outputs |
| `data/eval/error_type_analysis/` | H3 hypothesis test results (verdict by error type) |
| `baseline_findings/FINDINGS.md` | Full analysis notes |
| `HUMAN_FINDINGS.md` | Human evaluation methodology and findings |
| `human_vs_metrics_summary.csv` | Human match labels per scenario |
| `eval_metrics_summary.csv` | Auto metrics (F1, ROUGE-L, BLEU) per scenario |
