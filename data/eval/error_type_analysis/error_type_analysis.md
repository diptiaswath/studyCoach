# Error Type Analysis: H3 Hypothesis Test

## Hypothesis

**H3:** Factual errors detected more reliably than conceptual errors,
and visual context helps conceptual errors more than factual errors.

## Accuracy Matrix

| Error Type | text_only | caption_only | vision_only | multimodal | Avg |
|------------|-----------|--------------|-------------|------------|-----|
| factual | 51.9% (27/52) | 40.4% (21/52) | 34.6% (18/52) | 30.8% (16/52) | 39.4% | |
| conceptual | 63.4% (26/41) | 61.0% (25/41) | 43.9% (18/41) | 48.8% (20/41) | 54.3% | |
| omission | 53.3% (8/15) | 33.3% (5/15) | 46.7% (7/15) | 46.7% (7/15) | 45.0% | |

## Context Benefit (Δ = multimodal - text_only)

| Error Type | Δ (pp) | Interpretation |
|------------|--------|----------------|
| factual | -21.2 | Visual context **hurts** by 21.2pp |
| conceptual | -14.6 | Visual context **hurts** by 14.6pp |
| omission | -6.7 | Visual context **hurts** by 6.7pp |

**Key finding:** All Δ values are negative — visual context hurts verdict accuracy for ALL error types at 8B scale.

## H3 Hypothesis Test

### Part 1: Factual > Conceptual?

- Factual avg: 39.4%
- Conceptual avg: 54.3%
- **Result: FAIL**

### Part 2: Δ_conceptual > Δ_factual?

- Δ_factual: -21.2pp
- Δ_conceptual: -14.6pp
- **Result: PASS (technically, but misleading)**

**Clarification:** This "passes" only because -14.6 > -21.2 (less negative). Visual context does NOT help conceptual errors — it **hurts** them by 14.6pp. The "pass" only means it hurts factual errors even more (-21.2pp).

---

## Summary

**H3 Overall: FAIL**

Both parts of the hypothesis are effectively wrong:

1. **Part 1 (FAIL):** Factual errors are NOT detected more reliably — conceptual errors are easier to classify (54.3% vs 39.4%)

2. **Part 2 (technically PASS, effectively FAIL):** Visual context does NOT help conceptual errors — it hurts them by 14.6pp. The "pass" only indicates it hurts factual errors even more (-21.2pp).

### Interpretation

At 8B model scale:
- **Text-only performs best** for all error types
- **Visual input is a distraction** for verdict classification
- **Factual errors are hardest** to detect, despite being "visually obvious" — the model struggles to ground specific values from figures
- **Conceptual errors are easier** — possibly because the model can detect logical inconsistencies in text without needing precise visual grounding

This aligns with the baseline finding: text_only (56%) > multimodal (48%) for overall verdict accuracy.

### Note

This evaluation tests **verdict classification only**, not feedback quality. Prior human evaluation showed multimodal produces better explanations (80% match) despite worse classification (48% accuracy).