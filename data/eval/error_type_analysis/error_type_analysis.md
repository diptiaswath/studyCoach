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

| Error Type | Δ (pp) |
|------------|--------|
| factual | -21.2 |
| conceptual | -14.6 |
| omission | -6.7 |

## H3 Hypothesis Test

### Part 1: Factual > Conceptual?

- Factual avg: 39.4%
- Conceptual avg: 54.3%
- **Result: FAIL**

### Part 2: Δ_conceptual > Δ_factual?

- Δ_factual: -21.2pp
- Δ_conceptual: -14.6pp
- **Result: PASS**

---

**H3 Overall: PARTIAL**