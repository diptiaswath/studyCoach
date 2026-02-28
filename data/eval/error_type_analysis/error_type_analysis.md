# Error Type Analysis: H3 Hypothesis Test

## Hypothesis

**H3:** Factual errors detected more reliably than conceptual errors,
and visual context helps conceptual errors more than factual errors.

## Accuracy Matrix

| Error Type | text_only | caption_only | vision_only | multimodal | Avg |
|------------|-----------|--------------|-------------|------------|-----|
| factual | 33.3% (1/3) | 0.0% (0/3) | 33.3% (1/3) | 0.0% (0/3) | 16.7% | |
| conceptual | 33.3% (1/3) | 33.3% (1/3) | 0.0% (0/3) | 0.0% (0/3) | 16.7% | |
| omission | 33.3% (1/3) | 0.0% (0/3) | 33.3% (1/3) | 33.3% (1/3) | 25.0% | |

## Context Benefit (Δ = multimodal - text_only)

| Error Type | Δ (pp) |
|------------|--------|
| factual | -33.3 |
| conceptual | -33.3 |
| omission | +0.0 |

## H3 Hypothesis Test

### Part 1: Factual > Conceptual?

- Factual avg: 16.7%
- Conceptual avg: 16.7%
- **Result: FAIL**

### Part 2: Δ_conceptual > Δ_factual?

- Δ_factual: -33.3pp
- Δ_conceptual: -33.3pp
- **Result: FAIL**

---

**H3 Overall: FAIL**