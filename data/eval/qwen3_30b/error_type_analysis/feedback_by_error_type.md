# H3 Hypothesis Test: Feedback Quality by Error Type

## Method

Using Claude (claude-sonnet-4-20250514) as judge to evaluate feedback quality.
Each feedback is classified as Match / Partial / Unmatched.

## Match Rate by Error Type × Scenario

| Error Type | text_only | caption_only | vision_only | multimodal | Δ |
|------------|-----------|--------------|-------------|------------|---|
| factual | 13.5% (7/52) | 26.9% (14/52) | 28.8% (15/52) | 34.6% (18/52) | +21.2pp |
| conceptual | 26.8% (11/41) | 48.8% (20/41) | 36.6% (15/41) | 56.1% (23/41) | +29.3pp |
| omission | 0.0% (0/15) | 0.0% (0/15) | 26.7% (4/15) | 26.7% (4/15) | +26.7pp |

## Soft Match Rate (Match + Partial)

| Error Type | text_only | caption_only | vision_only | multimodal | Δ |
|------------|-----------|--------------|-------------|------------|---|
| factual | 30.8% (16/52) | 38.5% (20/52) | 42.3% (22/52) | 50.0% (26/52) | +19.2pp |
| conceptual | 65.9% (27/41) | 78.0% (32/41) | 61.0% (25/41) | 70.7% (29/41) | +4.9pp |
| omission | 46.7% (7/15) | 33.3% (5/15) | 46.7% (7/15) | 53.3% (8/15) | +6.7pp |

## Context Benefit for Feedback (Δ = multimodal - text_only)

| Error Type | Δ Match | Δ Soft Match | Helps? |
|------------|---------|--------------|--------|
| factual | +21.2pp | +19.2pp | Yes |
| conceptual | +29.3pp | +4.9pp | Yes |
| omission | +26.7pp | +6.7pp | Yes |

## H3 Hypothesis Test (Feedback)

**H3 Part 2:** Does visual context help conceptual feedback more than factual?

- Δ_factual (Match %): +21.2pp
- Δ_conceptual (Match %): +29.3pp
- **Result: PASS**

## Summary

Visual context **helps** feedback quality for some error types.

### Comparison with Verdict Accuracy

| Metric | Visual Context Effect |
|--------|----------------------|
| Verdict Accuracy | Hurts all error types (Δ negative) |
| Feedback Quality | Helps some |

This aligns with the baseline finding: visual input hurts classification but may help explanation.