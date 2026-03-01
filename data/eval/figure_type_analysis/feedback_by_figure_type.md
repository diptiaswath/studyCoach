# H4 Hypothesis Test: Feedback Quality by Figure Type

## Method

Using Claude (claude-sonnet-4-20250514) as judge to evaluate feedback quality.
Each feedback is classified as Match / Partial / Unmatched.

## Match Rate by Figure Type × Scenario

| Figure Type | text_only | caption_only | vision_only | multimodal | Avg | Δ |
|-------------|-----------|--------------|-------------|------------|-----|---|
| table | 1.4% (1/70) | 4.3% (3/70) | 11.4% (8/70) | 8.6% (6/70) | 6.4% | +7.1pp |
| plot | 13.2% (9/68) | 25.0% (17/68) | 16.2% (11/68) | 32.4% (22/68) | 21.7% | +19.1pp |
| schematic | 25.0% (8/32) | 37.5% (12/32) | 28.1% (9/32) | 50.0% (16/32) | 35.2% | +25.0pp |
| other | 0.0% (0/4) | 0.0% (0/4) | 25.0% (1/4) | 50.0% (2/4) | 18.8% | +50.0pp |

## Soft Match Rate (Match + Partial)

| Figure Type | text_only | caption_only | vision_only | multimodal | Δ |
|-------------|-----------|--------------|-------------|------------|---|
| table | 20.0% (14/70) | 15.7% (11/70) | 24.3% (17/70) | 24.3% (17/70) | +4.3pp |
| plot | 27.9% (19/68) | 33.8% (23/68) | 32.4% (22/68) | 36.8% (25/68) | +8.8pp |
| schematic | 53.1% (17/32) | 59.4% (19/32) | 50.0% (16/32) | 62.5% (20/32) | +9.4pp |
| other | 0.0% (0/4) | 25.0% (1/4) | 50.0% (2/4) | 50.0% (2/4) | +50.0pp |

## Context Benefit for Feedback (Δ = multimodal - text_only)

| Figure Type | Δ Match | Δ Soft Match | Helps? |
|-------------|---------|--------------|--------|
| table | +7.1pp | +4.3pp | Yes |
| plot | +19.1pp | +8.8pp | Yes |
| schematic | +25.0pp | +9.4pp | Yes |
| other | +50.0pp | +50.0pp | Yes |

## H4 Hypothesis Test (Feedback)

**H4:** Tables easier than Schematics?

- Table avg (Match %): 6.4%
- Schematic avg (Match %): 35.2%
- **Result: FAIL**

## Summary

Visual context **helps** feedback quality for some figure types.