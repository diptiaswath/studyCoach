# H4 Hypothesis Test: Feedback Quality by Figure Type

## Method

Using Claude (claude-sonnet-4-20250514) as judge to evaluate feedback quality.
Each feedback is classified as Match / Partial / Unmatched.

## Match Rate by Figure Type × Scenario

| Figure Type | text_only | caption_only | vision_only | multimodal | Avg | Δ |
|-------------|-----------|--------------|-------------|------------|-----|---|
| table | 54.5% (12/22) | 59.1% (13/22) | 50.0% (11/22) | 59.1% (13/22) | 55.7% | +4.5pp |
| plot | 31.6% (6/19) | 36.8% (7/19) | 31.6% (6/19) | 26.3% (5/19) | 31.6% | -5.3pp |
| schematic | 44.4% (4/9) | 55.6% (5/9) | 44.4% (4/9) | 55.6% (5/9) | 50.0% | +11.1pp |
| other | 0.0% (0/0) | 0.0% (0/0) | 0.0% (0/0) | 0.0% (0/0) | 0.0% | +0.0pp |

## Soft Match Rate (Match + Partial)

| Figure Type | text_only | caption_only | vision_only | multimodal | Δ |
|-------------|-----------|--------------|-------------|------------|---|
| table | 77.3% (17/22) | 77.3% (17/22) | 68.2% (15/22) | 72.7% (16/22) | -4.5pp |
| plot | 57.9% (11/19) | 52.6% (10/19) | 57.9% (11/19) | 52.6% (10/19) | -5.3pp |
| schematic | 66.7% (6/9) | 66.7% (6/9) | 66.7% (6/9) | 66.7% (6/9) | +0.0pp |
| other | 0.0% (0/0) | 0.0% (0/0) | 0.0% (0/0) | 0.0% (0/0) | +0.0pp |

## Context Benefit for Feedback (Δ = multimodal - text_only)

| Figure Type | Δ Match | Δ Soft Match | Helps? |
|-------------|---------|--------------|--------|
| table | +4.5pp | -4.5pp | Yes |
| plot | -5.3pp | -5.3pp | No |
| schematic | +11.1pp | +0.0pp | Yes |
| other | +0.0pp | +0.0pp | No |

## H4 Hypothesis Test (Feedback)

**H4:** Tables easier than Schematics?

- Table avg (Match %): 55.7%
- Schematic avg (Match %): 50.0%
- **Result: PASS**

## Summary

Visual context **helps** feedback quality for some figure types.