# H4 Hypothesis Test: Feedback Quality by Figure Type

## Method

Using Claude (claude-sonnet-4-20250514) as judge to evaluate feedback quality.
Each feedback is classified as Match / Partial / Unmatched.

## Match Rate by Figure Type × Scenario

| Figure Type | text_only | caption_only | vision_only | multimodal | Avg | Δ |
|-------------|-----------|--------------|-------------|------------|-----|---|
| table | 4.3% (3/70) | 7.1% (5/70) | 17.1% (12/70) | 20.0% (14/70) | 12.1% | +15.7pp |
| plot | 10.3% (7/68) | 32.4% (22/68) | 19.1% (13/68) | 44.1% (30/68) | 26.5% | +33.8pp |
| schematic | 31.2% (10/32) | 50.0% (16/32) | 46.9% (15/32) | 56.2% (18/32) | 46.1% | +25.0pp |
| other | 0.0% (0/4) | 0.0% (0/4) | 25.0% (1/4) | 25.0% (1/4) | 12.5% | +25.0pp |

## Soft Match Rate (Match + Partial)

| Figure Type | text_only | caption_only | vision_only | multimodal | Δ |
|-------------|-----------|--------------|-------------|------------|---|
| table | 18.6% (13/70) | 17.1% (12/70) | 25.7% (18/70) | 28.6% (20/70) | +10.0pp |
| plot | 30.9% (21/68) | 44.1% (30/68) | 27.9% (19/68) | 48.5% (33/68) | +17.6pp |
| schematic | 56.2% (18/32) | 62.5% (20/32) | 56.2% (18/32) | 62.5% (20/32) | +6.2pp |
| other | 25.0% (1/4) | 0.0% (0/4) | 50.0% (2/4) | 50.0% (2/4) | +25.0pp |

## Context Benefit for Feedback (Δ = multimodal - text_only)

| Figure Type | Δ Match | Δ Soft Match | Helps? |
|-------------|---------|--------------|--------|
| table | +15.7pp | +10.0pp | Yes |
| plot | +33.8pp | +17.6pp | Yes |
| schematic | +25.0pp | +6.2pp | Yes |
| other | +25.0pp | +25.0pp | Yes |

## H4 Hypothesis Test (Feedback)

**H4:** Tables easier than Schematics?

- Table avg (Match %): 12.1%
- Schematic avg (Match %): 46.1%
- **Result: FAIL**

## Summary

Visual context **helps** feedback quality for some figure types.