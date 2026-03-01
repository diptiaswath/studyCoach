# Figure Type Analysis: H4 Hypothesis Test

## Hypothesis

**H4:** Tables should be easiest (explicit, localized information),
while architecture diagrams/schematics should be hardest (require
understanding spatial relationships between components).

## Accuracy Matrix

| Figure Type | text_only | caption_only | vision_only | multimodal | Avg |
|-------------|-----------|--------------|-------------|------------|-----|
| plot | 48.5% (33/68) | 45.6% (31/68) | 39.7% (27/68) | 47.1% (32/68) | 45.2% | |
| other | 25.0% (1/4) | 100.0% (4/4) | 25.0% (1/4) | 50.0% (2/4) | 50.0% | |

## Context Benefit (Δ = multimodal - text_only)

| Figure Type | Δ (pp) | Interpretation |
|-------------|--------|----------------|
| plot | -1.5 | Visual context **hurts** |
| other | +25.0 | Visual context **helps** |

## H4 Hypothesis Test

**H4:** Tables > Schematics?

- Table avg: N/A
- Schematic avg: N/A
- **Result: FAIL**

---
