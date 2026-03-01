# Figure Type Analysis: H4 Hypothesis Test

## Hypothesis

**H4:** Tables should be easiest (explicit, localized information),
while architecture diagrams/schematics should be hardest (require
understanding spatial relationships between components).

## Accuracy Matrix

| Figure Type | text_only | caption_only | vision_only | multimodal | Avg |
|-------------|-----------|--------------|-------------|------------|-----|
| table | 51.4% (36/70) | 51.4% (36/70) | 48.6% (34/70) | 52.9% (37/70) | 51.1% | |
| plot | 52.9% (36/68) | 47.1% (32/68) | 48.5% (33/68) | 48.5% (33/68) | 49.3% | |
| schematic | 46.9% (15/32) | 56.2% (18/32) | 59.4% (19/32) | 65.6% (21/32) | 57.0% | |
| other | 50.0% (2/4) | 75.0% (3/4) | 50.0% (2/4) | 50.0% (2/4) | 56.2% | |

## Context Benefit (Δ = multimodal - text_only)

| Figure Type | Δ (pp) | Interpretation |
|-------------|--------|----------------|
| table | +1.4 | Visual context **helps** |
| plot | -4.4 | Visual context **hurts** |
| schematic | +18.8 | Visual context **helps** |
| other | +0.0 | Visual context **neutral** |

## H4 Hypothesis Test

**H4:** Tables > Schematics?

- Table avg: 51.1%
- Schematic avg: 57.0%
- **Result: FAIL**

---
