# Qwen3.5-9B Evaluation: Modification 2 (Architectural Ablation)

**Model:** `Qwen/Qwen3.5-9B`

## Hypothesis

If the bolt-on ViT architecture of Qwen3-VL causes the PC Recall collapse
observed under visual input, Qwen3.5-9B (early fusion) should show a smaller
or reversed verdict gap (C4 accuracy >= C1 accuracy).

## Intrinsic Metrics

| Condition | PC Recall ↑ | FSR ↓ | VGS ↑ | Verdict ↑ |
|-----------|-------------|-------|-------|-----------|
| C1 (text_only) | 32.8% | 38.9% | 30.3% | 38.0% |
| C4 (multimodal) | 31.1% | 24.1% | 42.7% | 44.4% |

## Comparison with Qwen3-VL-8B (Baseline)

| Model | Condition | PC Recall | FSR | VGS | Verdict |
|-------|-----------|-----------|-----|-----|---------|
| Qwen3-VL-8B | C1 | 44.3% | 13.9% | 35.5% | 56.0% |
| Qwen3-VL-8B | C4 | 19.7% | 28.7% | 49.4% | 48.0% |
| **Qwen3.5-9B** | **C1** | **32.8%** | **38.9%** | **30.3%** | **38.0%** |
| **Qwen3.5-9B** | **C4** | **31.1%** | **24.1%** | **42.7%** | **44.4%** |

### Key Differences (C1 → C4)

| Metric | Qwen3-VL-8B | Qwen3.5-9B | Interpretation |
|--------|-------------|------------|----------------|
| Verdict Gap | -8.0pp | **+6.4pp** | Early fusion: visual helps! |
| PC Recall Δ | -24.6pp | **-1.7pp** | Much more stable |
| FSR Δ | +14.8pp | **-14.8pp** | Visual reduces suppression |
| VGS Δ | +13.9pp | +12.4pp | Similar grounding gains |

## Accuracy Matrix (Error Type x Condition)

| Error Type | text_only | multimodal |
|------------|------------|------------|
| factual | 34.6% (18/52) | 40.4% (21/52) |
| conceptual | 48.8% (20/41) | 51.2% (21/41) |
| omission | 20.0% (3/15) | 40.0% (6/15) |

## Context Benefit (delta = multimodal - text_only)

| Error Type | delta (pp) |
|------------|-----------|
| factual | +5.8 |
| conceptual | +2.4 |
| omission | +20.0 |

## Interpretation

### Verdict Gap Analysis

| Comparison | Value |
|------------|-------|
| C1 (text_only) verdict | 38.0% |
| C4 (multimodal) verdict | 44.4% |
| Gap (C4 - C1) | **+6.4pp** |

**Result: Verdict gap REVERSED.** Visual input now **helps** classification (opposite of Qwen3-VL-8B).

### Key Findings

1. **Verdict-feedback dissociation is resolved**: The early fusion architecture shows visual input improving both verdict accuracy (+6.4pp) and VGS (+12.4pp) simultaneously.

2. **PC Recall is stable**: Only -1.7pp drop vs -24.6pp for Qwen3-VL-8B. The model maintains ability to detect partially correct answers with visual input.

3. **FSR improves with visual context**: Model suppresses feedback less when given figures (-14.8pp), the opposite behavior of Qwen3-VL-8B (+14.8pp).

4. **Lower absolute accuracy**: Despite resolving the gap, overall verdict accuracy (38-44%) is lower than Qwen3-VL-8B (48-56%).

## Extrinsic Metrics: Feedback Quality (LLM-as-Judge)

### Overall Match Rate

| Condition | Match% ↑ | SoftM% ↑ |
|-----------|----------|----------|
| C1 (text_only) | 15.7% | 38.9% |
| C4 (multimodal) | **41.7%** | **63.0%** |
| Δ (C4 - C1) | **+26.0pp** | **+24.1pp** |

### Match Rate by Error Type

| Error Type | C1 Match% | C4 Match% | Δ |
|------------|-----------|-----------|-----|
| Factual | 7.7% | 32.7% | +25.0pp |
| Conceptual | 31.7% | 58.5% | +26.8pp |
| Omission | 0.0% | 26.7% | +26.7pp |

### Comparison with Qwen3-VL-8B Feedback Quality

| Model | C1 Match% | C4 Match% | Δ |
|-------|-----------|-----------|-----|
| Qwen3-VL-8B | 2.0% | 6.0% | +4.0pp |
| **Qwen3.5-9B** | **15.7%** | **41.7%** | **+26.0pp** |

**Finding**: Qwen3.5-9B shows **6.5x higher feedback improvement** from visual context (+26.0pp vs +4.0pp), and achieves **7x higher absolute Match%** at C4 (41.7% vs 6.0%).

## Complete Metrics Summary for Table 1

| Metric | C1 (text_only) | C4 (multimodal) | Δ | Comparison to 8B |
|--------|----------------|-----------------|---|------------------|
| PC Recall ↑ | 32.8% | 31.1% | -1.7pp | Much smaller drop |
| FSR ↓ | 38.9% | 24.1% | -14.8pp | Opposite direction! |
| VGS ↑ | 30.3% | 42.7% | +12.4pp | Similar |
| Verdict ↑ | 38.0% | 44.4% | +6.4pp | **Gap reversed!** |
| Match% ↑ | 15.7% | 41.7% | +26.0pp | Much larger gain |
| SoftM% ↑ | 38.9% | 63.0% | +24.1pp | Much larger gain |

### Implication for Report 4

Per the decision matrix in Report 3:

> If C4 verdict accuracy >= C1 for Qwen3.5-9B: early fusion resolves the architectural mismatch.

**Result: CONFIRMED** - Early fusion architecture resolves the verdict-feedback dissociation:
- Verdict gap reversed: +6.4pp (vs -8.0pp for Qwen3-VL-8B)
- Feedback quality dramatically improved: +26.0pp Match% gain
- PC Recall stable: only -1.7pp (vs -24.6pp for Qwen3-VL-8B)

**Recommendation**:
1. **Primary**: Fine-tune Qwen3.5-9B on SPIQA+ - the architecture is correct, just needs domain adaptation
2. **Alternative**: Test Modification 3 (Qwen3.5-397B) to see if scale + early fusion achieves even better absolute performance