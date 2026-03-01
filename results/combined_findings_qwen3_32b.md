# Combined Findings: Qwen3-VL-32B Evaluation

**Model: Qwen3-VL-32B**

## Question

Can a larger vision-language model (32B) grade student answers about scientific figures — and provide useful feedback?

This question has two parts:
1. **Verdict Classification**: Can the model correctly identify if an answer is Correct, Partially Correct, or Incorrect?
2. **Coaching**: Can the model provide feedback (explanation) that would actually help a student learn?

---

## Setup

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-VL-32B-Instruct` (Together.ai) |
| H1/H2 Eval sample | 50 examples per condition |
| H3 Eval sample | 108 error examples (52 factual, 41 conceptual, 15 omission) |
| H4 Eval sample | 174 figure examples (70 table, 68 plot, 32 schematic, 4 other) |

### Conditions Tested

| Condition | Caption | Image |
|-----------|:-------:|:-----:|
| C1 (text_only) | - | - |
| C2 (caption_only) | Yes | - |
| C3 (vision_only) | - | Yes |
| C4 (multimodal) | Yes | Yes |

---

## H1: Verdict Accuracy

*Question: Did the model classify correctly?*

| Condition | Accuracy |
|-----------|----------|
| C1 (text_only) | 56.0% (28/50) |
| C2 (caption_only) | **58.0%** (29/50) |
| C3 (vision_only) | 56.0% (28/50) |
| C4 (multimodal) | 54.0% (27/50) |

**Finding:** Caption-only performs best at 32B scale. Adding visual input slightly hurts accuracy (58% → 54%).

### H1 Results

| Metric | Hypothesis | Result |
|--------|------------|--------|
| Verdict Accuracy | Visual context helps? | **FAIL** (54.0% < 56.0%) |

---

## H2: Feedback Quality

*Question: Does predicted feedback match ground truth feedback?*

### Auto Metrics

| Condition | F1 | ROUGE-L | BLEU |
|-----------|---:|--------:|-----:|
| C1 (text_only) | 0.298 | 0.192 | 5.93 |
| C2 (caption_only) | **0.325** | **0.207** | 7.16 |
| C3 (vision_only) | 0.294 | 0.193 | **8.13** |
| C4 (multimodal) | 0.298 | 0.202 | 7.44 |

**Finding:** Auto metrics show caption_only performs best on F1/ROUGE-L. Vision_only has highest BLEU. However, auto metrics fail to differentiate feedback quality across conditions.

### LLM-as-Judge (N=50 per condition)

Using Claude (claude-sonnet-4-6) as judge to evaluate feedback quality (Match/Partial/Unmatched).

| Condition | Match | Partial | Unmatched | Match % | Soft Match % |
|-----------|-------|---------|-----------|---------|--------------|
| C1 (text_only) | 1 | 20 | 29 | 2.0% | 42.0% |
| C2 (caption_only) | 4 | 20 | 26 | 8.0% | 48.0% |
| C3 (vision_only) | 3 | 21 | 26 | 6.0% | 48.0% |
| C4 (multimodal) | 10 | 19 | 21 | **20.0%** | **58.0%** |

**Finding:** Multimodal leads significantly on strict match (20%) with a clear gap over other conditions. Visual context improves feedback quality.

### H2 Results

| Metric | Hypothesis | Result |
|--------|------------|--------|
| Feedback Quality (LLM Judge) | Visual context helps? | **PASS** (20.0% > 2.0%) |
| Context Benefit on Feedback Quality | Δ = C4 - C1 | **+18.0pp** (Match), **+16.0pp** (Soft Match) |

---

## H3 Hypothesis Test: Verdict Accuracy by Error Type

**H3:** Factual errors detected more reliably than conceptual errors, and visual context helps conceptual errors more than factual errors.

### Accuracy by Error Type × Condition

| Error Type | C1 | C2 | C3 | C4 | Avg |
|------------|-----|-----|-----|-----|-----|
| factual (n=52) | 32.7% | 28.8% | 26.9% | 25.0% | 28.4% |
| conceptual (n=41) | 58.5% | 56.1% | 53.7% | 61.0% | 57.3% |
| omission (n=15) | 66.7% | 53.3% | 33.3% | 40.0% | 48.3% |

### Context Benefit (Δ = C4 - C1)

| Error Type | Δ (pp) | Interpretation |
|------------|--------|----------------|
| factual | -7.7 | Visual context **hurts** |
| conceptual | +2.4 | Visual context **helps** |
| omission | -26.7 | Visual context **hurts** |

**Key finding:** Visual context helps conceptual errors (+2.4pp) but hurts factual (-7.7pp) and omission (-26.7pp) errors.

### H3 Results

| Metric | Hypothesis | Result |
|--------|------------|--------|
| Verdict Accuracy | Factual > Conceptual? | **FAIL** (28.4% < 57.3%) |
| Context Benefit on Verdict Accuracy | Context helps conceptual more? | **PASS** (+2.4pp > -7.7pp) |

**H3 Overall: PARTIAL**

### Interpretation

1. **Verdict Accuracy failed:** Factual errors are NOT detected more reliably — conceptual errors are much easier to classify (57.3% vs 28.4%).

2. **Context Benefit on Verdict Accuracy passed:** Visual context helps conceptual errors (+2.4pp) while hurting factual errors (-7.7pp).

3. **Key difference from 8B:** At 32B scale, visual context actually **helps** conceptual error detection (+2.4pp), whereas at 8B it hurt all error types.

---

## H3 Hypothesis Test: Feedback Quality by Error Type

We evaluated feedback quality using Claude as LLM judge (Match/Partial/Unmatched).

### Match Rate by Error Type × Condition

| Error Type | C1 | C2 | C3 | C4 | Avg | Δ |
|------------|-----|-----|-----|-----|-----|---|
| factual (n=52) | 13.5% | 26.9% | 28.8% | 34.6% | 26.0% | +21.2pp |
| conceptual (n=41) | 26.8% | 48.8% | 36.6% | 56.1% | 42.1% | +29.3pp |
| omission (n=15) | 0.0% | 0.0% | 26.7% | 26.7% | 13.4% | +26.7pp |

### Soft Match Rate (Match + Partial)

| Error Type | C1 | C2 | C3 | C4 | Δ |
|------------|-----|-----|-----|-----|---|
| factual | 30.8% | 38.5% | 42.3% | 50.0% | +19.2pp |
| conceptual | 65.9% | 78.0% | 61.0% | 70.7% | +4.9pp |
| omission | 46.7% | 33.3% | 46.7% | 53.3% | +6.7pp |

### Context Benefit for Feedback Quality (Δ = C4 - C1)

| Error Type | Δ Match | Δ Soft Match | Helps? |
|------------|---------|--------------|--------|
| factual | +21.2pp | +19.2pp | Yes |
| conceptual | +29.3pp | +4.9pp | Yes |
| omission | +26.7pp | +6.7pp | Yes |

**Key finding:** All Δ values are positive — visual context helps feedback quality for ALL error types.

### H3 Results (Feedback Quality)

| Metric | Hypothesis | Result |
|--------|------------|--------|
| Feedback Quality | Factual > Conceptual? | **FAIL** (26.0% < 42.1%) |
| Context Benefit on Feedback Quality | Context helps conceptual more than factual? | **PASS** (+29.3pp > +21.2pp) |

---

## H3 Summary: Opposite Effects for Verdict Accuracy vs Feedback Quality by Error Type

| Metric | Factual Δ | Conceptual Δ | Omission Δ | Visual Context Effect |
|--------|-----------|--------------|------------|----------------------|
| **Verdict Accuracy** | -7.7pp | +2.4pp | -26.7pp | **Mixed** |
| **Feedback Quality** | +21.2pp | +29.3pp | +26.7pp | **Helps all** |

### Key Insight

At 32B scale, visual input has **mixed effects** on verdict accuracy but **consistently helps** feedback quality:
- **Verdict Accuracy:** Visual context helps conceptual errors (+2.4pp) but hurts factual (-7.7pp) and omission (-26.7pp)
- **Feedback Quality:** Visual context helps all error types (all Δ positive)

**Improvement over 8B:** At 32B, visual context now helps conceptual verdict accuracy (+2.4pp), whereas at 8B it hurt all error types.

---

## H4 Hypothesis Test: Verdict Accuracy by Figure Type

We evaluated verdict accuracy broken down by figure type to test H4.

**H4:** Tables should be easiest (explicit, localized information), while schematics should be hardest (require understanding spatial relationships).

### Accuracy by Figure Type × Condition

| Figure Type | C1 | C2 | C3 | C4 | Avg | Δ |
|-------------|-----|-----|-----|-----|-----|---|
| table (n=70) | 51.4% | 51.4% | 48.6% | 52.9% | 51.1% | +1.4pp |
| plot (n=68) | 52.9% | 47.1% | 48.5% | 48.5% | 49.3% | -4.4pp |
| schematic (n=32) | 46.9% | 56.2% | 59.4% | 65.6% | 57.0% | +18.8pp |
| other (n=4) | 50.0% | 75.0% | 50.0% | 50.0% | 56.2% | +0.0pp |

### Context Benefit (Δ = C4 - C1)

| Figure Type | Δ (pp) | Interpretation |
|-------------|--------|----------------|
| table | +1.4 | Visual context **helps slightly** |
| plot | -4.4 | Visual context **hurts** |
| schematic | +18.8 | Visual context **helps significantly** |
| other | +0.0 | Visual context **neutral** |

**Key finding:** Visual context significantly helps schematics (+18.8pp) — a major improvement over 8B where it showed no effect.

### H4 Results

| Metric | Hypothesis | Result |
|--------|------------|--------|
| Verdict Accuracy | Tables > Schematics? | **FAIL** (51.1% < 57.0%) |
| Context Benefit on Verdict Accuracy | Context helps schematics more than tables? | **PASS** (+18.8pp > +1.4pp) |

**H4 Overall: PARTIAL**

### Interpretation

1. **Tables are NOT easiest** (51.1% avg) — schematics are actually easier at 32B scale (57.0%)
2. **Schematics benefit most from visual context** (+18.8pp) — the 32B model can use spatial information effectively
3. **Plots still struggle** (49.3% avg, Δ=-4.4pp) — trend interpretation remains challenging

### Key Difference from 8B

| Figure Type | 8B Verdict Accuracy | 32B Verdict Accuracy | 8B Δ | 32B Δ |
|-------------|------------|-------------|------|-------|
| table | 70.5% | 51.1% | -4.5pp | +1.4pp |
| schematic | 33.3% | 57.0% | +0.0pp | **+18.8pp** |

**Major improvement:** At 32B, schematics improved dramatically (33.3% → 57.0%) and visual context now helps significantly (+18.8pp vs +0.0pp at 8B).

---

## H4 Hypothesis Test: Feedback Quality by Figure Type

We evaluated feedback quality using Claude as LLM judge (Match/Partial/Unmatched).

### Match Rate by Figure Type × Condition

| Figure Type | C1 | C2 | C3 | C4 | Avg | Δ |
|-------------|-----|-----|-----|-----|-----|---|
| table (n=70) | 4.3% | 7.1% | 17.1% | 20.0% | 12.1% | +15.7pp |
| plot (n=68) | 10.3% | 32.4% | 19.1% | 44.1% | 26.5% | +33.8pp |
| schematic (n=32) | 31.2% | 50.0% | 46.9% | 56.2% | 46.1% | +25.0pp |
| other (n=4) | 0.0% | 0.0% | 25.0% | 25.0% | 12.5% | +25.0pp |

### Soft Match Rate (Match + Partial)

| Figure Type | C1 | C2 | C3 | C4 | Δ |
|-------------|-----|-----|-----|-----|---|
| table | 18.6% | 17.1% | 25.7% | 28.6% | +10.0pp |
| plot | 30.9% | 44.1% | 27.9% | 48.5% | +17.6pp |
| schematic | 56.2% | 62.5% | 56.2% | 62.5% | +6.2pp |
| other | 25.0% | 0.0% | 50.0% | 50.0% | +25.0pp |

### Context Benefit for Feedback Quality (Δ = C4 - C1)

| Figure Type | Δ Match | Δ Soft Match | Helps? |
|-------------|---------|--------------|--------|
| table | +15.7pp | +10.0pp | Yes |
| plot | +33.8pp | +17.6pp | Yes |
| schematic | +25.0pp | +6.2pp | Yes |
| other | +25.0pp | +25.0pp | Yes |

### H4 Results (Feedback Quality)

| Metric | Hypothesis | Result |
|--------|------------|--------|
| Feedback Quality | Tables > Schematics? | **FAIL** (12.1% < 46.1%) |
| Context Benefit on Feedback Quality | Context helps schematics more than tables? | **PASS** (+25.0pp > +15.7pp) |

**H4 Feedback Quality: PARTIAL**

### Interpretation

1. **Schematics have best feedback quality** (46.1% Match) — spatial understanding enables better explanations at 32B
2. **Tables have worst feedback quality** (12.1% Match) — structured data feedback is challenging
3. **Plots benefit most from visual context** (+33.8pp) — trend interpretation improves dramatically with images
4. **All figure types benefit from visual context** — all Δ values positive

---

## H4 Summary: Verdict Accuracy vs Feedback Quality by Figure Type

| Figure Type | Verdict Accuracy Avg | Verdict Accuracy Δ | Feedback Quality Avg | Feedback Quality Δ |
|-------------|----------------------|--------------------|----------------------|--------------------|
| table | 51.1% | +1.4pp | 12.1% | +15.7pp |
| plot | 49.3% | -4.4pp | 26.5% | +33.8pp |
| schematic | 57.0% | +18.8pp | 46.1% | +25.0pp |
| other | 56.2% | +0.0pp | 12.5% | +25.0pp |

**Key Insight:** Schematics perform best for both verdict accuracy (57.0%) and feedback quality (46.1%) at 32B scale. This is the opposite of 8B where tables were easiest. Visual context helps all figure types for feedback quality.

---

## Comparison: 8B vs 32B

### H1: Verdict Accuracy

| Condition | 8B | 32B | Change |
|-----------|-----|-----|--------|
| C1 (text_only) | 56% | 56% | 0pp |
| C2 (caption_only) | 48% | **58%** | +10pp |
| C3 (vision_only) | 50% | 56% | +6pp |
| C4 (multimodal) | 48% | 54% | +6pp |

**Finding:** 32B shows improved performance with captions and visual input.

### H2: Feedback Quality (LLM-as-Judge)

| Condition | 8B Match % | 32B Match % | Improved? |
|-----------|------------|-------------|-----------|
| C1 (text_only) | 2% | 2% | Same (0pp) |
| C2 (caption_only) | 2% | **8%** | Yes (+6pp) |
| C3 (vision_only) | 4% | **6%** | Yes (+2pp) |
| C4 (multimodal) | 6% | **20%** | Yes (+14pp) |

| Condition | 8B Soft Match % | 32B Soft Match % | Improved? |
|-----------|-----------------|------------------|-----------|
| C1 (text_only) | 40% | **42%** | Yes (+2pp) |
| C2 (caption_only) | 46% | **48%** | Yes (+2pp) |
| C3 (vision_only) | 48% | 48% | Same (0pp) |
| C4 (multimodal) | 56% | **58%** | Yes (+2pp) |

**Finding:** 32B shows largest improvement on multimodal (+14pp strict match). Scaling helps most when visual+textual context is combined.

### H3: Context Benefit by Error Type

#### Verdict Accuracy (Δ = C4 - C1)

| Error Type | 8B Δ | 32B Δ | Improved? |
|------------|------|-------|-----------|
| factual | -21.2pp | -7.7pp | Yes (+13.5pp) |
| conceptual | -14.6pp | **+2.4pp** | Yes (+17.0pp) |
| omission | -6.7pp | -26.7pp | No (-20.0pp) |

**Finding:** 32B shows major improvement for conceptual errors — visual context now helps instead of hurts.

#### Feedback Quality (Δ = C4 - C1)

| Error Type | 8B Δ | 32B Δ | Improved? |
|------------|------|-------|-----------|
| factual | +21.2pp | +21.2pp | Same (0pp) |
| conceptual | +31.7pp | +29.3pp | No (-2.4pp) |
| omission | +33.3pp | +26.7pp | No (-6.6pp) |

**Finding:** Feedback quality context benefit remains strong at 32B, with slight decreases for conceptual and omission errors.

### H4: Context Benefit by Figure Type

#### Verdict Accuracy (Δ = C4 - C1)

| Figure Type | 8B Δ | 32B Δ | Improved? |
|-------------|------|-------|-----------|
| table | -4.5pp | +1.4pp | Yes (+5.9pp) |
| plot | -10.5pp | -4.4pp | Yes (+6.1pp) |
| schematic | +0.0pp | **+18.8pp** | Yes (+18.8pp) |

**Finding:** 32B shows dramatic improvement for schematics — visual context now helps significantly.

#### Feedback Quality (Δ = C4 - C1)

| Figure Type | 8B Δ | 32B Δ | Improved? |
|-------------|------|-------|-----------|
| table | +4.5pp | **+15.7pp** | Yes (+11.2pp) |
| plot | -5.3pp | **+33.8pp** | Yes (+39.1pp) |
| schematic | +11.1pp | **+25.0pp** | Yes (+13.9pp) |

**Finding:** 32B shows dramatic improvement in feedback quality for all figure types, especially plots (+39.1pp improvement).

---

## Summary

### What Improved at 32B Scale

1. **Multimodal feedback quality:** LLM judge match rate improved dramatically (6% → 20%, +14pp)
2. **Schematics:** Visual context now helps verdict accuracy (+18.8pp vs +0.0pp at 8B)
3. **Conceptual errors:** Visual context now helps verdict accuracy (+2.4pp vs -14.6pp at 8B)
4. **Caption understanding:** Caption-only verdict accuracy improved (48% → 58%)
5. **Plot feedback quality:** Visual context now helps (+33.8pp vs -5.3pp at 8B)

### What Remains Challenging

1. **Factual errors:** Still hard to detect (28.4% avg) and visual context still hurts (-7.7pp)
2. **Plots:** Visual context still hurts verdict accuracy (-4.4pp)
3. **Tables:** Verdict accuracy lower than expected (51.1%)
4. **Text-only feedback:** LLM judge match rate remains low (2%) with no improvement over 8B

### Key Insight

> "At 32B scale, the model can leverage visual context for spatial understanding (schematics +18.8pp), conceptual reasoning (+2.4pp), and feedback quality (multimodal +14pp), but still struggles with precise value extraction (factual -7.7pp)."

---

## Source Files

| File | Description |
|------|-------------|
| `data/eval/qwen3_32b/Qwen3-VL-32B-results.pdf` | Raw evaluation results |
| `data/eval/qwen3_32b/h1h2/` | H1/H2 evaluation outputs |
| `data/eval/qwen3_32b/error_type_analysis/` | H3 results by error type |
| `data/eval/qwen3_32b/figure_type_analysis/` | H4 results by figure type |
| `data/eval/qwen3_32b/feedback_judgments.json` | LLM judge outputs |
| `llm_as_judge/llm_as_judge_findings_qwen3_32b.md` | H2 LLM-as-Judge evaluation (branch: llm-as-judge) |
