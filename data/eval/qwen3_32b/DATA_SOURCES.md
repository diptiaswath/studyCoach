# Qwen3-VL-32B Evaluation Data Sources

Reference document for tracing Report 2 results back to source files.

## H1/H2: Modality Comparison (N=50 per condition)

**Source:** `llm_as_judge/llm_as_judge_findings_qwen3_32b.md` (on `llm-as-judge` branch)

```bash
git show origin/llm-as-judge:llm_as_judge/llm_as_judge_findings_qwen3_32b.md
```

| Metric | Source Section |
|--------|----------------|
| Verdict Accuracy | Phase 1: Verdict Accuracy |
| LLM Judge Match % | Phase 2c: LLM-as-Judge |
| LLM Judge Soft Match % | Phase 2c: LLM-as-Judge |
| Automated F1/ROUGE-L/BLEU | Phase 2a: Automated Metrics |
| Human Evaluation | Phase 2b: Human Evaluation |

**Local automated metrics:** `data/eval/qwen3_32b/h1h2/eval_scored/*.jsonl`

---

## H3: Error Type Analysis (N=108: 52 factual, 41 conceptual, 15 omission)

**Feedback (LLM-as-Judge):** `data/eval/qwen3_32b/feedback_judgments.json`

Structure:
```json
{
  "factual": { "text_only": { "match": N, "partial": N, "unmatched": N, "match_pct": X, "soft_match_pct": Y }, ... },
  "conceptual": { ... },
  "omission": { ... }
}
```

**Verdict + Raw Results:** `data/eval/qwen3_32b/error_type_analysis/error_type_results.json`

---

## H4: Figure Type Analysis (N=174: 70 table, 68 plot, 32 schematic, 4 other)

**Feedback (LLM-as-Judge):** `data/eval/qwen3_32b/figure_type_analysis/feedback_judgments.json`

Structure:
```json
{
  "table": { "text_only": { "match": N, ... }, ... },
  "plot": { ... },
  "schematic": { ... },
  "other": { ... }
}
```

**Verdict + Raw Results:** `data/eval/qwen3_32b/figure_type_analysis/figure_type_results.json`

---

## Verification Summary (Report 2 vs Source)

All values verified on 2025-03-15:

### H1/H2
| Condition | Verdict | Match % | Soft Match % |
|-----------|---------|---------|--------------|
| C1 (text_only) | 56% | 2% | 42% |
| C2 (caption_only) | 58% | 8% | 48% |
| C3 (vision_only) | 56% | 6% | 48% |
| C4 (multimodal) | 54% | 20% | 58% |

### H3 Feedback (Avg across C1-C4, Delta = C4-C1)
| Error Type | Avg | Delta |
|------------|-----|-------|
| Factual | 26.0% | +21.2pp |
| Conceptual | 42.1% | +29.3pp |
| Omission | 13.4% | +26.7pp |

### H4 Feedback (Avg across C1-C4, Delta = C4-C1)
| Figure Type | Avg | Delta |
|-------------|-----|-------|
| Table | 12.1% | +15.7pp |
| Plot | 26.5% | +33.8pp |
| Schematic | 46.1% | +25.0pp |

---

## File Structure

```
data/eval/qwen3_32b/
├── DATA_SOURCES.md              # This file
├── feedback_judgments.json      # H3 LLM-as-Judge results
├── error_type_analysis/
│   └── error_type_results.json  # H3 raw results + verdicts
├── figure_type_analysis/
│   ├── feedback_judgments.json  # H4 LLM-as-Judge results
│   └── figure_type_results.json # H4 raw results + verdicts
└── h1h2/
    ├── eval_scored/             # Automated metrics (F1, ROUGE-L, BLEU)
    ├── eval_summary/            # Aggregated automated metrics
    ├── h1_results.txt           # Verdict accuracy summary
    └── *_results.json           # Raw model outputs
```
