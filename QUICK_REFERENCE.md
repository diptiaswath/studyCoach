# Quick Reference: Phase 2 & 3 Implementation

## What Was Delivered

**3 Production-Ready Scripts + 3 Comprehensive Guides**

### Scripts

| Script | Purpose | Command | Input | Output |
|--------|---------|---------|-------|--------|
| `src/generate_congruent_samples.py` | Generate correct answers | `python src/generate_congruent_samples.py data/SPIQA_testA_part1.json` | SPIQA JSON | ~666 correct examples |
| `src/validator.py` | Validate error classification | `python src/validator.py data/spiqa_incongruent_samples.json` | Incongruent samples | Agreement metrics |
| `src/assemble_final.py` | Final dataset assembly + quality gate | `python src/assemble_final.py --incongruent ... --validation ...` | Incongruent + Validation | ~200-400 validated examples |

### Documentation

| Document | Content | Audience |
|----------|---------|----------|
| `PHASE2_PHASE3_IMPLEMENTATION_PLAN.md` | Detailed design, architecture, cost estimates | Researchers, designers |
| `PHASE2_PHASE3_WORKFLOW.md` | Step-by-step execution guide with commands | Developers, practitioners |
| `PHASE2_PHASE3_SUMMARY.md` | High-level overview and quick reference | Project managers, reviewers |

---

## One-Command Execution Path

```bash
# Step 5: Generate congruent samples from SPIQA
python src/generate_congruent_samples.py data/SPIQA_testA_part1.json \
  --output data/spiqa_congruent_samples.json

# Step 7: Validate incongruent samples (requires icl.py output from Step 6)
python src/validator.py data/spiqa_incongruent_samples.json \
  --images-root data/test-A/SPIQA_testA_Images \
  --output data/validation_results.json \
  --validator-model gpt-4-turbo

# Step 10: Assemble final SPIQA+ dataset
python src/assemble_final.py \
  --incongruent data/spiqa_incongruent_samples.json \
  --validation data/validation_results.json \
  --output data/SPIQA_plus_final.json
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ SPIQA Test-A (666 QAs across 118 papers)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    [Manual]         [Step 5]         [Step 6]
    Hand-Curate      Generate        Generate
    20-30            Congruent       Incongruent
    Exemplars        (666)           (666-1332)
         │               │               │
         └───────────────┼───────────────┘
                         │
                    [Step 7]
                   LLM Validator
                  (Agreement Check)
                         │
                    [Step 8]
                   Human Spot-Check
                   (50-100 samples)
                         │
                    [Step 9]
                   Quality Gate
                  (≥90% agreement)
                         │
                    [Step 10]
               Assemble Final Dataset
                    (~200-400)
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
    SPIQA+                         With Validation
    Final Dataset                  Metadata
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Input QAs | 666 (from SPIQA Test-A) |
| Expected Output | 200-400 validated examples |
| Quality Threshold | ≥90% LLM agreement |
| Error Categories | 3 (Omission, Factual, Conceptual) |
| Figure Types | 3 (plot, figure, table) |
| Manual Effort | ~8-10 hours (Steps 4, 8) |
| Automated Effort | ~6-9 hours (Steps 5-7, 9-10) |
| Estimated Cost | $73-75 (API calls) |

---

## Decision Points

| Step | Decision | Impact |
|------|----------|--------|
| 4 | Quality of seed exemplars | Determines quality of generated examples |
| 7 | Validator model choice | Affects agreement rate |
| 8 | Spot-check sample size | More samples = higher confidence |
| 9 | Quality gate threshold | 90% is standard; can be adjusted |

---

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| Image not found in validator | Check `--images-root` path and image existence |
| Agreement rate < 90% | Improve seed exemplars (Step 4) and regenerate |
| Memory error with large JSON | Process in batches; use `--max-samples` |
| API rate limit | Add delays between calls or reduce batch size |

---

## Data Flow & File Organization

```
Phase 2 & 3 Outputs:
data/
├── seed_exemplars.json               ← Step 4 (manual)
├── spiqa_congruent_samples.json      ← Step 5 output
├── spiqa_incongruent_samples.json    ← Step 6 output
├── validation_results.json           ← Step 7 output
├── human_spotcheck.json              ← Step 8 (manual)
└── SPIQA_plus_final.json             ← Step 10 output (FINAL)
```

---

## Quality Assurance Checklist

### Before Starting (Step 4)
- [ ] Understand the 3 error types (Omission, Factual, Conceptual)
- [ ] Have access to SPIQA Test-A images
- [ ] OpenAI API key configured

### After Step 7 (Validation)
- [ ] Agreement rate ≥90% achieved
- [ ] No images missing (0% "Image not found" errors)
- [ ] Validator model and settings documented

### After Step 8 (Human Review)
- [ ] 50-100 samples manually reviewed
- [ ] Quality scores recorded
- [ ] Problem areas identified

### Before Final Assembly (Step 10)
- [ ] Quality gate criteria met
- [ ] Error and figure distributions checked
- [ ] Metadata complete and accurate

---

## Performance Expectations

| Task | Time | Status |
|------|------|--------|
| Step 5 (generate_congruent_samples.py) | ~0.5 hr | Fast (no API calls) |
| Step 6 (icl.py, 666 QAs) | ~4-6 hrs | Depends on API rate limits |
| Step 7 (validator.py, 666 samples) | ~2-3 hrs | Depends on API rate limits |
| Step 10 (assemble_final.py) | ~1 min | Very fast (local processing) |

---

## Integration with Report Sections

This implementation directly addresses:

- **Report Section 3.2** (Compute Requirements)
  - Token estimates provided in plan
  - Cost estimates: $73-75

- **Report Appendix** (Dataset Construction Pipeline)
  - Phase 2 (Steps 4-6): Generation
  - Phase 3 (Steps 7-10): Validation

- **Report Section 1.1** (Dataset Augmentation)
  - SPIQA (666 QAs) → SPIQA+ (200-400 examples)
  - With synthetic user answers + coaching feedback

---

## Next Developer Actions

1. **Review Documentation**
   - Read PHASE2_PHASE3_WORKFLOW.md (15 min)
   - Understand seed exemplar requirements

2. **Execute Step 4**
   - Enhance seed.py to generate more exemplars
   - Manually curate 20-30 best examples
   - Save to data/seed_exemplars.json

3. **Run Steps 5-7**
   - Execute generate_congruent_samples.py
   - Integrate seed exemplars into icl.py
   - Run validator.py

4. **Conduct Step 8**
   - Spot-check 50-100 stratified samples
   - Document quality issues
   - Record human verdicts

5. **Complete Steps 9-10**
   - Run quality gate check
   - Assemble final dataset
   - Validate output structure

---

## For Questions

Refer to:
- **"How do I..."** → See PHASE2_PHASE3_WORKFLOW.md
- **"Why do we..."** → See PHASE2_PHASE3_IMPLEMENTATION_PLAN.md
- **"What are the..."** → See PHASE2_PHASE3_SUMMARY.md

