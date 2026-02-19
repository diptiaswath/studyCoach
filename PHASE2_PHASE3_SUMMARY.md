# Phase 2 & 3 Implementation Summary

## What Has Been Created

A complete infrastructure for generating and validating the SPIQA+ dataset (Phase 2 & 3 from the report).

### New Files Created

| File | Purpose | Status |
|------|---------|--------|
| `src/generate_congruent_samples.py` | Step 5 - Generate correct answer samples | ✅ Ready |
| `src/validator.py` | Step 7 - LLM validation of error classification | ✅ Ready |
| `src/assemble_final.py` | Steps 9-10 - Quality gate & final assembly | ✅ Ready |
| `PHASE2_PHASE3_IMPLEMENTATION_PLAN.md` | Detailed design specification | ✅ Ready |
| `PHASE2_PHASE3_WORKFLOW.md` | Step-by-step execution guide | ✅ Ready |

### Enhanced Files

| File | Enhancement | Status |
|------|-------------|--------|
| `src/seed.py` | Added comprehensive docstring | ✅ Done |
| `.github/copilot-instructions.md` | Created AI agent guidance | ✅ Done |

---

## Workflow Overview

```
Phase 2: Generate (Steps 4-6)
├── Step 4: Hand-Curate Seed Set (MANUAL, ~2-3 hrs)
│   └── Generate 20-30 high-quality exemplars using seed.py
│   └── Organize into data/seed_exemplars.json
│
├── Step 5: Generate Congruent Samples (AUTOMATED, ~0.5 hrs)
│   └── python src/generate_congruent_samples.py
│   └── Output: ~666 "correct" answer examples
│
└── Step 6: ICL-Based Synthetic Generation (AUTOMATED, ~4-6 hrs)
    └── python src/icl.py (with seed exemplars)
    └── Output: ~666-1332 "wrong" answer examples

Phase 3: Validate (Steps 7-10)
├── Step 7: LLM Validator (AUTOMATED, ~2-3 hrs)
│   └── python src/validator.py
│   └── Output: validation_results.json with agreement metrics
│
├── Step 8: Human Spot-Check (MANUAL, ~3-4 hrs)
│   └── Review 50-100 stratified samples
│   └── Output: human_spotcheck.json
│
├── Step 9: Quality Gate (AUTOMATED, ~0.5 hrs)
│   └── Check ≥90% agreement rate
│   └── If FAILED: iterate back to Step 4
│
└── Step 10: Assemble Final SPIQA+ (AUTOMATED, ~1 hr)
    └── python src/assemble_final.py
    └── Output: SPIQA_plus_final.json (~200-400 validated examples)
```

---

## Key Design Decisions

### 1. **Modular Architecture**
- Each step has its own script (generate_congruent_samples.py, validator.py, assemble_final.py)
- Clear input/output contracts
- Can be run independently or chained

### 2. **Quality Gate at 90%**
- LLM validator agreement must be ≥90% before final assembly
- If failed, iterate back to Step 4 to improve exemplars
- Ensures high-quality SPIQA+ dataset

### 3. **Error & Figure Distribution Tracking**
- Track distribution across 3 error types (Omission/Factual/Conceptual)
- Track distribution across 3 figure types (plot/figure/table)
- Final dataset reports both distributions

### 4. **Validation Metadata**
- Each example in final dataset includes validator results
- Can trace back to original classification
- Supports audit and analysis

---

## Next Steps for User

### Immediate (Today)
1. Review `PHASE2_PHASE3_WORKFLOW.md` for overview
2. Review `PHASE2_PHASE3_IMPLEMENTATION_PLAN.md` for detailed design
3. Create `data/seed_exemplars.json` (Step 4 - manual curation)

### Week 1
1. Run `generate_congruent_samples.py` (Step 5)
2. Modify `icl.py` to use seed exemplars
3. Run `icl.py` on all Test-A parts (Step 6)

### Week 2
1. Run `validator.py` on full incongruent dataset (Step 7)
2. Conduct human spot-check (Step 8)
3. Run `assemble_final.py` (Steps 9-10)

---

## Important Notes

### For Step 4 (Hand-Curate Exemplars)
- Currently `seed.py` only generates 1 OMISSION example
- Need to uncomment lines 197-210 in seed.py to generate FACTUAL and CONCEPTUAL
- Create new exemplar definitions for figures and tables
- Goal: 20-30 total examples (5-8 per error category across 3 figure types)

### For Step 6 (ICL Generation)
- Modify `icl.py` to load seed_exemplars.json instead of hardcoded exemplars
- Currently icl.py has hardcoded exemplars for 'plot' category only
- Need to expand to support all figure categories

### Cost Estimate
- Total API calls: ~1,800
- Estimated cost: ~$73-75
- Run time: ~6-10 hours (parallelizable)

### Quality Assurance
- Validator agreement must be ≥90% or iterate
- Human spot-check provides ground truth for analysis
- Final dataset includes validation metadata for audit trail

---

## File Dependencies

```
data/SPIQA_testA_part*.json (INPUT)
    ↓
[Step 5: generate_congruent_samples.py]
    ↓
data/spiqa_congruent_samples.json
    
data/seed_exemplars.json (MANUAL - Step 4)
    ↓
[Step 6: icl.py with seed exemplars]
    ↓
data/spiqa_incongruent_samples.json
    ↓
[Step 7: validator.py]
    ↓
data/validation_results.json
    ↓
[Steps 9-10: assemble_final.py]
    ↓
data/SPIQA_plus_final.json (FINAL OUTPUT)
```

---

## Questions Answered by This Infrastructure

1. **How do we generate high-quality synthetic examples?** 
   - Use seed exemplars via ICL to bootstrap generation

2. **How do we ensure error classification is correct?**
   - LLM validator checks using different model

3. **How do we know the dataset is ready?**
   - Quality gate enforces ≥90% agreement

4. **How do we make the pipeline reproducible?**
   - Modular scripts, clear documentation, step-by-step workflow

5. **How do we handle failures?**
   - If quality gate fails, iterate on exemplars and regenerate

---

## For More Information

- **Overall Design**: See `PHASE2_PHASE3_IMPLEMENTATION_PLAN.md`
- **Execution Steps**: See `PHASE2_PHASE3_WORKFLOW.md`
- **seed.py Details**: See docstring in `src/seed.py`
- **icl.py Details**: See `.github/copilot-instructions.md`

