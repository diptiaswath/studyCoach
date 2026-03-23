# SPIQA+ Dataset Statistics

Figure types in the source files are normalized to one of: `table`, `plot`, `schematic`.
Normalization uses `content_type` when `figure_type` contains `'N/A'`, otherwise uses `figure_type`.
Residual non-schematic/non-plot/non-table entries are classified as `schematic`.

---

## SPIQA_plus_train_1500

| Metric | Value |
|--------|-------|
| Papers | 1,500 |
| QA examples | 11,985 |
| Avg QA / paper | 8.0 |
| Min QA / paper | 1 |
| Max QA / paper | 49 |

### Verdict Breakdown

| Verdict | Count | % |
|---------|-------|---|
| correct | 1,712 | 14.3% |
| incorrect | 5,138 | 42.9% |
| partially correct | 5,135 | 42.8% |

### Error Category Breakdown

| Error Category | Count | % |
|----------------|-------|---|
| N/A (correct) | 1,712 | 14.3% |
| factual | 3,425 | 28.6% |
| omission | 3,425 | 28.6% |
| conceptual | 3,423 | 28.6% |

### Figure Type Breakdown

| Figure Type | Count | % |
|-------------|-------|---|
| table | 5,092 | 42.5% |
| plot | 4,491 | 37.5% |
| schematic | 2,402 | 20.0% |

---

## SPIQA_plus_val_200

| Metric | Value |
|--------|-------|
| Papers | 200 |
| QA examples | 2,085 |
| Avg QA / paper | 10.4 |
| Min QA / paper | 1 |
| Max QA / paper | 32 |

### Verdict Breakdown

| Verdict | Count | % |
|---------|-------|---|
| correct | 297 | 14.2% |
| incorrect | 894 | 42.9% |
| partially correct | 894 | 42.9% |

### Error Category Breakdown

| Error Category | Count | % |
|----------------|-------|---|
| N/A (correct) | 297 | 14.2% |
| factual | 596 | 28.6% |
| omission | 596 | 28.6% |
| conceptual | 596 | 28.6% |

### Figure Type Breakdown

| Figure Type | Count | % |
|-------------|-------|---|
| table | 923 | 44.3% |
| plot | 758 | 36.4% |
| schematic | 404 | 19.4% |

---

## SPIQA_plus_testA_118

| Metric | Value |
|--------|-------|
| Papers | 118 |
| QA examples | 666 |
| Avg QA / paper | 5.6 |
| Min QA / paper | 1 |
| Max QA / paper | 21 |

### Verdict Breakdown

| Verdict | Count | % |
|---------|-------|---|
| correct | 95 | 14.3% |
| incorrect | 286 | 42.9% |
| partially correct | 285 | 42.8% |

### Error Category Breakdown

| Error Category | Count | % |
|----------------|-------|---|
| N/A (correct) | 95 | 14.3% |
| factual | 191 | 28.7% |
| omission | 190 | 28.5% |
| conceptual | 190 | 28.5% |

### Figure Type Breakdown

| Figure Type | Count | % |
|-------------|-------|---|
| table | 278 | 41.7% |
| plot | 252 | 37.8% |
| schematic | 136 | 20.4% |
