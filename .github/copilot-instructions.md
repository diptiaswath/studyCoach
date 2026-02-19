# StudyCoach: AI Coding Agent Instructions

## Project Overview
**StudyCoach** is a CMU MultiModal ML course project that generates synthetic student responses for the SPIQA (Scientific Paper Image Question Answering) dataset. The system uses OpenAI's multimodal models with in-context learning (ICL) to simulate student errors on questions about figures, plots, and tables extracted from academic papers.

## Architecture & Data Flow

### Core Components

**[icl.py](src/icl.py)** - Main inference engine
- Reads SPIQA JSON files (nested structure: paper → all_figures → qa)
- Extracts questions and image references from academic papers
- Builds ICL prompts with system instructions and exemplars
- Calls OpenAI API with multimodal input (text + base64 images)
- Parses structured output (student answer, verdict, error_category, feedback)
- Updates JSON in-place with inference results

**[seed.py](src/seed.py)** - Seed example generator
- Generates initial exemplar examples for ICL bootstrapping
- Single image + caption → LLM generates wrong student answer
- Produces reference examples for different error types

**Data Format** ([data/SPIQA_testA_part*.json](data/SPIQA_testA_part1.json))
```json
{
  "paper_id": {
    "all_figures": {
      "image_file_name": {
        "caption": "...",
        "content_type": "figure|plot|table",
        "figure_type": "schematic|plot|N/A"
      }
    },
    "qa": [
      {
        "question": "...",
        "answer": "correct answer",
        "reference": "image_file_name",
        // Added by inference:
        "student": "generated wrong answer",
        "verdict": "Incorrect|Partially Correct|Correct",
        "error_category": "Factual|Omission|Conceptual|N/A",
        "feedback": "study coach explanation"
      }
    ]
  }
}
```

### Error Classification System
Three core error types guide both prompt engineering and output parsing:

- **Omission**: Missing key details (e.g., "The methods are similar" vs full metric comparison)
- **Factual**: Misreading visual elements (wrong axis interpretation, legend confusion, incorrect values)
- **Conceptual**: Wrong conclusion from correct data (e.g., confusing lower loss with better performance)

Error balancing is tracked globally (FACTUAL, OMISSION, CONCEPTUAL counters) to maintain distribution across exemplars.

## Key Patterns & Conventions

### Multimodal Prompt Structure
1. **System instructions**: Templated with dynamic error counts using `string.Template` substitution
2. **Exemplars**: 3 exemplar (user→assistant) pairs per figure_category (plot, table, figure), each with base64-encoded image
3. **User query**: Caption + Question + Current image as base64

Image encoding: [to_data_url()](src/icl.py#L180) converts file → base64 with `data:image/jpeg;base64,` prefix

### Output Parsing
- Case-insensitive regex search for sections: `Student:`, `Agent:`
- Extract: `Verdict = ...`, `Error Category = ...`, `Feedback = ...`
- Fallback: If verdict is "Correct", error_category and feedback auto-set to "N/A"
- See [parse_inference_output()](src/icl.py#L85) for exact regex patterns

### Image Organization
- Base path: `images_root/paper_id/image_file_name`
- Supported types: plot, figure (schematic), table
- Figure categories mapped from `content_type` + `figure_type` fields; defaults to 'figure' if ambiguous

### OpenAI Integration
- **Endpoint**: `client.responses.create()` with `instructions` parameter (not messages-based system role)
- **Model**: Set in DEFAULT_MODEL (update as needed; currently "gpt-5.1")
- **Message structure**: Input messages array with text + image content objects; metadata dict for tracking

## Critical Implementation Details

### Ordering Guarantee
When updating JSON with inference outputs, the system relies on **sorted iteration**:
```python
for paper_key in sorted(data.keys()):  # Alphabetical sort critical!
    for qa_idx, qa in enumerate(qa_list, 1):
```
Outputs must match this exact ordering or parsing will misalign. The `metadata` field in API calls logs paper_id and qa_index for debugging.

### Figure Category Routing
Only process QA if figure_category is in exemplars dict:
```python
if figure_category not in exemplars:
    continue  # Skip if no exemplars for this type
```
This prevents inference calls for unsupported content types (e.g., if only plot exemplars exist).

### Global State Mutation
Error type counters (FACTUAL, OMISSION, CONCEPTUAL) are global and updated by string-matching the output:
```python
if "Factual" in resp.output_text:
    FACTUAL += 1
```
These are **not** used to re-balance within a single run, only logged and used in subsequent runs' prompt templates.

## Workflow & Running Code

### Primary Workflow
```bash
python src/icl.py data/SPIQA_testA_part1.json data/test-A/SPIQA_testA_Images \
  --output data/SPIQA_testA_part1_output.json
```
- Reads JSON, processes all papers/QAs in sorted order
- Calls OpenAI for each (with debug output to console)
- Writes updated JSON to output path
- Prints final error distribution

### Debug Output
Each call prints:
- Full system prompt (with substituted error counts)
- Full user query (caption + question)
- Model output (student answer + agent verdict/feedback)

### Seed Generation
[seed.py](src/seed.py) is a standalone utility to create exemplar examples:
```python
generate_seed_example(image_path, caption, question, answer, 
                      verdict, error_category, ...)
```
Not used in the main pipeline but useful for manually crafting new exemplars.

## Important Gotchas

1. **Image paths must exist**: Code uses `Path(figure_path)` and `to_data_url()` directly; if image file is missing, the code will crash.
2. **JSON ordering matters**: If output list order doesn't match sorted(data.keys()), results will misalign with wrong QAs.
3. **Regex parsing fragility**: If model output format deviates (e.g., "verdict =" with lowercase), parsing fails silently and returns empty strings.
4. **Figure category matching**: Both `content_type` and `figure_type` are checked; if neither maps to ['plot', 'table', 'figure'], defaults to 'figure'.
5. **Base64 encoding mime type**: Currently hardcoded to `image/jpeg`; update if images are PNG/WebP.

## Testing & Validation

- **No automated tests exist** (yet); manual testing via running icl.py with a small data subset
- **Verify output**: Check that JSON has all 4 fields (student, verdict, error_category, feedback) for processed QAs
- **Check error distribution**: Final printout shows Factual/Omission/Conceptual counts
- **Early exit**: If all error types > 0, the main loop breaks early (optimization for fast iteration)
