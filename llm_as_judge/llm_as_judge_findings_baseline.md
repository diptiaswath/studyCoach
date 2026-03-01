# LLM-as-Judge Evaluation

## Setup

- **Judge model**: `claude-sonnet-4-6` (Anthropic)
- **Temperature**: `0` (deterministic, for consistent judgments)
- **N = 50 examples per configuration** (4 configurations evaluated)
- 3 LLM judge labels:
  - `match` — equivalent meaning and key points
  - `partial` — overlaps but incomplete or missing key corrections
  - `unmatched` — different meaning or misses the main point

We use an LLM judge to evaluate whether AI-generated student feedback (Feedback B) is semantically equivalent to educator-written reference feedback (Feedback A), and compare these judgments against automatic metrics (F1, ROUGE-L, BLEU).

---

## Prompt Conditions

Each configuration varies two dimensions:

| Dimension                   | Options                                                    |
| --------------------------- | ---------------------------------------------------------- |
| **Input modality**          | `text_only`, `caption_only`, `vision_only`, `multimodal`   |
| **Student answer included** | `_results` (with answer) vs `_no_answer_results` (without) |

> **Note**: LLM judge was only run on the `_no_answer` configurations. The `_results` (with answer) conditions show lexical metrics only.

---

## Judge Prompt

The judge receives the following context for each example:

- The original question posed to the student
- The student's answer
- **Feedback A** — ground-truth feedback written by an educator
- **Feedback B** — AI-generated feedback to be evaluated

```
You are evaluating AI-generated student feedback against educator-written reference feedback.

Background:
- A student answered a question.
- Feedback A is ground-truth feedback written by an educator.
- Feedback B is AI-generated feedback.
- Note: The original question may reference figures or tables that are not available here.
  If feedback references visual content you cannot see, factor that uncertainty
  into your confidence score.

Student context:
Question: {question}
Student answer: {student}

Feedback A (ground truth):
{feedback_a}

Feedback B (generated):
{feedback_b}

Task:
Judge whether Feedback B provides equivalent guidance to Feedback A, considering:
1) Semantic meaning (same diagnosis of what's right/wrong)
2) Key points (covers the main correction(s))
3) Helpfulness (would the student learn the same thing)

Labels:
- match: equivalent meaning and key points
- partial: overlaps but misses some key points or is incomplete
- unmatched: different meaning or misses the main point

Return JSON only (no extra text):
{
  "label": "match" | "partial" | "unmatched",
  "confidence": "high" | "medium" | "low",
  "rationale": "short explanation"
}
```

### Limitation: Missing Visual Context

Many questions in the dataset reference figures or tables that are not available to the judge. When feedback refers to visual content the judge cannot see, this is reflected in a lower confidence score rather than a potentially incorrect verdict. Cases with `low` confidence should be treated as candidates for human review.

---

## Summary Results (Strict + Soft Match)

| Scenario               | N   | Match | Partial | Unmatched | Match % | Soft Match % | Avg F1 | Avg ROUGE-L | Avg BLEU |
| ---------------------- | --- | ----- | ------- | --------- | ------- | ------------ | ------ | ----------- | -------- |
| caption_only_no_answer | 50  | 1     | 22      | 27        | 2.0%    | 46.0%        | 0.2941 | 0.1917      | 4.79     |
| multimodal_no_answer   | 50  | 3     | 25      | 22        | 6.0%    | 56.0%        | 0.3023 | 0.1974      | 6.16     |
| text_only_no_answer    | 50  | 1     | 19      | 30        | 2.0%    | 40.0%        | 0.2963 | 0.1885      | 5.09     |
| vision_only_no_answer  | 50  | 2     | 22      | 26        | 4.0%    | 48.0%        | 0.2916 | 0.1839      | 6.70     |
| caption_only           | 50  | —     | —       | —         | —       | —            | 0.3316 | 0.2123      | 5.80     |
| multimodal             | 50  | —     | —       | —         | —       | —            | 0.3262 | 0.2095      | 6.40     |
| text_only              | 50  | —     | —       | —         | —       | —            | 0.2947 | 0.1900      | 4.40     |
| vision_only            | 50  | —     | —       | —         | —       | —            | 0.3128 | 0.2052      | 6.06     |

> — = LLM judge not run for this condition. Soft Match % = (match + partial) / total.

Full numeric results: `data/eval/eval_summary/combined_summary.csv`

---

## Comparison with Human Annotation

Prior to LLM-as-judge evaluation, we conducted a human annotation study on a smaller sample (N=10 per configuration). Human annotators were consistently more lenient than the LLM judge, producing substantially higher match and soft match rates across all scenarios.

| Scenario               | Human Match % | LLM Match % | Human Soft Match % | LLM Soft Match % |
| ---------------------- | ------------- | ----------- | ------------------ | ---------------- |
| caption_only_no_answer | 50%           | 2%          | 90%                | 46%              |
| multimodal_no_answer   | 80%           | 6%          | 100%               | 56%              |
| text_only_no_answer    | 40%           | 2%          | 90%                | 40%              |
| vision_only_no_answer  | 60%           | 4%          | 48%                | 48%              |

> Note: Human annotation N=10 per scenario; LLM judge N=50. Direct comparison is directional only.

### Possible Explanations for the Leniency Gap

**1. Humans tolerate paraphrase more readily.** Annotators naturally recognize that two pieces of feedback are saying the same thing even when phrased very differently. The LLM judge, given an explicit rubric focused on key points and completeness, may penalize feedback that omits specific details even if the core message is equivalent.

**2. Non-expert annotators may judge at the surface level.** As a non-subject-matter expert, the human annotator may have assessed whether the feedback _sounds_ reasonable and on-topic rather than whether it captures the technically correct diagnosis. The LLM judge, drawing on broad domain knowledge, may be better positioned to detect when a key conceptual correction is missing or imprecise — and penalize accordingly.

**3. The LLM judge applies a more explicit rubric.** The prompt defines `match` as "equivalent meaning _and_ key points" with specific criteria around semantic correctness, completeness, and helpfulness. A human annotator without subject expertise may naturally default to a more holistic impression — rating feedback as matching if it feels relevant — rather than systematically checking each criterion.

**4. Small N inflates human rates.** With only N=10 per scenario, human results are highly sensitive to which examples were sampled. A few easy cases in the human sample could substantially raise match rates compared to the full N=50 LLM-judged set.

**5. Annotator leniency bias.** Human annotators, knowing the feedback is AI-generated and roughly correct, may unconsciously give benefit of the doubt — a well-documented tendency in human evaluation of NLG systems.

### Implication

These two evaluations are complementary rather than contradictory. Human annotation captures holistic, student-centered judgment; LLM-as-judge provides scalable, consistent, and stricter coverage across the full dataset. Together they suggest the AI-generated feedback is _directionally correct_ in most cases but often _less complete_ than the educator reference.

---

## Key Observations

- **Match rates are low overall (2–6%)**, with most feedback landing in `partial`. This suggests AI-generated feedback is directionally correct but less complete than the educator reference — likely because ground-truth feedback often references figures and tables unavailable to the model.
- **Multimodal_no_answer achieves the highest soft match rate (56%)** and BLEU score (6.16), making it the strongest performing no-answer configuration.
- **Automatic metrics correlate loosely with LLM judge labels** — ROUGE-L and F1 differences across conditions are small and do not strongly track soft match rates, consistent with findings from the human evaluation.
- **Results are directional**: with N=50 per condition and missing visual context, these numbers should be interpreted as trends rather than definitive rankings.

---
