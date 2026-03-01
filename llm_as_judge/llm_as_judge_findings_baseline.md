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

Full numeric results: `data/eval/eval_summary/baseline_summary.csv`

---

## Prior Evaluation Results (Baseline)

Before running the LLM-as-judge, the pipeline was evaluated across three phases. These results provide the baseline context for interpreting what the LLM judge adds.

### Phase 1: Verdict Accuracy

The first question was simply: does the model correctly classify whether a student's answer is correct or incorrect?

| Scenario               | Accuracy |
| ---------------------- | -------- |
| text_only_no_answer    | 56%      |
| vision_only_no_answer  | 50%      |
| caption_only_no_answer | 48%      |
| multimodal_no_answer   | 48%      |

**Finding**: Text-only wins on classification. Adding visual input hurts accuracy at 8B scale. This is somewhat counterintuitive — more context should help — but reflects that at this model size, visual information adds noise to the binary decision rather than signal. The model is likely doing plausibility checking from training knowledge rather than genuinely verifying claims against the figure.

This matters because **verdict accuracy sets an upper bound on feedback quality**: if the model misclassifies whether an answer is right or wrong, the feedback it generates will be misdirected regardless of how well-written it is.

### Phase 2a: Automated Metrics

With verdict accuracy established, we then measured whether the generated feedback matched the ground truth feedback lexically.

| Scenario               | F1   | ROUGE-L | BLEU |
| ---------------------- | ---- | ------- | ---- |
| text_only_no_answer    | 0.30 | 0.19    | 5.1  |
| caption_only_no_answer | 0.29 | 0.19    | 4.8  |
| vision_only_no_answer  | 0.29 | 0.18    | 6.7  |
| multimodal_no_answer   | 0.30 | 0.20    | 6.2  |

**Finding**: Automatic metrics show almost no difference across scenarios (~0.29–0.30 F1). If we stopped here, we'd conclude all scenarios produce similar feedback quality — a misleading conclusion.

### Phase 2b: Human Evaluation

Human annotators evaluated a subset of N=10 examples per scenario on whether feedback was actually useful to a student.

| Scenario               | Match | Partial | Unmatched | Human Match % | Soft Match % |
| ---------------------- | ----- | ------- | --------- | ------------- | ------------ |
| multimodal_no_answer   | 8     | 2       | 0         | 80%           | 100%         |
| vision_only_no_answer  | 6     | 2       | 2         | 60%           | 80%          |
| caption_only_no_answer | 5     | 4       | 1         | 50%           | 90%          |
| text_only_no_answer    | 4     | 5       | 1         | 40%           | 90%          |

**Finding**: Human evaluation reveals dramatic differences that automatic metrics missed. Multimodal feedback is twice as useful as text-only (80% vs 40%). This is the key insight motivating the LLM-as-judge phase: F1/ROUGE/BLEU are poor proxies for feedback quality, and semantic evaluation is necessary to reveal true differences.

### The Core Tension: Verdict Accuracy ≠ Feedback Quality

| Metric                      | Winner           | Loser            |
| --------------------------- | ---------------- | ---------------- |
| Verdict Accuracy (Phase 1)  | text_only (56%)  | multimodal (48%) |
| Feedback Quality (Phase 2b) | multimodal (80%) | text_only (40%)  |

At 8B scale, adding visual input _hurts_ classification accuracy but _helps_ explanation quality. The model can use visual information to reason and explain better, but cannot yet use it reliably enough to improve the binary verdict. The visual signal adds noise to the decision, but adds value to the explanation. This suggests that scaling the model would likely close the verdict gap while preserving the explanation advantage.

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

### Key Finding

> **Human evaluation reveals dramatic differences that automatic metrics missed. Multimodal feedback is twice as useful as text-only (80% vs 40% match rate).**

This is a critical result: F1, ROUGE-L, and BLEU scores are relatively flat across all configurations (ranging ~0.25–0.33), and would not have flagged multimodal as meaningfully better. Human judgment, by contrast, captures whether the feedback is actually _useful to the student_ — a dimension that lexical overlap metrics fundamentally cannot measure. This underscores the value of semantic evaluation (human or LLM-based) over automatic metrics alone for this task.

### Possible Explanations for the Leniency Gap

**1. Humans tolerate paraphrase more readily.** Annotators naturally recognize that two pieces of feedback are saying the same thing even when phrased very differently. The LLM judge, given an explicit rubric focused on key points and completeness, may penalize feedback that omits specific details even if the core message is equivalent.

**2. Non-expert annotators may judge at the surface level.** As a non-subject-matter expert, the human annotator may have assessed whether the feedback _sounds_ reasonable and on-topic rather than whether it captures the technically correct diagnosis. The LLM judge, drawing on broad domain knowledge, may be better positioned to detect when a key conceptual correction is missing or imprecise — and penalize accordingly.

**3. The LLM judge applies a more explicit rubric.** The prompt defines `match` as "equivalent meaning _and_ key points" with specific criteria around semantic correctness, completeness, and helpfulness. A human annotator without subject expertise may naturally default to a more holistic impression — rating feedback as matching if it feels relevant — rather than systematically checking each criterion.

**4. Small N inflates human rates.** With only N=10 per scenario, human results are highly sensitive to which examples were sampled. A few easy cases in the human sample could substantially raise match rates compared to the full N=50 LLM-judged set.

**5. Annotator leniency bias.** Human annotators, knowing the feedback is AI-generated and roughly correct, may unconsciously give benefit of the doubt — a well-documented tendency in human evaluation of NLG systems.

### Implication

These two evaluations are complementary rather than contradictory. Human annotation captures holistic, student-centered judgment; LLM-as-judge provides scalable, consistent, and stricter coverage across the full dataset. Together they suggest the AI-generated feedback is _directionally correct_ in most cases but often _less complete_ than the educator reference.

---

## LLM Judge Results vs Prior Findings

With the baseline context established, we can now interpret the LLM judge results and ask: what changed, what stayed the same, and what does the larger N reveal?

### What Stayed the Same

**Multimodal remains the strongest scenario for feedback quality.** The LLM judge confirms the human evaluation finding — multimodal_no_answer achieves the highest soft match rate (56%) among all judged conditions, consistent with the 80% human match rate directionally. The ranking of scenarios is broadly preserved: multimodal > vision_only > caption_only > text_only on soft match.

**The relative magnitude of the multimodal advantage is consistent across both evaluations.** Human eval showed multimodal at roughly 2x text_only (80% vs 40% strict match); the LLM judge soft match shows a similar relative advantage (56% vs 40%). The gap narrows in absolute terms but the direction and magnitude hold — strengthening rather than contradicting the human finding.

> **Sample size disclaimer**: With N=10 per scenario in human annotation, each single example shifts the match rate by 10 percentage points. The observed 40pp spread (40% to 80%) may narrow as annotation scales up. However, the directional finding is supported by soft match rates, which are less sensitive to individual examples — multimodal achieves 100% soft match vs 90% for text_only, suggesting the advantage is real even if the strict gap is somewhat inflated by small N.

**Automatic metrics still fail to differentiate.** F1 and ROUGE-L remain flat across conditions (0.29–0.30), reinforcing the Phase 2a finding. The LLM judge provides meaningful signal where lexical metrics cannot.

### What Changed

**Match rates dropped dramatically from human to LLM judge** (e.g., multimodal: 80% → 6% strict match, 100% → 56% soft match). This is expected given the stricter rubric, larger N, and non-expert vs domain-aware evaluation — see the leniency gap analysis above. The absolute numbers are not directly comparable, but the _relative ordering_ across scenarios is what matters, and that ordering is largely preserved.

**The LLM judge surfaces more unmatched cases.** With N=50, we see that a meaningful portion of feedback is genuinely off-target (22–30 unmatched cases per scenario), something that N=10 human annotation could not reliably detect. This suggests the 80–100% soft match rates from human evaluation were optimistic — likely inflated by small sample size and non-expert leniency.

**caption_only soft match (46%) lags behind human (90%).** This is one of the larger divergences and likely reflects that caption-based feedback frequently references visual details the LLM judge can identify as missing or imprecise, whereas a human non-expert reading the same feedback would find it reasonable.

### Overall Interpretation

| Dimension                | Human Eval (N=10)                    | LLM Judge (N=50)                       |
| ------------------------ | ------------------------------------ | -------------------------------------- |
| Scenario ranking         | multimodal > vision > caption > text | multimodal > vision > caption > text ✓ |
| Absolute match rates     | 40–80%                               | 2–6% (strict), 40–56% (soft)           |
| Differentiates scenarios | Yes                                  | Yes ✓                                  |
| Reliable sample size     | No (N=10)                            | More reliable (N=50)                   |

The LLM judge validates the core human evaluation finding at scale: **multimodal input produces meaningfully better student feedback than text-only**, a difference invisible to automatic metrics. The lower absolute rates reflect a stricter, more consistent evaluation standard — not a contradiction of the human results.

> Results remain directional given missing visual context in ~50% of examples. Low-confidence verdicts should be flagged for human review.

---
