#!/usr/bin/env python3
"""Generates synthetic student responses for SPIQA dataset questions with balanced error distribution.

Like icl.py but uses a deterministic round-robin target assignment to guarantee
a balanced mix across all 7 (verdict, error_type) slots:
  - incorrect        × factual / omission / conceptual
  - partially correct × factual / omission / conceptual
  - correct          × N/A

For each non-correct QA the target (verdict, error_type) pair is assigned via
round-robin and passed as a hard constraint in the system prompt, removing all
LLM discretion over error category.

Example usage:
python src/datagen/icl_balanced.py data/SPIQA_testA_part1.json data/test-A/SPIQA_testA_Images --output data/out/SPIQA_testA_part1_output.json
"""
from __future__ import annotations

import base64
import itertools
import json
import re
import sys
import time

from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from string import Template
from typing import Dict, List, Any, Tuple

# Round-robin schedule over all 7 (verdict, error_type) slots.
TARGETS = [
    ("incorrect",         "factual"),
    ("incorrect",         "omission"),
    ("incorrect",         "conceptual"),
    ("partially correct", "factual"),
    ("partially correct", "omission"),
    ("partially correct", "conceptual"),
    ("correct",           "N/A"),
]

ANSWER_TYPE_DESCRIPTIONS = {
    "partially correct": "A partially correct answer gets some aspects right but misses or misidentifies others.",
    "incorrect":         "An incorrect answer provides wrong information or a wrong conclusion as the response to the question posed.",
    "correct":           "A correct answer is an answer which provides the accurate and complete information to the question posed.",
}

# $ERROR_TYPE is locked in both the instruction and the output format so the
# model has no discretion over which error category to produce.
DEFAULT_SYSTEM_PROMPT = \
Template('''
You are a study coach who helps students improve their understanding of scientific papers.
Students are required to answer a question based on an image (and its caption) obtained from a scientific paper.
You are given an example question and student answer where the student has made a $ERROR_TYPE error
as described below (out of all possible different errors described below).

omission: An error due to omitting one or more key points in the answer
factual: An error due to student giving factual data which contradicts the information in the figure/chart/table, or misreading the figure/chart/table (e.g., misreading axes, legends, trends, etc.)
conceptual: An error due to student misunderstanding a concept or using figure/chart/table data to come to a wrong conclusion

You are given a question (and the associated image and caption) and your task is to generate
  1. A $ANSWER_TYPE answer a student is likely to give when posed the question
  2. The study coach feedback on why the student answer is wrong.

$ANSWER_TYPE_DESCRIPTION

Follow below guidelines

- You must generate a $ERROR_TYPE error. Do not use any other error type.
- Use a second person instructional tone in study coach feedback. Aim to explain what the student\'s misunderstanding or confusion is.
- Make the feedback constructive but as concise as possible without missing any important points which aids student understanding.
- Do not repeat information.
- When explaining errors due to misreading the plots mention what higher values means for important axes (if applicable to the error explanation).
- Pay special attention to correctly read plot legends.
- Don\'t do any text formatting (e.g., don\'t use bullet points, bold, etc.) in the output.
- Format the output as below.
Student:
<Wrong student answer>

Agent:
Verdict = $ANSWER_TYPE
Error Category = $ERROR_TYPE
Feedback = <Study coach explanation on why student answer is wrong>
''')
DEFAULT_MODEL = "gpt-5.1"


def load_json(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def parse_inference_output(output_text: str) -> Dict[str, str]:
    text = output_text.replace("\r\n", "\n").replace("\r", "\n")

    lower = text.lower()
    student_idx = lower.find("student:")
    agent_idx = lower.find("agent:")

    if student_idx != -1 and agent_idx != -1 and agent_idx > student_idx:
        student = text[student_idx + len("student:"):agent_idx].strip()
        agent_block = text[agent_idx + len("agent:"):].strip()
    else:
        student = ""
        agent_block = text

    def _clean_value(val: str) -> str:
        return val.strip()

    verdict = ""
    error_category = ""
    feedback = ""

    verdict_match   = re.search(r"Verdict\s*=\s*([^\n]+)", agent_block)
    error_match     = re.search(r"Error Category\s*=\s*([^\n]+)", agent_block)
    feedback_match  = re.search(r"Feedback\s*=\s*(.*?)(?:\n[A-Za-z ]+\s*=|\Z)", agent_block, re.DOTALL)

    if verdict_match:
        verdict = _clean_value(verdict_match.group(1))
        verdict = re.sub(r"\\s+", " ", verdict)
    if error_match:
        error_category = _clean_value(error_match.group(1))
        error_category = re.sub(r"\\s+", " ", error_category)
    if feedback_match:
        feedback = feedback_match.group(1).strip()

    if verdict.lower() == "correct":
        feedback = "N/A"
        error_category = "N/A"

    return {
        "student":        student,
        "verdict":        verdict,
        "error_category": error_category if error_category else "N/A",
        "feedback":       feedback if feedback else "N/A",
    }

def to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def build_exemplar_messages(exemplar: Tuple[str, str, str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build ICL message pair from a single (user, assistant, image_path) exemplar."""
    user, assistant, exemplar_image_path = exemplar
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text",  "text":      user},
                {"type": "input_image", "image_url": to_data_url(exemplar_image_path)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": assistant}],
        },
    ]
    return messages, [user, assistant]

client = OpenAI()

_MAX_RETRY_WAIT = 300  # seconds (5 minutes)

def _call_with_backoff(call: dict):
    """Call client.responses.create with exponential backoff. Exits on timeout."""
    delay = 1
    total_waited = 0
    attempt = 0
    while True:
        try:
            return client.responses.create(**call)
        except Exception as e:
            attempt += 1
            if total_waited + delay > _MAX_RETRY_WAIT:
                tqdm.write(f"[Fatal] API call failed after {attempt} attempts ({total_waited}s waited): {e}")
                sys.exit(1)
            tqdm.write(f"[Retry {attempt}] {e} — retrying in {delay}s (total waited: {total_waited}s)")
            time.sleep(delay)
            total_waited += delay
            delay = min(delay * 2, _MAX_RETRY_WAIT - total_waited)

def run_inference(
    json_path: str | Path,
    images_root: str | Path,
    exemplars: dict,
    output: str | Path | None = None,
    checkpoint_every: int = 100,
    num_papers: int | None = None,
    paper_range: tuple[int, int] | None = None,
    ):
    """Run inference with balanced (verdict, error_type) assignment via round-robin."""
    json_path = Path(json_path)
    images_root = Path(images_root)
    output_path = Path(output) if output else None

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f"JSON : {json_path}")
    print(f"Images: {images_root}")
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

    data = load_json(json_path)

    stats = {
        'correct':          0,
        'incorrect':        {'factual': 0, 'omission': 0, 'conceptual': 0},
        'partially correct':{'factual': 0, 'omission': 0, 'conceptual': 0},
        'parse_errors':     0,
    }
    total_qa_count = 0
    processed_papers = {}

    paper_keys = sorted(data.keys())
    if paper_range is not None:
        start, end = paper_range
        paper_keys = paper_keys[start - 1:end]
        index_offset = start - 1
    else:
        index_offset = 0
        if num_papers is not None:
            paper_keys = paper_keys[:num_papers]
    paper_count = len(paper_keys)

    target_cycle = itertools.cycle(TARGETS)

    paper_bar = tqdm(paper_keys, desc="Papers", unit="paper")
    for paper_idx, paper_key in enumerate(paper_bar, 1):
        paper_bar.set_postfix(paper=paper_key)

        paper = data[paper_key]
        all_figures = paper.get("all_figures", {})

        qa_list = paper.get("qa", [])
        qa_counter = 1
        for qa_idx, qa in tqdm(enumerate(qa_list, 1), desc="  QAs", unit="qa", leave=False, total=len(qa_list)):
            total_qa_count += 1
            question = (qa.get("question", "") or "").strip()
            figure   = (qa.get("reference", "") or "").strip()

            # Normalize figure category to one of 'plot', 'figure', 'table'
            figure_details = all_figures.get(figure, None) if figure != "" else None
            figure_path    = images_root / paper_key / figure if figure_details else None

            figure_content_type = figure_details.get("content_type", "")
            figure_type         = figure_details.get("figure_type", "")
            figure_category     = figure_content_type if "N/A" in figure_type else figure_type

            figure_category = figure_category.strip()
            if figure_category not in ['plot', 'table', 'figure']:
                figure_category = 'figure'

            assert figure_category in exemplars, (
                f"Figure category '{figure_category}' not found in exemplars. "
                "Must be one of 'plot', 'table', 'figure'."
            )

            # Deterministic round-robin target assignment
            answer_type, error_type = next(target_cycle)
            answer_type_description = ANSWER_TYPE_DESCRIPTIONS[answer_type]

            if answer_type != "correct":
                exemplar = exemplars[figure_category][answer_type][error_type]
                messages, _ = build_exemplar_messages(exemplar)

                caption    = figure_details.get("caption", "").strip()
                user_input = f'Caption:\n{caption}\nQuestion:\n{question}'
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text",  "text":      user_input},
                            {"type": "input_image", "image_url": to_data_url(figure_path)},
                        ],
                    }
                )

                call = {
                    "model": DEFAULT_MODEL,
                    "instructions": DEFAULT_SYSTEM_PROMPT.safe_substitute(
                        ANSWER_TYPE=answer_type,
                        ANSWER_TYPE_DESCRIPTION=answer_type_description,
                        ERROR_TYPE=error_type,
                    ),
                    "input": messages,
                    "metadata": {
                        "paper":    paper_key,
                        "qa_index": f'{qa_idx}',
                    },
                }

                resp = _call_with_backoff(call)

                qa_counter += 1

                parsed = parse_inference_output(resp.output_text)
                qa["student"]        = parsed["student"]
                qa["verdict"]        = parsed["verdict"]
                qa["error_category"] = parsed["error_category"]
                qa["feedback"]       = parsed["feedback"]

                parsed_verdict = parsed["verdict"]
                parsed_error   = parsed["error_category"]
                if parsed_verdict in stats and isinstance(stats[parsed_verdict], dict) and parsed_error in stats[parsed_verdict]:
                    stats[parsed_verdict][parsed_error] += 1
                else:
                    stats['parse_errors'] += 1
                    tqdm.write(f"[Parse Error] paper={paper_key} qa={qa_idx} verdict='{parsed_verdict}' error_category='{parsed_error}' (target: {answer_type} / {error_type})")

            else:
                qa["student"]        = qa.get("answer", "").strip()
                qa["student"]        = qa["student"] if qa["student"].endswith(".") else qa["student"] + "."
                qa["verdict"]        = "correct"
                qa["error_category"] = "N/A"
                qa["feedback"]       = "Your answer is correct. Great job!"
                stats['correct'] += 1

        processed_papers[paper_key] = paper

        if output_path and paper_idx % checkpoint_every == 0:
            checkpoint_path = output_path.with_stem(f"{output_path.stem}_{paper_idx + index_offset}")
            checkpoint_path.write_text(
                json.dumps(processed_papers, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tqdm.write(f"[Checkpoint] Saved {paper_idx} papers to {checkpoint_path}")
            tqdm.write(f"[Checkpoint] Waiting 10 seconds before continuing...")
            time.sleep(10)

    if output_path and paper_count % checkpoint_every != 0:
        checkpoint_path = output_path.with_stem(f"{output_path.stem}_{paper_count + index_offset}")
        checkpoint_path.write_text(
            json.dumps(processed_papers, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tqdm.write(f"[Checkpoint] Saved {paper_count} papers to {checkpoint_path}")

    # Print stats
    print(f"\nStats:\n")
    print(f"Total Papers Processed:    {paper_count}")
    print(f"Total QA Pairs Processed:  {total_qa_count}")
    print(f"Correct:                   {stats['correct']}")
    print(f"Incorrect:                 {stats['incorrect']}")
    print(f"Partially Correct:         {stats['partially correct']}")
    print(f"Parse Errors:              {stats['parse_errors']}")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate balanced SPIQA student responses.")
    parser.add_argument("json",        help="Path to JSON file")
    parser.add_argument("images_root", help="Path to images root folder")
    parser.add_argument("--output", required=True, help="Path to write updated JSON output")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save a checkpoint every N papers (default: 100)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num-papers", type=int, default=None,
                       help="Number of papers to process (default: all)")
    group.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                       help="1-based inclusive range of paper indexes (e.g. --range 101 200)")
    args = parser.parse_args()

    ############ Plot Exemplars ############

    plot_incorrect_factual_user = \
'''Caption:
Goodput: CoAP vs. HTTP/TCP.

Question:
What is the difference in response time between CoAP and HTTP for a response size of 50 KiB?'''
    plot_incorrect_factual_assistant = \
'''Student:
Difference is around 5 seconds, with HTTP being slower.

Agent:
Verdict = incorrect
Error Category = factual
Feedback = The answer is incorrect for two reasons. Firstly, the difference is approximately 15 seconds. You probably read the chart (b) y-axis scale wrong. It is easier to determine this difference at 50KB using chart (a). Secondly, HTTP has the lower response time of around 25 seconds while CoAP is around 40 seconds, meaning that HTTP is actually faster (less response time is faster!).'''

    plot_incorrect_omission_user = \
'''Caption:
The average number of k-hop neighbors that two items share in the KG w.r.t. whether they have common raters in (a) MovieLens-1M, (b) Book-Crossing, and (c) BingNews datasets. (d) The ratio of the two average numbers with different hops.

Question:
How does the number of common k-hop neighbors change as the hop distance increases for items with and without common raters?'''
    plot_incorrect_omission_assistant = \
'''Student:
As the hop distance increases, the number of common k-hop neighbors changes for both items with and without common raters.

Agent:
Verdict = incorrect
Error Category = omission
Feedback = The answer is incomplete as you did not specify how the number of common k-hop neighbors changes — it increases as hop distance increases for both types of items. Additionally, the items with common raters consistently have more common neighbors than items without common raters across all datasets and hop distances. However, the ratio of common neighbors for both item types at a given hop tends lower when the number of hops increases.'''

    plot_incorrect_conceptual_user = \
'''Caption:
Figure 1: Two error rates as a function of CPU time on UCR Non-Invasive Fetal ECG Thorax1 dataset.

Question:
How does the observed error compare to the underlying true error as CPU time increases?'''
    plot_incorrect_conceptual_assistant = \
'''Student:
The observed error is initially higher than the underlying true error, but it quickly decreases and converges to the true error as CPU time increases.

Agent:
Verdict = incorrect
Error Category = conceptual
Feedback = This is incorrect for two reasons. The observed error starts approximately similar to the underlying true error, not higher (see the first two data points). Also, observed error does not converge to the true error. While both errors converge as time progresses, the observed error (blue curve) is consistently lower than the underlying true error (red curve) after the third data point. While the difference between the converged error values is small (around 0.001) the observed error is an underestimation of the underlying true error.'''

    ############ Plot - Partially Correct Exemplars ############

    plot_partially_correct_omission_user = \
'''Caption:
Approximation errors Err(X,X*) := ||X - X*||_F / ||X*||_F for sparse PCA and NMF on synthetic data with varying column sparsity k of W and projection dimension d. The values of d correspond to 10x, 5x, and 2.5x compression respectively. Err(W_tilde, PW) measures the distance between factors in the compressed domain: low error here is necessary for accurate sparse recovery. Err(W_hat, W) measures the error after sparse recovery: the recovered factors W_hat typically incur only slightly higher error than the oracle lower bound (dotted lines) where PW is known exactly.

Question:
What is the effect of increasing the projection dimension d on the approximation error for sparse PCA and NMF?'''
    plot_partially_correct_omission_assistant = \
'''Student:
As d increases, the approximation error becomes lower for both sparse PCA and NMF.

Agent:
Verdict = partially correct
Error Category = omission
Feedback = Your answer is partially correct because you noticed that increasing d lowers the approximation error. However, you omitted the explanation of why this happens and what the impact is on recovering the original factors. The key reason is that a higher projection dimension d enables a more accurate representation of the original data, leading to improved approximation and sparse recovery. This insight is important for fully understanding the figure.'''

    plot_partially_correct_factual_user = \
'''Caption:
Figure 4: Scatter plot of SMT (statistical machine translation) and CCA BLEU scores versus human ratings.

Question:
What is the relationship between BLEU score and human ranking for CCA and SMT systems?'''
    plot_partially_correct_factual_assistant = \
'''Student:
Higher BLEU scores are strongly associated with higher human rankings for both CCA and SMT systems, as seen in the scatter plot where points with higher BLEU scores generally correspond to higher ratings.

Agent:
Verdict = partially correct
Error Category = factual
Feedback = Your answer is partially correct because you noticed that BLEU scores and human rankings are being compared, but the factual error is in stating that the association is strong. The correlation values reported (0.3 for CCA and 0.31 for SMT) are quite low, indicating only a weak positive correlation. This means that, although there is a slight trend where higher BLEU scores may correspond to higher ratings, it is not a strong relationship—higher BLEU scores do not consistently equate to higher human rankings.'''

    plot_partially_correct_conceptual_user = \
'''Caption:
Results of LSTNet in the ablation tests on the Solar-Energy, Traffic and Electricity dataset

Question:
How does the performance of LSTNet-attn vary with the horizon on the Solar-Energy dataset?'''
    plot_partially_correct_conceptual_assistant = \
'''Student:
As the horizon increases on the Solar-Energy dataset, the performance of LSTNet-attn gets worse because the RMSE values get higher and the correlation goes down.

Agent:
Verdict = partially correct
Error Category = conceptual
Feedback = Your answer is partially correct because you correctly observed that RMSE is one measure of performance and correlation is another. However, your conceptual error is misunderstanding the direction of improvement: for the Solar-Energy dataset, lower RMSE values actually indicate better performance, while higher correlation values are also better. In the LSTNet-attn case, as the horizon increases, RMSE decreases and correlation increases, which means performance is actually improving with longer horizons, not worsening.'''

    ############ Table Exemplars ############

    table_incorrect_factual_user = \
'''Caption:
Table 1. Single-shot and multi-shot person re-identification performance on the test set of DPI-T, BIWI and IIT PAVIS.
Dashes indicate that no published result is available.

Question:
Which method achieves the highest Top-1 Accuracy for multi-shot person re-identification on the BIWI dataset, and how does it compare to the best single-shot method on the same dataset?'''
    table_incorrect_factual_assistant = \
'''Student:
The authors method with RTA attention does best for the multi-shot task with 76.3% accuracy. It has a better accuracy than the best single-shot method which is the authors proposed method using CNN. That only has an accuracy of 66.8%.

Agent:
Verdict = incorrect
Error Category = factual
Feedback = This answer is incorrect mainly due to the fact that you quoted the wrong data-set numbers. Above answer would be correct for DPI-T dataset but the question asks the comparison with respect to the BIWI dataset. For BIWI dataset authors method with RTA attention is the best performer with 50% Top-1 accuracy and performs much better than the best single-shot performer which is the 3D-RAM method that only achieves 30.1% Top-1 accuracy.'''

    table_incorrect_omission_user = \
'''Caption:
Table 2: Scene description evaluation results on the test set, comparing the systems from Ortiz et al. to our CCA inference algorithm (the first six results are reported from the Ortiz et al. paper).
The CCA result uses m = 120 and η = 0.05, tuned on the development set.
See text for details about each of the first six baselines.

Question:
Which system from Ortiz et al. achieved the highest BLEU and METEOR scores, and how does it compare to the CCA inference algorithm in terms of performance?'''
    table_incorrect_omission_assistant = \
'''Student:
The Template method.

Agent:
Verdict = incorrect
Error Category = omission
Feedback = Your answer is incorrect for two reasons. Firstly, the SMT method has the best BLUE(43.7) and METEOR(36.6) scores. Secondly, the answer didn't compare it against the CCA method. The CCA method has much lower BLEU(26.1) and METEOR(25.6) scores than the SMT method.'''

    table_incorrect_conceptual_user = \
'''Caption:
Table 1 Worst-case number of blanks in a maximal arrangement of pairwise-disjoint Spieces contained in a cake C. From Akopyan and Segal-Halevi (2018).

Question:
What can you say about the relationship between the complexity of a cake shape and the minimum number of blanks required for a complete partition into smaller pieces?'''
    table_incorrect_conceptual_assistant = \
'''Student:
The number of blanks doesn't depend on the complexity of the cake shape; as long as you use the right kind of pieces, you can always partition the cake without any blanks.

Agent:
Verdict = incorrect
Error Category = conceptual
Feedback = This answer is incorrect because it misunderstands the relationship between cake complexity and the number of blanks in the partition. The table shows that while simple polygons and general polygons can be partitioned without blanks, more complex shapes (such as axes-parallel rectangles or rectilinear polygons with reflex vertices) require a larger minimum number of blanks, as indicated by nonzero formulas that grow with the cake's complexity. Thus, as the shape becomes more complex, the minimum number of blanks required also tends to increase.'''

    ############ Table - Partially Correct Exemplars ############

    table_partially_correct_omission_user = \
'''Caption:
Table 4: Evaluation results on response selection

Question:
Which model performs the best for response selection, and how can we tell?'''
    table_partially_correct_omission_assistant = \
'''Student:
The KEHNN model performs the best because it has the highest score in R2@1.

Agent:
Verdict = partially correct
Error Category = omission
Feedback = Your answer is partially correct because you identified that KEHNN has the top score for one of the metrics (R2@1). However, you omitted that KEHNN actually has the highest scores across all evaluation metrics (R2@1, R10@1, R10@2, R10@5) in the table. To fully justify which model performs best, you need to consider performance across all reported metrics, not just one.'''

    table_partially_correct_factual_user = \
'''Caption:
Table 2. Results on unsupervised monocular depth estimation using the KITTI dataset, building upon the model from "Baseline". By replacing the per-pixel loss used by Baseline with several variants of our own per-wavelet general loss function in which our loss's shape parameters are fixed, annealed, or adaptive, we see a significant performance improvement. The top three techniques are colored red, orange, and yellow for each metric.

Question:
Which method for setting the shape parameter of the proposed loss function achieved the best performance in terms of average error? How much improvement did it offer compared to the reproduced baseline?'''
    table_partially_correct_factual_assistant = \
'''Student:
Ours, fixed alpha = 0 achieved the best average error and improved over the baseline by about 11%.

Agent:
Verdict = partially correct
Error Category = factual
Feedback = Your answer is partially correct because, although "Ours, fixed alpha = 0" does show improvement over the baseline, it is not the best performing method in terms of average error. The "adaptive alpha in (0, 2)" strategy actually achieves the lowest (best) average error, with 0.332, which is a greater improvement—about 17% over the reproduced baseline value of 0.398. Make sure to carefully compare all values and look for the lowest number in the Avg column, which represents better performance since lower is better.'''

    table_partially_correct_conceptual_user = \
'''Caption:
Table 4: Test set accuracies (%) for all models; Match. represents test set performance on the MultiNLI genres that are also represented in the training set, Mis. represents test set performance on the remaining ones; Most freq. is a trivial 'most frequent class' baseline.

Question:
How does the performance of the ESIM model differ when trained on MNLI alone versus trained on both MNLI and SNLI combined?'''
    table_partially_correct_conceptual_assistant = \
'''Student:
When trained on MNLI alone, ESIM achieves its highest accuracy on the MNLI matched and mismatched genres, but when trained on both MNLI and SNLI, the improvement is mainly in SNLI accuracy, and MNLI performance stays about the same.

Agent:
Verdict = partially correct
Error Category = conceptual
Feedback = Your answer is partially correct because you have noticed that ESIM's SNLI accuracy improves significantly when training data is combined, but there is a conceptual error in your comparison of MNLI results. The improvement in ESIM's SNLI performance from 60.7% (MNLI only) to 79.7% (MNLI+SNLI) is correct, but for MNLI genres, the accuracy actually shows slight changes—72.3% to 72.4% (matched) and 72.1% to 71.9% (mismatched). Make sure to interpret small differences accurately and note that improvement is primarily SNLI-specific.'''

    ############ Figure Exemplars ############

    figure_incorrect_factual_user = \
'''Caption:
Overview of Step 2 of TFBA. Induction of higher-order schemata from the tri-partite graph formed from the columns of matrices A, B, and C.
Triangles in this graph (solid) represent a 3-ary schema, n-ary schemata for n > 3 can be induced from the 3-ary schemata.
Please refer to Section 3.2.2 for details.

Question:
What does the induced schema Win <A4, B3, C2> represent?'''
    figure_incorrect_factual_assistant = \
'''Student:
Player B3 won tournament C2, defeating player A4.

Agent:
Verdict = incorrect
Error Category = factual
Feedback = Your answer reverses the roles of the players in the schema. In the induction of the schema Win <A4, B3, C2>, A4 is listed first, which consistently represents the winner, while B3 is the defeated player, and C2 is the tournament. It is critical to follow the schema's structure, where the first argument is the winner. So, Win <A4, B3, C2> specifically means that player A4 won tournament C2, with B3 being the player defeated by A4.'''

    figure_incorrect_omission_user = \
'''Caption:
Figure 2: The dialog history and KB tuples stored in the memory have memory cell representations and token representations.
The encoder understands the last user utterance using only the memory cell representations.
The decoder generates the next response using both representations.

Question:
How does the encoder understand the last user utterance?'''
    figure_incorrect_omission_assistant = \
'''Student:
It uses the dialog history to understand the last user utterance.

Agent:
Verdict = incorrect
Error Category = omission
Feedback = Your answer is incomplete because it leaves out a key detail: the encoder does not just use the dialog history, but specifically uses the memory cell representations of both the dialog history and the KB (knowledge base) tuples.'''

    figure_incorrect_conceptual_user = \
'''Caption:
Fig. 3. Our model architecture consists of a frame-level feature embedding fCNN , which provides input to both a recurrent layer fLSTM and the Reinforced Temporal Attention (RTA) unit fw (highlighted in red).
The classifier is attached to the hidden state ht and its video prediction is the weighted sum of single-frame predictions, where the weights wt for each frame t are predicted by the RTA unit.

Question:
Which part of the model is responsible for deciding which frames are most important for the re-identification task?'''
    figure_incorrect_conceptual_assistant = \
'''Student:
The fLSTM (recurrent layer).

Agent:
Verdict = incorrect
Error Category = conceptual
Feedback = This answer is incorrect because the fLSTM processes frame-level features sequentially and maintains temporal information through hidden states, but it does not explicitly determine the importance or weight of each frame for the final prediction. Instead, the Reinforced Temporal Attention (RTA) unit fw, highlighted in red in the diagram, is specifically designed to assign weights (wt) to each frame based on their relevance to the task. These weights are then used to compute a weighted sum of frame predictions, identifying which frames are most critical for re-identification. Thus, the RTA unit—not the LSTM—makes decisions about frame importance.'''

    ############ Figure - Partially Correct Exemplars ############

    figure_partially_correct_omission_user = \
'''Caption:
Figure 2: An overview of the Long- and Short-term Time-series network (LSTNet)

Question:
What are the different types of layers in the LSTNet model and how are they connected?'''
    figure_partially_correct_omission_assistant = \
'''Student:
The LSTNet model contains a convolutional layer, a recurrent layer, and a fully connected output layer. The convolutional layer processes the input data first and passes it to the recurrent layer, which then passes its output to the fully connected layer to generate the prediction.

Agent:
Verdict = partially correct
Error Category = omission
Feedback = Your answer is partially correct because you identified several important layers and some connections in the LSTNet model, but you omitted key components. You did not mention the autoregressive layer, which is a crucial part of the model since it directly receives the input data and provides a linear bypass to the non-linear neural network path. Additionally, you combined the recurrent and recurrent-skip layers as just a recurrent layer, leaving out that LSTNet explicitly separates these for better capturing long-term dependencies.'''

    figure_partially_correct_factual_user = \
'''Caption:
Cost Volumes

Question:
What is the difference between a standard cost volume and a deformable cost volume?'''
    figure_partially_correct_factual_assistant = \
'''Student:
In a standard cost volume, the matching costs are computed between a neighborhood in the first image and a neighborhood in the second image directly. In a deformable cost volume, the matching costs are computed using a larger neighborhood around each location, but it uses the same locations in both images, just with a bigger region.

Agent:
Verdict = partially correct
Error Category = factual
Feedback = Your answer is partially correct because you acknowledge that the deformable cost volume uses a larger (dilated) neighborhood compared to the standard version. However, you missed a key distinction: in the deformable cost volume, the neighborhood in the second image is not centered at the same location as in the first image, but is instead offset by a flow vector. This offset allows the cost computation to account for motion or deformation between the two images, which is not the case in the standard cost volume.'''

    figure_partially_correct_conceptual_user = \
'''Caption:
No single saliency map can perform best in all metrics even when the true fixation distribution is known. This problem can be solved by separating saliency models from saliency maps. a) Fixations are distributed according to a ground truth fixation density p(x, y | I) for some stimulus I. b) This ground truth density predicts different saliency maps depending on the intended metric. c) Performances of the saliency maps from b) under seven saliency metrics on a large number of fixations sampled from the model distribution in a).

Question:
What is the relationship between the ground truth fixation density and the saliency maps?'''
    figure_partially_correct_conceptual_assistant = \
'''Student:
The ground truth fixation density generates a single optimal saliency map, and this map is then evaluated with different metrics. The performance of the saliency map might vary across metrics because not all metrics are equally suited, but the saliency map itself is determined just by the fixation density.

Agent:
Verdict = partially correct
Error Category = conceptual
Feedback = Your answer is partially correct because you recognize the central role of the ground truth fixation density and acknowledge that metric choice affects performance outcomes. However, the key conceptual error is thinking that a single optimal saliency map arises from the fixation density. In fact, each metric requires a different saliency map to achieve optimal performance—the fixation density predicts different best saliency maps depending on which metric you want to optimize. Each map is specifically tailored to its corresponding metric, which is why they look different and perform best only for their metric.'''

    # Exemplars restructured as {figure_category: {verdict: {error_type: (user, assistant, image_path)}}}
    exemplars = {
        'plot': {
            'incorrect': {
                'factual':    (plot_incorrect_factual_user,    plot_incorrect_factual_assistant,    'data/test-A/SPIQA_testA_Images/1811.02721v3/1811.02721v3-Figure8-1.png'),
                'omission':   (plot_incorrect_omission_user,   plot_incorrect_omission_assistant,   'data/test-A/SPIQA_testA_Images/1803.03467v4/1803.03467v4-Figure4-1.png'),
                'conceptual': (plot_incorrect_conceptual_user, plot_incorrect_conceptual_assistant, 'data/test-A/SPIQA_testA_Images/1702.03584v3/1702.03584v3-Figure1-1.png'),
            },
            'partially correct': {
                'factual':    (plot_partially_correct_factual_user,    plot_partially_correct_factual_assistant,    'data/test-A/SPIQA_testA_Images/1608.02784v2/1608.02784v2-Figure4-1.png'),
                'omission':   (plot_partially_correct_omission_user,   plot_partially_correct_omission_assistant,   'data/test-A/SPIQA_testA_Images/1706.08146v3/1706.08146v3-Figure2-1.png'),
                'conceptual': (plot_partially_correct_conceptual_user, plot_partially_correct_conceptual_assistant, 'data/test-A/SPIQA_testA_Images/1703.07015v3/1703.07015v3-Figure5-1.png'),
            },
        },
        'table': {
            'incorrect': {
                'factual':    (table_incorrect_factual_user,    table_incorrect_factual_assistant,    'data/test-A/SPIQA_testA_Images/1705.09882v2/1705.09882v2-Table1-1.png'),
                'omission':   (table_incorrect_omission_user,   table_incorrect_omission_assistant,   'data/test-A/SPIQA_testA_Images/1608.02784v2/1608.02784v2-Table2-1.png'),
                'conceptual': (table_incorrect_conceptual_user, table_incorrect_conceptual_assistant, 'data/test-A/SPIQA_testA_Images/1603.00286v5/1603.00286v5-Table1-1.png'),
            },
            'partially correct': {
                'factual':    (table_partially_correct_factual_user,    table_partially_correct_factual_assistant,    'data/test-A/SPIQA_testA_Images/1701.03077v10/1701.03077v10-Table2-1.png'),
                'omission':   (table_partially_correct_omission_user,   table_partially_correct_omission_assistant,   'data/test-A/SPIQA_testA_Images/1611.04684v1/1611.04684v1-Table4-1.png'),
                'conceptual': (table_partially_correct_conceptual_user, table_partially_correct_conceptual_assistant, 'data/test-A/SPIQA_testA_Images/1704.05426v4/1704.05426v4-Table4-1.png'),
            },
        },
        'figure': {
            'incorrect': {
                'factual':    (figure_incorrect_factual_user,    figure_incorrect_factual_assistant,    'data/test-A/SPIQA_testA_Images/1805.01216v3/1805.01216v3-Figure2-1.png'),
                'omission':   (figure_incorrect_omission_user,   figure_incorrect_omission_assistant,   'data/test-A/SPIQA_testA_Images/1805.01216v3/1805.01216v3-Figure2-1.png'),
                'conceptual': (figure_incorrect_conceptual_user, figure_incorrect_conceptual_assistant, 'data/test-A/SPIQA_testA_Images/1705.09882v2/1705.09882v2-Figure3-1.png'),
            },
            'partially correct': {
                'factual':    (figure_partially_correct_factual_user,    figure_partially_correct_factual_assistant,    'data/test-A/SPIQA_testA_Images/1802.07351v2/1802.07351v2-Figure2-1.png'),
                'omission':   (figure_partially_correct_omission_user,   figure_partially_correct_omission_assistant,   'data/test-A/SPIQA_testA_Images/1703.07015v3/1703.07015v3-Figure2-1.png'),
                'conceptual': (figure_partially_correct_conceptual_user, figure_partially_correct_conceptual_assistant, 'data/test-A/SPIQA_testA_Images/1704.08615v2/1704.08615v2-Figure1-1.png'),
            },
        },
    }

    run_inference(
        args.json, args.images_root,
        exemplars=exemplars,
        output=args.output,
        checkpoint_every=args.checkpoint_every,
        num_papers=args.num_papers,
        paper_range=tuple(args.range) if args.range else None,
    )
