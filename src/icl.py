#!/usr/bin/env python3
"""Generates synthetic student responses for SPIQA dataset questions.

Does the following:
- Reads a JSON file (same format as SPIQA_testA.json)
- For each QA example:
    - Creates one OpenAI inference call using
        - a templated system prompt
        - several ICL exemplars, each referencing the some image from test-A images (exemplars are reused across questions) 
        - a final user message containing the actual question.
    - Runs the inference and updates the JSON with model outputs (student answer, verdict, error category, feedback).

Example usage:
python src/icl.py data/SPIQA_testA_part1.json data/test-A/SPIQA_testA_Images --output data/test-A/SPIQA_testA_part1_output.json
"""
from __future__ import annotations

import base64
import json
import re

from openai import OpenAI
from pathlib import Path
from string import Template
from typing import Dict, List, Any

ANSWER_TYPE = "incorrect"
ANSWER_TYPE_DESCRIPTION = ""

FACTUAL, OMISSION, CONCEPTUAL = (0, 0, 0)
DEFAULT_SYSTEM_PROMPT = \
Template('''
You are a study coach who helps students improve their understanding of scientific papers. 
Students are required to answer a question based on an image (and its caption) obtained from a scientific paper. 
You are given examples questions and student answers where students have made particular a type of mistake
as described below.

omission: An error due to omitting one or more key points in the answer
factual: An error due to student giving factual data which contradicts the information in the figure/chart/table, or misreading the figure/chart/table (e.g., misreading axes, legends, trends, etc.)
conceptual: An error due to student misunderstanding a concept or using figure/chat/table data to come to a wrong conclusion

You are given a question (and the associated image and caption) and your task is to generate 
  1. A $ANSWER_TYPE answer a student is likely to give when posed the question
  2. The study coach feedback on why the student answer is wrong. 
  
$ANSWER_TYPE_DESCRIPTION

Follow below guidelines

- Use examples given above to figure out what type of error (i.e.: omission, factual, conceptual) the student is most likely to make for the given question.
- Assume that you have previously generated $FACTUAL factual, $OMISSION omission and $CONCEPTUAL conceptual error examples so far. Try to balance the error types across all examples. 
  For instance, if you have already generated 2 examples of omission and 0 examples of factual and conceptual, then for the current example you should try to generate a factual or conceptual error if it makes sense for the question.
- You need to generate the wrong answer having one of the errors above (i.e.: omission, factual, conceptual).
- Use a second person instructional tone in study coach feedback. Aim to explain what the student's misunderstanding or confusion is. 
- Make the feedback constructive but as concise as possible without missing any important points which aids student understanding.
- Do not repeat information. 
- When explaining errors due to misreading the plots mention what higher values means for important axes (if applicable to the error explanation).
- Pay special attention to correctly read plot legends.
- Don't do any text formatting (e.g., don't use bullet points, bold, etc.) in the output.
- Format the output as below.
Student: 
<Wrong student answer>

Agent:
Verdict = $ANSWER_TYPE 
Error Category = <Error type - One of omission, factual, conceptual>
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

    verdict_match = re.search(r"Verdict\s*=\s*([^\n]+)", agent_block) 
    error_match = re.search(r"Error Category\s*=\s*([^\n]+)", agent_block)
    feedback_match =  re.search(r"Feedback\s*=\s*(.*?)(?:\n[A-Za-z ]+\s*=|\Z)", agent_block, re.DOTALL)

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
        "student": student,
        "verdict": verdict,
        "error_category": error_category if error_category else "N/A",
        "feedback": feedback if feedback else "N/A",
    }

def to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # change mime type if needed: image/png, image/webp, etc.
    return f"data:image/jpeg;base64,{b64}"

def build_exemplar_messages(exemplars: List[tuple[str, str, str]]) -> tuple[List[Dict[str, Any]], List[str]]:
    messages: List[Dict[str, Any]] = []
    texts = []
    for exemplar in exemplars:
        user, assistant, exemplar_image_path = exemplar
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"{user}"},
                    {"type": "input_image", "image_url": to_data_url(exemplar_image_path)}
                ],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": f"{assistant}"}
                ],
            }
        )
        texts.append(user)
        texts.append(assistant)

    return messages, texts

client = OpenAI()

def run_inference(
    json_path: str | Path,
    images_root: str | Path,
    exemplars: dict
    ):
    """Run inference and inline-update JSON with model outputs."""
    global FACTUAL, OMISSION, CONCEPTUAL, ANSWER_TYPE, ANSWER_TYPE_DESCRIPTION

    json_path = Path(json_path)
    images_root = Path(images_root)
    
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f"JSON : {json_path}")
    print(f"Images: {images_root}")
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

    data = load_json(json_path)

    # num_papers = len(data)
    num_papers = 9
    partially_correct_split, incorrect_split, correct_split = num_papers // 3, num_papers * 2 // 3, num_papers
    for paper_idx, paper_key in enumerate(sorted(data.keys()), 1):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(f"Processing paper - {paper_key}")
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

        if paper_idx <= partially_correct_split:
            ANSWER_TYPE = "partially correct"
            ANSWER_TYPE_DESCRIPTION = "A partially correct answer is an answer which gets some aspects correct with respect to the question posed in the context of the provided image and caption. But not all aspects are correct."
        elif paper_idx <= incorrect_split:
            ANSWER_TYPE = "incorrect"
            ANSWER_TYPE_DESCRIPTION = "An incorrect answer is an answer which is incorrect with respect to the question posed in the context of the provided image and caption."
        else:
            ANSWER_TYPE = "correct"
            ANSWER_TYPE_DESCRIPTION = "A correct answer is an answer which is provides the accurate and complete information to the question posed in the context of the provided image and caption."

        paper = data[paper_key]
        all_figures = paper.get("all_figures", {})

        qa_list = paper.get("qa", [])
        qa_counter = 1
        for qa_idx, qa in enumerate(qa_list, 1):
            question = (qa.get("question", "") or "").strip()
            figure = (qa.get("reference", "") or "").strip()
            
            # Normalize figure category to be one of 'plot', 'figure' or 'table'
            figure_details = all_figures.get(figure, None) if figure != "" else None
            figure_path = images_root/ paper_key / figure if figure_details else None
            
            figure_content_type = figure_details.get("content_type", "") 
            figure_type = figure_details.get("figure_type", "")
            figure_category = figure_content_type if "N/A" in figure_type else figure_type
            
            figure_category = figure_category.strip()
            if figure_category not in ['plot', 'table', 'figure']:
                figure_category = 'figure'

            assert figure_category in exemplars, f"Figure category {figure_category} not found in exemplars. Please check the figure content type and figure type in the JSON and ensure it is one of 'plot', 'table' or 'figure'."

            # Build the message list (system prompt passed via top-level instructions)
            messages, texts = build_exemplar_messages(exemplars[figure_category])

            # Final user message with placeholders for unspecified content
            caption = figure_details.get("caption", "").strip() 
            user_input = f'Caption:\n{caption}\nQuestion:\n{question}'
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{user_input}"},
                        {"type": "input_image", "image_url": to_data_url(figure_path)}
                    ],
                }
            )

            # texts.append("\n\n" + user_input)
            # prompt = "\n".join(texts) 

            # Only run inference for incorrect and partially correct answers since correct answers can be sourced from the SPIQA ground truth answers.
            if ANSWER_TYPE != "correct":
                call = {
                        "model": DEFAULT_MODEL,
                        "instructions": DEFAULT_SYSTEM_PROMPT.safe_substitute(FACTUAL=FACTUAL, OMISSION=OMISSION, CONCEPTUAL=CONCEPTUAL, ANSWER_TYPE=ANSWER_TYPE, ANSWER_TYPE_DESCRIPTION=ANSWER_TYPE_DESCRIPTION),
                        "input": messages,
                        "metadata": {
                            "paper": paper_key,
                            "qa_index": f'{qa_idx}',
                        },
                    }

                print(f'\n--------------------------------- Question {qa_counter} ---------------------------------')
                print('Prompt>\n')
                print(call["instructions"] + "\n\n" + user_input)
                print('==============================================================================')
                
                resp = client.responses.create(**call)

                print('Output>\n')
                print(f'{resp.output_text}')
                print('------------------------------------------------------------------------------\n')

                if "factual" in resp.output_text:
                    FACTUAL += 1 
                elif "omission" in resp.output_text:
                    OMISSION += 1
                elif "conceptual" in resp.output_text:
                    CONCEPTUAL += 1
                qa_counter += 1

                parsed = parse_inference_output(resp.output_text)
                qa["student"] = parsed["student"]
                qa["verdict"] = parsed["verdict"]
                qa["error_category"] = parsed["error_category"]
                qa["feedback"] = parsed["feedback"]
                
            else:
                qa["student"] = qa.get("answer", "").strip()
                qa["student"] = qa["student"] if qa["student"].endswith(".") else qa["student"] + "."
                qa["verdict"] = "correct"
                qa["error_category"] = "N/A"
                qa["feedback"] = "Your answer is correct. Great job!"
            
        if FACTUAL > 15 and OMISSION > 15 and CONCEPTUAL > 15:
            break

    print(f"\nFactual: {FACTUAL}")
    print(f"Omission: {OMISSION}")
    print(f"Conceptual: {CONCEPTUAL}")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build OpenAI inference call payloads.")
    parser.add_argument("json", help="Path to JSON file")
    parser.add_argument("images_root", help="Path to images root folder")
    parser.add_argument("--output", required=True, help="Path to write updated JSON output")
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
The average number of k-hop neighbors that two items share in the KG w.r.t. whether they have common raters in (a) MovieLens-1M, (b) Book-Crossing, and (c) BingNews datasets. (d) The ratio of the two average numberswith different hops.

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

    ## --> TBD(Dipti): Add partially incorrect exemplars for the plots here.

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
Feedback = Your answer is incorrect for two reasons. Firstly, the SMT method has the best BLUE(43.7) and METEOR(36.6) scores. Secondly, the answer didn’t compare it against the CCA method. The CCA method has much lower BLEU(26.1) and METEOR(25.6) scores than the SMT method.'''

    table_incorrect_conceptual_user = \
'''Caption:
Table 1 Worst-case number of blanks in a maximal arrangement of pairwise-disjoint Spieces contained in a cake C. From Akopyan and Segal-Halevi (2018).

Question:
What can you say about the relationship between the complexity of a cake shape and the minimum number of blanks required for a complete partition into smaller pieces?'''
    table_incorrect_conceptual_assistant = \
'''Student:
The number of blanks doesn’t depend on the complexity of the cake shape; as long as you use the right kind of pieces, you can always partition the cake without any blanks.

Agent:
Verdict = incorrect
Error Category = conceptual
Feedback = This answer is incorrect because it misunderstands the relationship between cake complexity and the number of blanks in the partition. The table shows that while simple polygons and general polygons can be partitioned without blanks, more complex shapes (such as axes-parallel rectangles or rectilinear polygons with reflex vertices) require a larger minimum number of blanks, as indicated by nonzero formulas that grow with the cake's complexity. Thus, as the shape becomes more complex, the minimum number of blanks required also tends to increase.'''

    ## --> TBD(Dipti): Add partially incorrect exemplars for the tables here.

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
Feedback = Your answer reverses the roles of the players in the schema. In the induction of the schema Win <A4, B3, C2>, A4 is listed first, which consistently represents the winner, while B3 is the defeated player, and C2 is the tournament. It is critical to follow the schema’s structure, where the first argument is the winner. So, Win <A4, B3, C2> specifically means that player A4 won tournament C2, with B3 being the player defeated by A4.'''

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

    ## --> TBD(Dipti): Add partially incorrect exemplars for the figures here.

    exemplars = {
        'plot' : [
            (plot_incorrect_factual_user, plot_incorrect_factual_assistant, 'data/test-A/SPIQA_testA_Images/1811.02721v3/1811.02721v3-Figure8-1.png'),
            (plot_incorrect_omission_user, plot_incorrect_omission_assistant, 'data/test-A/SPIQA_testA_Images/1803.03467v4/1803.03467v4-Figure4-1.png'),
            (plot_incorrect_conceptual_user, plot_incorrect_conceptual_assistant, 'data/test-A/SPIQA_testA_Images/1702.03584v3/1702.03584v3-Figure1-1.png')
            ## --> TBD(Dipti): Add partially incorrect exemplars for the plots here.
        ],
        'table' : [
            (table_incorrect_factual_user, table_incorrect_factual_assistant, 'data/test-A/SPIQA_testA_Images/1705.09882v2/1705.09882v2-Table1-1.png'),
            (table_incorrect_omission_user, table_incorrect_omission_assistant, 'data/test-A/SPIQA_testA_Images/1608.02784v2/1608.02784v2-Table2-1.png'),
            (table_incorrect_conceptual_user, table_incorrect_conceptual_assistant, 'data/test-A/SPIQA_testA_Images/1603.00286v5/1603.00286v5-Table1-1.png')
            ## --> TBD(Dipti): Add partially incorrect exemplars for the tables here.
        ],
        'figure' : [
            (figure_incorrect_factual_user, figure_incorrect_factual_assistant, 'data/test-A/SPIQA_testA_Images/1805.01216v3/1805.01216v3-Figure2-1.png'),
            (figure_incorrect_omission_user, figure_incorrect_omission_assistant, 'data/test-A/SPIQA_testA_Images/1805.01216v3/1805.01216v3-Figure2-1.png'),
            (figure_incorrect_conceptual_user, figure_incorrect_conceptual_assistant, 'data/test-A/SPIQA_testA_Images/1705.09882v2/1705.09882v2-Figure3-1.png')
            ## --> TBD(Dipti): Add partially incorrect exemplars for the figures here.
        ]
    }

    updated_json = run_inference(args.json, args.images_root, exemplars=exemplars)

    Path(args.output).write_text(
        json.dumps(updated_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
