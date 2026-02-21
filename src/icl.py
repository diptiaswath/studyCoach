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
python icl.py test-A/SPIQA_testA_part1.json test-A/SPIQA_testA_Images --output test-A/SPIQA_testA_part1_output.json
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

Omission: An error due to omitting one or more key points in the answer
Factual: An error due to student giving factual data which contradicts the information in the figure/chart/table, or misreading the figure/chart/table (e.g., misreading axes, legends, trends, etc.) 
Conceptual: An error due to student misunderstanding a concept or using figure/chat/table data to come to a wrong conclusion

You are given a question (and the associated image and caption) and your task is to generate 
  1. A $ANSWER_TYPE answer a student is likely to give when posed the question
  2. The study coach feedback on why the student answer is wrong. 
  
$ANSWER_TYPE_DESCRIPTION

Follow below guidelines

- Use examples given above to figure out what type of error (i.e.: Omission, Factual, Conceptual) the student is most likely to make for the given question.
- Assume that you have previously generated $FACTUAL factual, $OMISSION omission and $CONCEPTUAL conceptual error examples so far. Try to balance the error types across all examples. 
  For instance, if you have already generated 2 examples of Omission and 0 examples of Factual and Conceptual, then for the current example you should try to generate a Factual or Conceptual error if it makes sense for the question.
- You need to generate the wrong answer having one of the errors above (i.e.: Omission, Factual, Conceptual).
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
Error Category = <Error type - One of Omission, Factual, Conceptual> 
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
            
            if figure_category not in exemplars:
                continue

            figure_category = figure_category.strip()
            if figure_category not in ['plot', 'table', 'figure']:
                figure_category = 'figure'

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

                if "Factual" in resp.output_text:
                    FACTUAL += 1 
                elif "Omission" in resp.output_text:
                    OMISSION += 1
                elif "Conceptual" in resp.output_text:
                    CONCEPTUAL += 1
                qa_counter += 1

                parsed = parse_inference_output(resp.output_text)
                qa["student"] = parsed["student"]
                qa["verdict"] = parsed["verdict"]
                qa["error_category"] = parsed["error_category"]
                qa["feedback"] = parsed["feedback"]
                
            else:
                qa["student"] = qa.get("answer", "").strip()
                qa["student"] = qa["student"] + " " + qa.get("explanation", "").strip() if qa["student"].endswith(".") else qa["student"] + ". " + qa.get("explanation", "").strip()
                qa["verdict"] = "correct"
                qa["error_category"] = "N/A"
                qa["feedback"] = "Your answer is correct. Great job!"
            
        if FACTUAL > 4 and OMISSION > 4 and CONCEPTUAL > 4: 
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
    
    chart_incorrect_omission_user = \
'''Caption:
Figure 3: Results on real data. Regarding the scale of precision and F-measure, see the comment at the last paragraph just before Section 3. 
The y-axis is in logarithmic scale. C-Tarone is shown in red and the binarization approach is shown in blue. 
Higher (taller) is better in precision, recall, and F-measure, while lower is better in running time.

Question:
How does the C-Tarone method compare to the binarization method in terms of precision, recall, F-measure, and running time?'''
    chart_incorrect_omission_assistant = \
'''Student:
The C-Tarone method is generally similar to the binarization method across all datasets.

Agent:
Verdict = Incorrect
Error Category = Omission
Feedback = Your answer omits key details regarding the differences between the two methods. 
You should specify how C-Tarone compares to binarization in terms of precision, recall, F-measure, and running time. 
According to the figure, C-Tarone consistently has higher precision and F-measure than binarization across all datasets. 
Its recall is generally better or at least competitive, and its running time is also competitive with binarization.'''

    chart_incorrect_factual_user = \
'''Caption:
Experiments with synthetic data: error is reported against the size of the conditional probability table (lower is better). 
The error is the estimated total variation distance to the ground truth Bayes net. We use the error of MLE without noise as our benchmark. 
We plot the performance of our algorithm (Filtering), empirical mean with noise (MLE), and RANSAC. 
We report two settings: the underlying structure of the Bayes net is a random tree (left) or a random graph (right).

Question: 
How does the performance of the Filtering algorithm compare to the performance of MLE with noise?'''
    chart_incorrect_factual_assistant = \
'''Student: 
Filtering performs about the same as MLE with noise since their error lines overlap for both the random tree and random graph cases.

Agent:
Verdict = Incorrect
Error Category = Factual
Feedback = This answer is incorrect because it misreads the figure. 
When looking at both plots, the Filtering algorithm (black, xx line) always lies below the red line for MLE with noise, indicating Filtering consistently achieves lower error.
In this chart, a lower value on the y-axis (Estimated d_{TV}) means better performance.
So, Filtering clearly outperforms MLE with noise for all numbers of parameters in both the random tree (left) and random graph (right) experiments.'''

    chart_incorrect_conceptual_user = \
'''Caption:
Training with different gradient approximations: validation loss with a simplified advection (red), and the correct gradient from forward advection (green).
The simplified version does not converge.

Question: 
Which gradient approximation method trains better?'''
    chart_incorrect_conceptual_assistant = \
'''Student: 
The naive gradient approximation trains better because its loss stays high and stable, which I think means it’s not overfitting, while the corrected gradient’s loss keeps dropping and fluctuates more.

Agent:
Verdict = Incorrect
Error Category = Conceptual
Feedback = Your answer is incorrect because a higher loss means the model is performing worse, not better. 
The y-axis shows loss, and lower values indicate better model performance. The "naive gradient" loss stays high and does not decrease, showing it doesn’t converge or improve with training. 
The "corrected gradient" results in a steadily decreasing loss, reaching much lower values, which demonstrates better training and generalization. 
The goal is to minimize loss, not keep it high.'''

    exemplars = {
        'plot' : [
            (chart_incorrect_omission_user, chart_incorrect_omission_assistant, 'data/test-A/SPIQA_testA_Images/1702.08694v3/1702.08694v3-Figure3-1.png'),
            (chart_incorrect_factual_user, chart_incorrect_factual_assistant, 'data/test-A/SPIQA_testA_Images/1606.07384v2/1606.07384v2-Figure1-1.png'),
            (chart_incorrect_conceptual_user, chart_incorrect_conceptual_assistant, 'data/test-A/SPIQA_testA_Images/1704.07854v4/1704.07854v4-Figure15-1.png')
        ]
    }

    updated_json = run_inference(args.json, args.images_root, exemplars=exemplars)

    Path(args.output).write_text(
        json.dumps(updated_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
