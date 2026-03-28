"""Seed Example Generator for SPIQA+ Dataset

This module generates synthetic "seed" exemplar examples for in-context learning (ICL) bootstrapping.
Seed examples are used as reference examples in icl.py to teach the model how to generate different
types of student errors (Omission, Factual, Conceptual) on scientific paper figure questions.

WORKFLOW:
1. User provides: image, caption, question, correct_answer + desired error_type
2. generate_seed_example() calls OpenAI API with image + prompt
3. OpenAI generates: wrong_student_answer + coaching_feedback
4. Output is a SeedExample object, printed for manual review & curation
5. User manually copies high-quality examples to icl.py as hardcoded exemplars

INPUT FORMAT (to generate_seed_example):
- image_path (str): Path to image file, e.g., 'data/test-A/SPIQA_testA_Images/1702.08694v3/1702.08694v3-Figure3-1.png'
- caption (str): Figure caption from SPIQA metadata
- question (str): Question from SPIQA qa.question
- answer (str): Correct answer from SPIQA qa.answer
- verdict (str): 'incorrect' or 'partially correct' - what type of wrong answer to generate
- error_category (str): 'omission', 'factual', or 'conceptual' - what type of error to induce
- verdict_explanation (str): Definition of verdict, e.g., "an answer which gets none of the required key insights correct"
- error_category_explanation (str): Definition of error type, e.g., "an error due to omitting key details in the answer"

OUTPUT FORMAT (SeedExample):
A pipe-delimited string with 4 fields:
  verdict | error_category | student_answer | feedback

Example:
  Incorrect | Omission | The C-Tarone method is generally similar to the binarization method... | Your answer omits key details...

Each field:
- verdict: "Incorrect" or "Partially Correct"
- error_category: "Omission", "Factual", or "Conceptual"
- student_answer: Generated wrong answer (1-3 sentences)
- feedback: Study coach explanation of the error (2-4 sentences, instructional tone)

TYPICAL USAGE:
  python src/seed.py  # Generates Partially Correct exemplars for chart4-6, figure4-6, table4-6

NOTE: seed.py is a ONE-TIME curation tool. Generated examples are manually reviewed and then
      hardcoded into icl.py as exemplars for the main inference pipeline.
"""

import base64
from openai import OpenAI

client = OpenAI()

def to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

class SeedExample:
    verdict: str
    error_category: str
    student_answer: str
    feedback: str

    def __init__(self, verdict, error_category, student_answer, feedback):
        self.verdict = verdict
        self.error_category = error_category
        self.student_answer = student_answer
        self.feedback = feedback

    def __str__(self):
        return f"{self.verdict} | {self.error_category} | {self.student_answer} | {self.feedback}"

def generate_seed_example(image_path, caption, question, answer, verdict, error_category, verdict_explanation, error_category_explanation) -> SeedExample:
    prompt = \
    f"""
You are a study coach who helps students to improve their understanding of scientific papers.
You have been given following image from a scientific paper along with its caption and a question related to it.
The correct answer is also given.

Caption: '{caption}'

Question: '{question}'

Correct Answer: '{answer}'

I need you to generate the following.
- A potential '{verdict}' answer that a student is likely to give when posed the same question.
  Do your best to come up with a common potential mistake a student might make when reading above data.
  I prefer subtle errors over obvious or superficial errors as possible.
  The '{verdict}' answer should contain a '{error_category}' error.
- Feedback which describes why student answer is '{verdict}' while also explaining the correct answer.

A '{verdict}' answer is {verdict_explanation}.
A '{error_category}' error is {error_category_explanation}.

Give output in following format.

<Wrong student answer> | <Agent feedback>

<Wrong student answer> is the wrong answer provided by the student.
<Agent feedback> is the agent explanation as to why student answer is {verdict}.

Use a second person instructional tone in the agent explanation. Aim to explain what the student's misunderstanding or confusion is.
Make the feedback constructive but as concise as possible without missing any important points which aids student understanding.
Do not repeat information. When explaining chart/ graph errors mention what higher for important axes mean (if applicable for error
explanation).
    """

    image = to_data_url(image_path)
    resp = client.responses.create(
    model="gpt-4.1",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": f"{prompt}"},
            {"type": "input_image", "image_url": image},
        ],
    }],)

    print('=============================================================================')
    print(f'Prompt:\n')
    print(f'{prompt}')

    student_answer, feedback = resp.output_text.split("|")
    return SeedExample(verdict, error_category, student_answer, feedback)


def to_string(dict):
    return "\n".join(f"{k}: {v}" for k, v in dict.items())

if __name__ == "__main__":
    error_categories = {
        'omission' : 'an error due to omitting key details in the answer',
        'factual' : 'an error due to reading the content in the figure/chart/table incorrectly',
        'conceptual' : 'an error due to student misunderstanding a concept or using figure/chat/table data to come to a wrong conclusion'}

    verdicts = {
        "partially correct" : "an answer which gets some details partially correct but is gets some key insights, concepts or information wrong",
        "incorrect" : "an answer which gets none of the required key insights, concepts or information correct"
    }

    #######################################################################################
    # INCORRECT EXEMPLARS
    # Variable definitions for chart1-3, figure1-3, table1-3
    #######################################################################################

    ### Plot - Incorrect ###
    # Plot - Incorrect, Omission
    chart1 = 'data/test-A/SPIQA_testA_Images/1702.08694v3/1702.08694v3-Figure3-1.png'
    chart1_caption = '''Figure 3: Results on real data. Regarding the scale of precision and F-measure, see the comment at the last paragraph just before Section 3. The y-axis is in logarithmic scale. C-Tarone is shown in red and the binarization approach is shown in blue. Higher (taller) is better in precision, recall, and F-measure, while lower is better in running time.'''
    chart1_question = '''How does the C-Tarone method compare to the binarization method in terms of precision, recall, F-measure, and running time?'''
    chart1_answer = '''The C-Tarone method has higher precision and F-measure than the binarization method in all datasets. The C-Tarone method has better or competitive recall than the binarization method. The running time of the C-Tarone method is competitive with the binarization method.'''

    # Plot - Incorrect, Factual
    chart2 = 'data/test-A/SPIQA_testA_Images/1606.07384v2/1606.07384v2-Figure1-1.png'
    chart2_caption = '''Experiments with synthetic data: error is reported against the size of the conditional probability table (lower is better). The error is the estimated total variation distance to the ground truth Bayes net. We use the error of MLE without noise as our benchmark. We plot the performance of our algorithm (Filtering), empirical mean with noise (MLE), and RANSAC. We report two settings: the underlying structure of the Bayes net is a random tree (left) or a random graph (right).'''
    chart2_question = 'How does the performance of the Filtering algorithm compare to the performance of MLE with noise?'
    chart2_answer = '''The Filtering algorithm performs better than MLE with noise in both the random tree and random graph settings. The figure shows that the error of the Filtering algorithm is lower than the error of MLE with noise for all values of the number of parameters. This is true for both the random tree and random graph settings.'''

    # Plot - Incorrect, Conceptual
    chart3 = 'data/test-A/SPIQA_testA_Images/1704.07854v4/1704.07854v4-Figure15-1.png'
    chart3_caption = '''Training with different gradient approximations: validation loss with a simplified advection (red), and the correct gradient from forward advection (green). The simplified version does not converge.'''
    chart3_question = '''Which gradient approximation method trains better?'''
    chart3_answer = '''The corrected gradient method leads to a more stable and lower loss value, so leads to a better training. Simplified advection method doesn't converge.'''

    ### Figure - Incorrect ###
    # Figure - Incorrect, Omission
    figure1 = 'data/test-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Figure1-1.png'

    # Figure - Incorrect, Factual
    figure2 = 'data/test-A/SPIQA_testA_Images/1704.07854v4/1704.07854v4-Figure12-1.png'

    # Figure - Incorrect, Conceptual
    figure3 = 'data/test-A/SPIQA_testA_Images/1704.07854v4/1704.07854v4-Figure12-1.png'

    ### Table - Incorrect ###
    # Table - Incorrect, Omission
    table1 = 'data/test-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Table1-1.png'

    # Table - Incorrect, Factual
    table2 = 'data/test-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Table1-1.png'

    # Table - Incorrect, Conceptual
    table3 = 'data/test-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Table1-1.png'

    #######################################################################################
    # INCORRECT EXEMPLAR GENERATION CALLS
    # Uncomment to generate Incorrect exemplars
    #######################################################################################

    # chart_incorrect_omission = generate_seed_example(chart1, chart1_caption, chart1_question, chart1_answer, 'incorrect', 'omission', verdicts['incorrect'], error_categories['omission'])
    # print('------------------------------------------------------------------------------')
    # print('Seed Example (Plot - Incorrect - Omission):\n')
    # print(chart_incorrect_omission)
    # print('==============================================================================\n\n')

    # chart_incorrect_factual = generate_seed_example(chart2, chart2_caption, chart2_question, chart2_answer, 'incorrect', 'factual', verdicts['incorrect'], error_categories['factual'])
    # print('------------------------------------------------------------------------------')
    # print('Seed Example (Plot - Incorrect - Factual):\n')
    # print(chart_incorrect_factual)
    # print('==============================================================================\n\n')

    # chart_incorrect_conceptual = generate_seed_example(chart3, chart3_caption, chart3_question, chart3_answer, 'incorrect', 'conceptual', verdicts['incorrect'], error_categories['conceptual'])
    # print('------------------------------------------------------------------------------')
    # print('Seed Example (Plot - Incorrect - Conceptual):\n')
    # print(chart_incorrect_conceptual)
    # print('==============================================================================\n\n')

    #######################################################################################
    # PARTIALLY CORRECT EXEMPLARS
    # Variable definitions and generation calls for chart4-6, figure4-6, table4-6
    #######################################################################################

    ### Plot - Partially Correct ###
    # Plot - Partially Correct, Omission
    chart4 = 'data/test-A/SPIQA_testA_Images/1706.08146v3/1706.08146v3-Figure2-1.png'
    chart4_caption = '''Approximation errors Err(X,X*) := ||X - X*||_F / ||X*||_F for sparse PCA and NMF on synthetic data with varying column sparsity k of W and projection dimension d. The values of d correspond to 10x, 5x, and 2.5x compression respectively. Err(W_tilde, PW) measures the distance between factors in the compressed domain: low error here is necessary for accurate sparse recovery. Err(W_hat, W) measures the error after sparse recovery: the recovered factors W_hat typically incur only slightly higher error than the oracle lower bound (dotted lines) where PW is known exactly.'''
    chart4_question = '''What is the effect of increasing the projection dimension d on the approximation error for sparse PCA and NMF?'''
    chart4_answer = '''Increasing the projection dimension d decreases the approximation error for both sparse PCA and NMF. The figure shows that the approximation error decreases as the projection dimension d increases. This is because a higher projection dimension allows for a more accurate representation of the original data.'''

    # Plot - Partially Correct, Factual
    chart5 = 'data/test-A/SPIQA_testA_Images/1608.02784v2/1608.02784v2-Figure4-1.png'
    chart5_caption = '''Figure 4: Scatter plot of SMT (statistical machine translation) and CCA BLEU scores versus human ratings.'''
    chart5_question = '''What is the relationship between BLEU score and human ranking for CCA and SMT systems?'''
    chart5_answer = '''The correlation between BLEU scores and human ranking is not high for either CCA or SMT systems. The passage states that the correlation between the x-axis (ranking) and y-axis (BLEU scores) for CCA is 0.3 and for the SMT system 0.31. This indicates a weak positive correlation, meaning that higher BLEU scores are not necessarily associated with higher human rankings.'''

    # Plot - Partially Correct, Conceptual
    chart6 = 'data/test-A/SPIQA_testA_Images/1703.07015v3/1703.07015v3-Figure5-1.png'
    chart6_caption = '''Results of LSTNet in the ablation tests on the Solar-Energy, Traffic and Electricity dataset'''
    chart6_question = '''How does the performance of LSTNet-attn vary with the horizon on the Solar-Energy dataset?'''
    chart6_answer = '''The performance of LSTNet-attn generally improves as the horizon increases on the Solar-Energy dataset. This is evident from the fact that both the RMSE and correlation values improve with increasing horizon.'''

    ### Figure - Partially Correct ###
    # Figure - Partially Correct, Omission
    figure4 = 'data/test-A/SPIQA_testA_Images/1703.07015v3/1703.07015v3-Figure2-1.png'
    figure4_caption = '''Figure 2: An overview of the Long- and Short-term Time-series network (LSTNet)'''
    figure4_question = '''What are the different types of layers in the LSTNet model and how are they connected?'''
    figure4_answer = '''The LSTNet model has four main types of layers: 1) Convolutional layer: extracts local dependency patterns from the input data. 2) Recurrent and recurrent-skip layer: capture long-term dependencies in the data. 3) Fully connected and element-wise sum output layer: combines the outputs from the convolutional and recurrent layers to produce the final prediction. 4) Autoregressive layer: provides a linear bypass to the non-linear neural network part of the model. The convolutional layer receives the input data and passes its output to the recurrent and recurrent-skip layers. These layers then pass their output to the fully connected and element-wise sum output layer. The autoregressive layer receives the input data directly and its output is also fed into the fully connected and element-wise sum output layer.'''

    # Figure - Partially Correct, Factual
    figure5 = 'data/test-A/SPIQA_testA_Images/1802.07351v2/1802.07351v2-Figure2-1.png'
    figure5_caption = '''Cost Volumes'''
    figure5_question = '''What is the difference between a standard cost volume and a deformable cost volume?'''
    figure5_answer = '''A standard cost volume computes the matching costs for a neighborhood of the same location on the feature maps of the first and second images. A deformable cost volume computes the matching costs for a dilated neighborhood of the same location on the feature maps of the first and second images, offset by a flow vector.'''

    # Figure - Partially Correct, Conceptual
    figure6 = 'data/test-A/SPIQA_testA_Images/1704.08615v2/1704.08615v2-Figure1-1.png'
    figure6_caption = '''No single saliency map can perform best in all metrics even when the true fixation distribution is known. This problem can be solved by separating saliency models from saliency maps. a) Fixations are distributed according to a ground truth fixation density p(x, y | I) for some stimulus I. b) This ground truth density predicts different saliency maps depending on the intended metric. c) Performances of the saliency maps from b) under seven saliency metrics on a large number of fixations sampled from the model distribution in a).'''
    figure6_question = '''What is the relationship between the ground truth fixation density and the saliency maps?'''
    figure6_answer = '''The ground truth fixation density predicts different saliency maps depending on the intended metric. The saliency maps differ dramatically due to the different properties of the metrics but always reflect the same underlying model. The predicted saliency map for the specific metric yields best performance in all cases.'''

    ### Table - Partially Correct ###
    # Table - Partially Correct, Omission
    table4 = 'data/test-A/SPIQA_testA_Images/1611.04684v1/1611.04684v1-Table4-1.png'
    table4_caption = '''Table 4: Evaluation results on response selection'''
    table4_question = '''Which model performs the best for response selection, and how can we tell?'''
    table4_answer = '''The KEHNN model performs the best for response selection. This is evident because it achieves the highest scores across all metrics (R2@1, R10@1, R10@2, and R10@5) compared to all other models in the table.'''

    # Table - Partially Correct, Factual
    table5 = 'data/test-A/SPIQA_testA_Images/1701.03077v10/1701.03077v10-Table2-1.png'
    table5_caption = '''Table 2. Results on unsupervised monocular depth estimation using the KITTI dataset, building upon the model from "Baseline". By replacing the per-pixel loss used by Baseline with several variants of our own per-wavelet general loss function in which our loss's shape parameters are fixed, annealed, or adaptive, we see a significant performance improvement. The top three techniques are colored red, orange, and yellow for each metric.'''
    table5_question = '''Which method for setting the shape parameter of the proposed loss function achieved the best performance in terms of average error? How much improvement did it offer compared to the reproduced baseline?'''
    table5_answer = '''The "adaptive alpha in (0, 2)" strategy, where each wavelet coefficient has its own shape parameter that is optimized during training, achieved the best performance in terms of average error. It reduced the average error by approximately 17% compared to the reproduced baseline.'''

    # Table - Partially Correct, Conceptual
    table6 = 'data/test-A/SPIQA_testA_Images/1704.05426v4/1704.05426v4-Table4-1.png'
    table6_caption = '''Table 4: Test set accuracies (%) for all models; Match. represents test set performance on the MultiNLI genres that are also represented in the training set, Mis. represents test set performance on the remaining ones; Most freq. is a trivial 'most frequent class' baseline.'''
    table6_question = '''How does the performance of the ESIM model differ when trained on MNLI alone versus trained on both MNLI and SNLI combined?'''
    table6_answer = '''When trained on MNLI alone, the ESIM model achieves an accuracy of 60.7% on SNLI, 72.3% on matched genres in MNLI, and 72.1% on mismatched genres in MNLI. However, when trained on both MNLI and SNLI combined, the ESIM model's performance improves across all tasks, reaching 79.7% accuracy on SNLI, 72.4% on matched MNLI genres, and 71.9% on mismatched MNLI genres.'''

    #######################################################################################
    # PARTIALLY CORRECT GENERATION CALLS
    #######################################################################################

    ### Plot - Partially Correct ###
    chart_partially_correct_omission = generate_seed_example(chart4, chart4_caption, chart4_question, chart4_answer, 'partially correct', 'omission', verdicts['partially correct'], error_categories['omission'])
    print('------------------------------------------------------------------------------')
    print('Seed Example (Plot - Partially Correct - Omission):\n')
    print(chart_partially_correct_omission)
    print('==============================================================================\n\n')

    chart_partially_correct_factual = generate_seed_example(chart5, chart5_caption, chart5_question, chart5_answer, 'partially correct', 'factual', verdicts['partially correct'], error_categories['factual'])
    print('------------------------------------------------------------------------------')
    print('Seed Example (Plot - Partially Correct - Factual):\n')
    print(chart_partially_correct_factual)
    print('==============================================================================\n\n')

    chart_partially_correct_conceptual = generate_seed_example(chart6, chart6_caption, chart6_question, chart6_answer, 'partially correct', 'conceptual', verdicts['partially correct'], error_categories['conceptual'])
    print('------------------------------------------------------------------------------')
    print('Seed Example (Plot - Partially Correct - Conceptual):\n')
    print(chart_partially_correct_conceptual)
    print('==============================================================================\n\n')

    ### Figure - Partially Correct ###
    figure_partially_correct_omission = generate_seed_example(figure4, figure4_caption, figure4_question, figure4_answer, 'partially correct', 'omission', verdicts['partially correct'], error_categories['omission'])
    print('------------------------------------------------------------------------------')
    print('Seed Example (Figure - Partially Correct - Omission):\n')
    print(figure_partially_correct_omission)
    print('==============================================================================\n\n')

    figure_partially_correct_factual = generate_seed_example(figure5, figure5_caption, figure5_question, figure5_answer, 'partially correct', 'factual', verdicts['partially correct'], error_categories['factual'])
    print('------------------------------------------------------------------------------')
    print('Seed Example (Figure - Partially Correct - Factual):\n')
    print(figure_partially_correct_factual)
    print('==============================================================================\n\n')

    figure_partially_correct_conceptual = generate_seed_example(figure6, figure6_caption, figure6_question, figure6_answer, 'partially correct', 'conceptual', verdicts['partially correct'], error_categories['conceptual'])
    print('------------------------------------------------------------------------------')
    print('Seed Example (Figure - Partially Correct - Conceptual):\n')
    print(figure_partially_correct_conceptual)
    print('==============================================================================\n\n')

    ### Table - Partially Correct ###
    table_partially_correct_omission = generate_seed_example(table4, table4_caption, table4_question, table4_answer, 'partially correct', 'omission', verdicts['partially correct'], error_categories['omission'])
    print('------------------------------------------------------------------------------')
    print('Seed Example (Table - Partially Correct - Omission):\n')
    print(table_partially_correct_omission)
    print('==============================================================================\n\n')

    table_partially_correct_factual = generate_seed_example(table5, table5_caption, table5_question, table5_answer, 'partially correct', 'factual', verdicts['partially correct'], error_categories['factual'])
    print('------------------------------------------------------------------------------')
    print('Seed Example (Table - Partially Correct - Factual):\n')
    print(table_partially_correct_factual)
    print('==============================================================================\n\n')

    table_partially_correct_conceptual = generate_seed_example(table6, table6_caption, table6_question, table6_answer, 'partially correct', 'conceptual', verdicts['partially correct'], error_categories['conceptual'])
    print('------------------------------------------------------------------------------')
    print('Seed Example (Table - Partially Correct - Conceptual):\n')
    print(table_partially_correct_conceptual)
    print('==============================================================================\n\n')
