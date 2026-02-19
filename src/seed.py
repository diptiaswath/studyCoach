import base64
from openai import OpenAI

client = OpenAI()

def to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # change mime type if needed: image/png, image/webp, etc.
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
  Do you best to come up with a common potential mistake a student might make when reading above data. 
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
    
    ### Charts ###
    chart1 = 'data/test-A/SPIQA_testA_Images/1702.08694v3/1702.08694v3-Figure3-1.png' # Incorrect, Omission 
    chart1_caption = \
'''Figure 3: Results on real data. Regarding the scale of precision and F-measure, see the comment at the last paragraph just before Section 3. 
The y-axis is in logarithmic scale. C-Tarone is shown in red and the binarization approach is shown in blue. Higher (taller) is better in precision, i
recall, and F-measure, while lower is better in running time.'''
    chart1_question = \
'''How does the C-Tarone method compare to the binarization method in terms of precision, recall, F-measure, and running time?'''
    chart1_answer = \
'''The C-Tarone method has higher precision and F-measure than the binarization method in all datasets. The C-Tarone method has better or competitive recall 
than the binarization method. The running time of the C-Tarone method is competitive with the binarization method.
'''

    chart2 = 'data/test-A/SPIQA_testA_Images/1606.07384v2/1606.07384v2-Figure1-1.png' # Incorrect, Factual 
    chart2_caption = \
'''Experiments with synthetic data: error is reported against the size of the conditional probability table (lower is better). 
The error is the estimated total variation distance to the ground truth Bayes net. We use the error of MLE without noise as our benchmark. 
We plot the performance of our algorithm (Filtering), empirical mean with noise (MLE), and RANSAC. 
We report two settings: the underlying structure of the Bayes net is a random tree (left) or a random graph (right).'''
    chart2_question = 'How does the performance of the Filtering algorithm compare to the performance of MLE with noise?'
    chart2_answer = \
'''The Filtering algorithm performs better than MLE with noise in both the random tree and random graph settings. 
The figure shows that the error of the Filtering algorithm is lower than the error of MLE with noise for all values of the number of parameters. 
This is true for both the random tree and random graph settings.'''

    chart3 = 'data/test-A/SPIQA_testA_Images/1704.07854v4/1704.07854v4-Figure15-1.png' # Incorrect, Conceptual 
    chart3_caption = \
'''Training with different gradient approximations: validation loss with a simplified advection (red), and the correct gradient from forward advection (green). The simplified version does not converge.'''
    chart3_question = \
'''Which gradient approximation method trains better?'''
    chart3_answer = \
'''The corrected gradient method leads to a more stable and lower loss value, so leads to a better training. Simplified advection method doesn't converge.'''

    chart4 = 'data/test-A/SPIQA_testA_Images/1706.08146v3/1706.08146v3-Figure2-1.png' # Partially Correct, Omission
    chart4_caption = \
'''Approximation errors Err(X,X∗) := ‖X −X∗‖F /‖X∗‖F for sparse PCA and NMF on synthetic data with varying column sparsity k of W and projection dimension d. 
The values of d correspond to 10×, 5×, and 2.5× compression respectively. Err(W̃ , PW ) measures the distance between factors in the compressed domain: low error here is necessary for accurate sparse recovery. 
Err(Ŵ ,W ) measures the error after sparse recovery: the recovered factors Ŵ typically incur only slightly higher error than the oracle lower bound (dotted lines) where PW is known exactly.'''
    chart4_question= \
'''What is the effect of increasing the projection dimension d on the approximation error for sparse PCA and NMF?'''
    chart4_answer = \
'''Increasing the projection dimension d decreases the approximation error for both sparse PCA and NMF. he figure shows that the approximation error decreases as the projection dimension d increases. 
This is because a higher projection dimension allows for a more accurate representation of the original data.'''

    chart5 = 'data/test-A/SPIQA_testA_Images/1606.07384v2/1606.07384v2-Figure1-1.png' # Partially Correct, Factual 
    chart6 = 'test-A/SPIQA_testA_Images/1606.07384v2/1606.07384v2-Figure1-1.png' # Partially Correct, Conceptual 

    chart_incorrect_omission = generate_seed_example(chart1, chart1_caption, chart1_question, chart1_answer, 'incorrect', 'omission', verdicts['incorrect'], error_categories['omission'])
    print('------------------------------------------------------------------------------')
    print('Seed Example:\n')
    print(chart_incorrect_omission)
    print('==============================================================================\n\n')


    # chart_incorrect_factual = generate_seed_example(chart2, chart2_caption, chart2_question, chart2_answer, 'incorrect', 'factual', verdicts['incorrect'], error_categories['factual'])
    # print('------------------------------------------------------------------------------')
    # print('Seed Example:\n')
    # print(chart_incorrect_factual)
    # print('==============================================================================\n\n')

    # chart_incorrect_conceptual = generate_seed_example(chart3, chart3_caption, chart3_question, chart3_answer, 'incorrect', 'conceptual', verdicts['incorrect'], error_categories['conceptual'])
    # print('------------------------------------------------------------------------------')
    # print('Seed Example:\n')
    # print(chart_incorrect_conceptual)
    # print('==============================================================================\n\n')

    # chart_partially_correct_omission = generate_seed_example(chart4, chart4_caption, chart4_question, chart4_answer, 'partially correct', 'omission', verdicts['partially correct'], error_categories['omission'])
    # print('------------------------------------------------------------------------------')
    # print('Seed Example:\n')
    # print(chart_partially_correct_omission)
    # print('==============================================================================\n\n')


    ### Figures ###
    figure1 = 'data/test-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Figure1-1.png'  # Incorrect, Omission 
    figure2 = 'data/test-A/SPIQA_testA_Images/1704.07854v4/1704.07854v4-Figure12-1.png' # Incorrect, Factual 
    figure3 = 'data/test-A/SPIQA_testA_Images/1704.07854v4/1704.07854v4-Figure12-1.png' # Incorrect, Conceptual
    figure4 = 'data/test-A/SPIQA_testA_Images/1704.07854v4/1704.07854v4-Figure12-1.png' # Partially Correct, Omission 
    figure5 = 'data/test-A/SPIQA_testA_Images/1704.07854v4/1704.07854v4-Figure12-1.png' # Partially Correct, Factual 
    figure6 = 'test-A/SPIQA_testA_Images/1704.07854v4/1704.07854v4-Figure12-1.png' # Partiallly Correct, Conceptual 

    ### Tables ###
    table1 = 'data/test-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Table1-1.png' # Incorrect, Omission
    table2 = 'datatest-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Table1-1.png' # Incorrect, Factual 
    table3 = 'data/test-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Table1-1.png' # Incorrect, Conceptual 
    table4 = 'data/test-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Table1-1.png' # Partially Correct, Omission
    table5 = 'data/test-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Table1-1.png' # Partially Correct, Factual 
    table6 = 'data/test-A/SPIQA_testA_Images/1703.04887v4/1703.04887v4-Table1-1.png' # Partially Correct, Conceptual 
    
    