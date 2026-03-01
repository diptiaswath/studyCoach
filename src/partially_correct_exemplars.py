#######################################################################################
# PARTIALLY CORRECT EXEMPLARS
# Variable definitions and generation calls for chart4-6, figure4-6, table4-6
# Validated: 9/9 exemplars across Plot/Figure/Table x Omission/Factual/Conceptual
#######################################################################################
from 
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
