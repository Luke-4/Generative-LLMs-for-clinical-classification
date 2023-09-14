# Generative-LLMs-for-clinical-classification

## Project Overview

This work investigates the application of medical domain-specific Generative Large
Language Models (LLMs) as feature extractors for intent classification in medical letters. The central
focus is to leverage the strengths of these models and compare their performance with discriminative
LLMs on the same dataset.

Three medical domain-specific generative LLMs, namely BioGPT, PMC_LLaMA_7B, and
BioMedLM, are evaluated for their capabilities in generating context-rich embeddings. These
embeddings are utilized in a two-stage experiment. In the first stage, embeddings are directly extracted
from the base models without fine-tuning, and Support Vector Machines (SVMs) are employed for
classification. In the second stage, the models are used for classification with a linear head, both before
and after fine-tuning using LoRA.


## Rationale for Using Large Language Models
Large language models, such as BERT (Bidirectional Encoder Representations from Transformers),
GPT-3 (Generative Pre-trained Transformer 3), and their variants, have demonstrated remarkable
success in a wide range of natural language processing tasks. The use of large language models in text
classification offers several compelling reasons:
  - Contextual Understanding: Large language models leverage deep learning techniques to encode
contextual information and relationships between words in a sentence. This contextual understanding
allows them to capture subtle nuances and semantics, which is especially relevant in the medical domain
where precise interpretation of clinical text is vital.
  - Transfer Learning: Pre-training on vast corpora of textual data enables large language models to learn
general language patterns. This pre-trained knowledge can be fine-tuned on domain-specific datasets,
making them adaptable and effective for text classification tasks in the medical field with relatively
limited labelled data.

<hr>

### Data Preprocessing

The first step is to encode the texts into feature vectors. The following steps have been taken:

1. **Encoding the texts:** 
   - Extract the embeddings from all the models before and after fine-tuning

2. **Fine tunning**
   - Low Rank Adaptation (LoRA) 

3. **Oversampling:** 
   - Synthetic Minority Over-sampling Technique (SMOTE)

### Classification Models

The two stages of the experiment employ diferent classification techniques, first the embeddings are extracted and classified using a SVC before and after fine tunning. In the second stage the models are classified directly by the LLMs by aplying a liniar head on top of the model. Here are the models implemented:

1. **Support Vector Classifier**

2. **Model for sequence classification**
   - To do this AutoModelForSequenceClassification model class was used, this aplies a linear head on top of the base model to perform the classification directly. 


### Model Evaluation

To assess the performance of the models, the following evaluation metrics have been utilized:

1. **Precision**

2. **Recall**

3. **Macro F1-Score**

TO DO
<hr></hr>

## Files

1. Read_data.R
   - Read original data
   - Outlier Detection
   - Discretization
   - Write to CSV to use in python
2. Preprocessing.R
   - Check for null values
   - Plot the correlation of each feature with the target variable
3. Resampling_and_RFE.ipynb
   - Resampling
   - RFECV Recursive feature elimination with cross-validation
   - Write to CSV to use in R
4. read_new_data.R
   - Read resampled data 
   - Drop discarted variables
5. PC_and_HC_for_BN.R
   - PC algorithm to create dag
   - Hill climbing algorithm to create dag 
6. BN_final.R
   - Create two manual dags, dag 3 and 4 in report 
   - Plot the dags
   - Fit the models on the training data
   - Predict on the testing set
   - Evaluate the predictions
7. RandomForest.R
   - Fit the Random Forest model on the training set
   - Predict on the test set 
   - Evaluate the results

