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

Before diving into machine learning models, it's essential to preprocess the data to ensure its quality and suitability for analysis. The following steps have been taken:

1. **Resampling Techniques:** 
   - Repeated Edited Nearest Neighbours (Undersampling)
   - Random Over Sampler (Oversampling)

2. **Standardization**

3. **Feature Selection:** 
   - Recursive Feature Elimination

### Machine Learning Models

Here are the models implemented:

1. **Decision Tree Classifier**

2. **Logistic Regression**

3. **Support Vector Machine**

### Model Evaluation

To assess the performance of the machine learning models, the following evaluation metrics have been utilized:

1. **Precision**

2. **Recall**

3. **F1-Score**

### Hyperparameter Tuning

Fine-tuning the model parameters is crucial for achieving optimal performance.

1. **Grid Search**
