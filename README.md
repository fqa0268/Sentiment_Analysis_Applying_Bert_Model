# Sentiment Analysis Applying Bert Model (TensorFlow)
## Introduction
This project is a demonstration of:
- pretrained large language model (LLM) application to perform sentiment analysis from a financial domain
- several skills of model improvement and manipulation

Dataset: HuggingFace financial_phrasebank is a collection of hand-annotated sentences with sentiment classified into Negative, Neutral, Positive
- Subdataset: sentences_50agree, sentences_66agree, sentences_75agree, sentences_allagree (The hand-annotated datasets have 50%/66%/75%/100% agreement rate by 5-8 annotators); this project will mainly use the sentences_allagree subdataset.
Notes: LLM traing process will consume a large amount of computing resources and memory. I recommend to run this program using Google Colab Pro, or other high-performance computing resources.

## Table of Contents
1. Pre-Training Model + Supervised Fine-Tuning
2. Have a Look at Data
3. Create Train/Validation/Test Sets
4. Define Performance Measure
5. Exploratory Data Analysis
6. Prepare the Data
7. Pre-Trained Model + Supervised Fine-Tuning
  - Fine-Tune Only Classifier Head
  - Fine-Tune All Weights
8. Error Analysis
9. Additional: Create and Fit Model With TensorFlow Dataset(TFDS)
10. Additional: Create Customized Classification Head
11. Additional: In-Context Learning
