# Introduction
A major challenge in education is the task of producing quiz questions for students. Quiz questions are one of the most important tools for learning outcomes and knowledge retention, but are time-consuming and effort-laden to produce. 

The transformer model is a deep learning model introduced in 2017 that solved some of the issues of its predecessors by capturing long-term dependencies without using recursion. The Text-to-Text Transfer Transformer (T5) is a recent transformer model that has not received much attention, but has achieved state of the art performance in most natural language processing tasks. 

This is a repository for a transformer architecture project using a T5 model fine-tuned on the SQuAD dataset for the downstream task of question generation.  It will be applied to the task of quiz question generation in education and pedagogy.  

To begin, install the Transformers library from HuggingFace (for the T5 model and the SQuAD dataset), and the NLTK library (for the sentence tokenizer).

# Data Preparation
I will be using the Stanford Question Answering Dataset (SQuAD) to fine-tune the model. SQuAD is the most canonical dataset for question answering and generation tasks. There are two versions, SQuAD 1.1, and SQuAD 2.0. The first version is a crowdsourced set of 100,000+ questions and answers about various Wikipedia articles. Version 2.0 builds upon this by also including 50,000 questions that look similar to those in the dataset, but actually cannot be answered by the information in the passage. This dataset ensures that models can also handle situations in which there is no answer supported by the paragraph, and know when not to answer.

My exploratory data analysis shows a dataset split that consists of 129,941 training examples, 6,078 dev examples, and 5,915 test examples. By loading and subsetting the SQuAD dataset, we can see that it is a JSON file containing triples, which are combinations of questions, passages, and answers. Each answer is a segment, or span, of the corresponding passage.

Furthermore, it shows that the dataset is split into paragraphs, where each paragraph represents a Wikipedia article, and there are one or more questions about each paragraph. A paragraph is composed of a title, context, and a question and answer. For instance, we can see that the structure of a paragraph looks like the following.

```Prepare_data.py```
* Imports dataset from nlp
* Takes in arguments
* Saves the special tokens in the tokenizer, processes and caches the dataset according to the arguments
* Outputs and saves data in data/squad_multitask

If the model needs further fine-tuning, I may elect to assemble a more domain-specific question answering dataset. I would collect at least a couple thousand examples from a humanities, or history-specific domain. Further data collection can be done using the Wikipedia API in Python, or the Wikidata API with SPARQL queries to access the endpoint. Another alternative is to scrape European history flashcards on Anki for questions and answers.

# Modeling
I will be using a transfer learning approach, in which I use a model pre-trained with massive unsupervised data sets, before fine-tuning it downstream on a more domain-specific dataset. In this approach, I will pick a transformer model as well as a question-answer dataset with which to fine-tune the model.

I will use a transformer architecture, most likely either a T5 or GPT-2 model. The transformer architecture is at the cutting-edge for many NLP tasks, having superseded RNNs/LSTMs and CNNs, because it can perform better at lower training times and cost​.  RNN/LSTMs have the ability to memorize but not train in parallel, while CNNs have the opposite ability. Transformers are powerful because they combine both the ability to parallelize and memorize. Specifically, they do this by capturing long-term dependencies without using recursion, by processing sentences as a whole instead of word by word (i.e. non-sequential processing). They also capture the relationships between words with multi-head attention mechanisms and positional embeddings​. 

```Run_qg.py```
* Imports data_collator.py, utils.py, trainer.py
* Trains the model

```Eval.py```
* Validates and evaluates the model

If the model needs further improvement, then after fine-tuning the pre-trained
transformer model on the SQuAD data so that the model learns the task of question generation, I will apply transfer learning a second time on a more domain-specific dataset so that it can generate more humanities or history-specific questions. While there are other ways of teaching a model to generate domain-specific questions, this is the simplest and most intuitive.

# System Architecture

```Pipelines.py```
* This calls the entire pipeline and runs the cached model

I implement this data pipeline through notebooks, and then split it up into separate scripts. At a high level, the scripts will handle 1) data preprocessing by downloading and building the SQuAD dataset and vocabularies, 2) defining the model architecture, 3) training the model, and 4) evaluating the model by using it to generate questions on previously unseen data. Other steps include handling configuration and hyperparameters, and defining the various layers to be used by the model.

The first step towards deploying the question generation model would be to create a demo as a proof of concept. To do that, I would use the Hugging Face Inference API via Model Hub to upload and share the model. After that, I would build towards deploying an API that users can interact with over a web interface. My first choice would be to use a hosted model deployment platform, such as AWS SageMaker, or GCP AI predictions. However, if I have time, I would like to use my own model server, such as Flask, and deploy it on a serving platform like AWS Elastic Beanstalk or Heroku.