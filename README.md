# Pretraining InLegalBERT

This repository contains the codes for pre-training a BERT-base model on a large, un-annotated corpus of text using dynamic Masked Language Modeling (MLM) and dynamic Next Sentence Prediction (NSP).
All settings in this repository are configured for replicating the pre-training procedure of [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT).
For details please refer to our paper ["Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law"](https://arxiv.org/abs/2209.06049).

## In this Repository

<var>data_helpers.py</var>: Custom data loader and chunking operations.
<var>model.py</var>: Implementation of BertForPreTraining.
<var>training.py</var>: Custom trainer and metrics.
<var>run.py</var>: Main code for running the model.
<var>sample.jsonl</var>: Sample document for the dataset format.

## Dataset Format

There must be two files "train.jsonl" and "test.jsonl" in the main directory.
Each line in these files must contain a json dictionary, similar to "sample.jsonl".
The keys of the dict are:
```
  id: String          // identifier
  title: String       // title of the case
  source: String      // court where the case was heard
  text: List[String]  // main document text kept as a list of sentences
```
This format expects the 'text' field to be divided into a List of sentences. 
In case your data is not pre-divided, and the text is a single string, wrap that single string in a List.
You may use a different dataset schema by modifying run.py:39.

## Settings and Hyperparameters
The main settings are available from run.py:16.
```
  SOURCE_PATH: Source model/tokenizer to start with. Can be a folder with the relevant files or a repo in HuggingFace.
  OUTPUT_PATH: All relevant outputs will be stored in "Output/<OUTPUT_PATH>"
  CACHE_PATH: Path to the cache directory
  FROM_SCRATCH(True/False): Whether to train from scratch or using the existing checkpoint at "<SOURCE_PATH>".
```

Other hyperparameters are available from run.py:91.

## Running the code
Setup the parameters and hyperparameters, and all relevant files.
```
  python run.py
```

## Requirements
```
  python=3.9.7
  torch=1.10.2
  transformers=4.17.0
  datasets=2.4.0
  pyarrow=7.0.0
  sklearn=1.0.2
  tqdm=4.64.0
```
