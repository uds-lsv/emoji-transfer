# emoji-transfer

This is the repository for Emoji-Based Transfer Learning for Sentiment Tasks.
https://arxiv.org/abs/2102.06423

## Requirements

- Install pytorch (https://pytorch.org/)
- Install the simple transformers library(https://pypi.org/project/simpletransformers/): 
pip install simpletransformers
- Install scikit learn (https://scikit-learn.org/stable/install.html)

## DATA

Datasets to train an LM on emoji (cluster) prediction can be found below.

https://repos.lsv.uni-saarland.de/sboy/emoji-transfer-data

## train_pretraining

Train a pretrained LM on emoji prediction or emoji cluster prediction

python train_pretraining.py <model architecture> <model name or path> <path to train/test data> <args>

- model architecture: "bert" or "xlmroberta"
- model name or path: bert-base-german-cased for German BERT (monolingual), bert-base-multilingual cased or xlm-roberta-base
- path to the dataset: must contain a classes.txt with the labels, a train, validation and test set in tsv format
- args: json file with model parameters (please refer to the example_args.json)

## train_downstream

Train an emoji (cluster) predictor on a downstream task 

python train_downstream.py <model architecture> <model name or path> <path to train/test data> <args>

## train_bert.delta

Train an emoji prediction or emoji cluster prediction 10 times on a downstream task and evaluate ACC and macro F1.

python train_bert.delta.py germeval/train.tsv germeval/val.tsv germeval/test.tsv bert-emoji-prediction 2 bert-emoji-prediction-on-downstream-task

## Contact
If you have questions feel free to contact me: susannb@coli.uni-saarland.de
