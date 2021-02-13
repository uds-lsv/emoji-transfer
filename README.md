# emoji-transfer

UNDER CONSTRUCTION

## train_pretraining

Train a pretrained LM on emoji prediction or emoji cluster prediction

python train_pretraining.py <model architecture> <model name or path> <path to train/test data> <args>

## train_downstream

Train an emoji prediction or emoji cluster prediction  on a downstream task 

 python train_downstream.py <model architecture> <model name or path> <path to train/test data> <args>

## train_bert.delta

Train an emoji prediction or emoji cluster prediction 10 times on a downstream task and evaluate ACC and macro F1.

python train_bert.delta.py germeval/train.tsv germeval/val.tsv germeval/test.tsv bert-emoji-prediction 2 bert-emoji-prediction-on-downstream-task

## DATA

Datasets to train an LM on emoji (cluster) prediction can be found below

https://repos.lsv.uni-saarland.de/sboy/emoji-transfer-data

