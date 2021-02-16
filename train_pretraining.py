import pandas as pd
import logging
from simpletransformers.classification import ClassificationModel
import sys
from pathlib import Path
import sklearn
import json
import torch

def main():
    if len(sys.argv) != 5:
        print('Usage: python train_pretraining.py <model architecture> <model name or path> <data_path> <args>', file=sys.stderr)
        sys.exit(1)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    model_type = sys.argv[1]
    model_path = sys.argv[2]
    data_directory = Path(sys.argv[3])
    args = json.load(open(sys.argv[4], "r"))
    train_path = data_directory / "train.tsv"
    test_path = data_directory / "test.tsv"
    val_path = data_directory / "val.tsv"
    classes = data_directory / "classes.txt"
    with open(classes, encoding="utf-8") as f:
        num_labels = sum(1 for line in f if line.rstrip())
    # Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
    train_data = pd.read_csv(train_path, sep="\t", header=None)
    train_df = pd.DataFrame(train_data)
    train_df.dropna(subset=[0], inplace=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    logger.info(dict(train_df.dtypes))
    eval_data = pd.read_csv(test_path, sep="\t", header=None)
    eval_df = pd.DataFrame(eval_data)
    eval_df.dropna(subset=[0], inplace=True)
    val_data = pd.read_csv(val_path, sep="\t", header=None)
    val_df = pd.DataFrame(val_data)
    val_df.dropna(subset=[0], inplace=True)
    # Create a ClassificationModel
    model = ClassificationModel(model_type, model_path, num_labels=num_labels, use_cuda=True, args=args)
    logger.info("This is a")
    logger.info(model_type)
    logger.info("model from:")
    logger.info(model_path)
    logger.info(args)
    logger.info("The train, test and val data was obtained from:")
    logger.info(data_directory)
    logger.info("Total number of labels:")
    logger.info(num_labels)
    logger.info("Start training now.")
    # Train the model
    model.train_model(train_df, eval_df=val_df, acc=sklearn.metrics.accuracy_score, f1=sklearn.metrics.f1_score)
    logger.info("Training concluded. Start with evaluation...")
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score, f1=sklearn.metrics.f1_score)
    logger.info(result)


if __name__ == '__main__':
    main()
