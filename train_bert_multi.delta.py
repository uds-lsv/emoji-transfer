import sys
#sys.path.append('/nethome/druiter/code/simpletransformers/')
from simpletransformers.classification import (
    ClassificationModel)
import pandas as pd
import logging
import math
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import os
import statistics
import string
import json
import math
from pathlib import Path

print("BERT sat pmi")
# Input
train = sys.argv[1]
valid = sys.argv[2]
test_path = Path(sys.argv[3])
output_dir = sys.argv[4]
num_labels = int(sys.argv[5])
model_args = dict()
model_args["reprocess_input_data"] = True
model_args["num_train_epochs"] = 10
model_args["evaluate_during_training"] = True
model_args["overwrite_output_dir"] = True
model_args["fp16"] = False
model_args["use_early_stopping"] = True
model_args["early_stopping_delta"] = 0.01
model_args["early_stopping_metric"] = 'mcc'
model_args["early_stopping_metric_minimize"] = False
model_args["early_stopping_patience"] = 3 # 10
model_args["evaluate_during_training_steps"] = 1000
model_args["save_eval_checkpoints"] = False
model_args["use_cuda"] = True

if len(sys.argv) > 6:
    model_dir = sys.argv[6]
else:
    model_dir = False


def f1_score_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def f1_score_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

# Set loggers
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

# Get Data
train_data = pd.read_csv(train, sep='\t', header=None)
train_df = pd.DataFrame(train_data)
train_df.dropna(subset=[0], inplace=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)

valid_data = pd.read_csv(valid, sep='\t', header=None)
valid_df = pd.DataFrame(valid_data)
valid_df.dropna(subset=[0], inplace=True)
valid_df = valid_df.sample(frac=1).reset_index(drop=True)

test_data_en = pd.read_csv(test_path / "test_en.tsv", sep='\t', header=None)
test_df_en = pd.DataFrame(test_data_en)
test_df_en.dropna(subset=[0], inplace=True)
test_df_en = test_df_en.sample(frac=1).reset_index(drop=True)

test_data_ar = pd.read_csv(test_path / "test_ar.tsv", sep='\t', header=None)
test_df_ar = pd.DataFrame(test_data_ar)
test_df_ar.dropna(subset=[0], inplace=True)
test_df_ar = test_df_ar.sample(frac=1).reset_index(drop=True)

accs_en = []
f1s_en = []
aucs_en = []
accs_ar = []
f1s_ar = []
aucs_ar = []
for i in range(0,10):
    # Prepare cross validation
    # Setup
    cur_output = '{}/{}/'.format(output_dir, i)
    if not os.path.exists(cur_output):
        os.mkdir(cur_output)
    model_args["output_dir"] = '{}/output'.format(cur_output)
    model_args["cache_dir"] = '{}/cache/'.format(cur_output)
    model_args["tensorboard_dir"] = '{}/runs/'.format(cur_output)
    model_args["best_model_dir"] = '{}/output'.format(cur_output)
        
    if model_dir:
        model_pretrained = ClassificationModel(
            "bert",
            model_dir,
            use_cuda=True,
        )
        model_downstream =  ClassificationModel(
            "bert",
            "bert-base-multilingual-cased",
            args=model_args,
            num_labels=num_labels
        )
        model_downstream.model.bert = model_pretrained.model.bert
        model = model_downstream
        

    else:
        model = ClassificationModel(
            "bert",
            "bert-base-multilingual-cased",
            args=model_args,
            num_labels=num_labels
        )

    # Train model
    model.train_model(train_df, eval_df=valid_df)

    # Validate model
    result, model_outputs, wrong_predictions = model.eval_model(test_df_en,
                                                                cm=confusion_matrix,
                                                               acc=accuracy_score,
                                                               f1_macro=f1_score_macro,
                                                               f1_micro=f1_score_micro)
                                                               #auc=roc_auc_score)
    print(result, flush=True)

    accs_en.append(result['acc'])
    f1s_en.append(result['f1_macro'])
    #aucs_en.append(result['auc'])

    result, model_outputs, wrong_predictions = model.eval_model(test_df_ar,
                                                                cm=confusion_matrix,
                                                               acc=accuracy_score,
                                                               f1_macro=f1_score_macro,
                                                               f1_micro=f1_score_micro)
                                                               #auc=roc_auc_score)
    print(result, flush=True)

    accs_ar.append(result['acc'])
    f1s_ar.append(result['f1_macro'])
    #aucs_ar.append(result['auc'])

def avg(lst):
    return sum(lst) / len(lst)
 
acc_stdev_en = statistics.stdev(accs_en)
acc_avg_en = avg(accs_en)
acc_se_en = acc_stdev_en/math.sqrt(10)

f1_stdev_en = statistics.stdev(f1s_en)
f1_avg_en = avg(f1s_en)
f1_se_en = f1_stdev_en/math.sqrt(10)

#auc_stdev_en = statistics.stdev(aucs_en)
#auc_avg_en = avg(aucs_en)
#auc_se_en = auc_stdev_en/math.sqrt(10)

print("Results for: EN")
print('Acc... Mean: {}\tStandard Deviaton: {}\tStandard Error: {}'.format(acc_avg_en, acc_stdev_en, acc_se_en))
print('F1 Macro... Mean: {}\tStandard Deviaton: {}\tStandard Error: {}'.format(f1_avg_en, f1_stdev_en, f1_se_en))
#print('AUC... Mean: {}\tStandard Deviaton: {}\tStandard Error: {}'.format(auc_avg_en, auc_stdev_en, auc_se_en))

acc_stdev_ar = statistics.stdev(accs_ar)
acc_avg_ar = avg(accs_ar)
acc_se_ar = acc_stdev_ar/math.sqrt(10)

f1_stdev_ar = statistics.stdev(f1s_ar)
f1_avg_ar = avg(f1s_ar)
f1_se_ar = f1_stdev_ar/math.sqrt(10)

#auc_stdev_ar = statistics.stdev(aucs_ar)
#auc_avg_ar = avg(aucs_ar)
#auc_se_ar = auc_stdev_ar/math.sqrt(10)

print("Results for: AR")
print('Acc... Mean: {}\tStandard Deviaton: {}\tStandard Error: {}'.format(acc_avg_ar, acc_stdev_ar, acc_se_ar))
print('F1 Macro... Mean: {}\tStandard Deviaton: {}\tStandard Error: {}'.format(f1_avg_ar, f1_stdev_ar, f1_se_ar))
#print('AUC... Mean: {}\tStandard Deviaton: {}\tStandard Error: {}'.format(auc_avg_ar, auc_stdev_ar, auc_se_ar))
