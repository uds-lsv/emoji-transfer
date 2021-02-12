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

print("BERT germeval swear")
# Input
train = sys.argv[1]
valid = sys.argv[2]
test = sys.argv[3]
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

test_data = pd.read_csv(test, sep='\t', header=None)
test_df = pd.DataFrame(test_data)
test_df.dropna(subset=[0], inplace=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

accs = []
f1s = []
aucs = []
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
    result, model_outputs, wrong_predictions = model.eval_model(test_df,
                                                                cm=confusion_matrix,
                                                               acc=accuracy_score,
                                                               f1_macro=f1_score_macro,
                                                               f1_micro=f1_score_micro,
                                                               auc=roc_auc_score)
    print(result, flush=True)

    accs.append(result['acc'])
    f1s.append(result['f1_macro'])
    aucs.append(result['auc'])

def avg(lst):
    return sum(lst) / len(lst)
 
acc_stdev = statistics.stdev(accs)
acc_avg = avg(accs)
acc_se = acc_stdev/math.sqrt(10)

f1_stdev = statistics.stdev(f1s)
f1_avg = avg(f1s)
f1_se = f1_stdev/math.sqrt(10)

auc_stdev = statistics.stdev(aucs)
auc_avg = avg(aucs)
auc_se = auc_stdev/math.sqrt(10)

print('Acc... Mean: {}\tStandard Deviaton: {}\tStandard Error: {}'.format(acc_avg, acc_stdev, acc_se))
print('F1 Macro... Mean: {}\tStandard Deviaton: {}\tStandard Error: {}'.format(f1_avg, f1_stdev, f1_se))
print('AUC... Mean: {}\tStandard Deviaton: {}\tStandard Error: {}'.format(auc_avg, auc_stdev, auc_se))
