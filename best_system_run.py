import pandas as pd
import gc, argparser
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score
from transformers import EvalPrediction
import torch
import numpy as np

from utils import compute_metrics, get_dds, labs


test = pd.read_csv("arguments-test.tsv", delimiter="\t")
f = pd.DataFrame(test["Argument ID"].copy())
for col in labs:
    f[col] = 0
f.to_csv("labels-test.tsv", index=None, sep="\t")

# model_name='roberta-base'
# model_name = "microsoft/deberta-v2-xxlarge"
parser = argparse.ArgumentParser(description='Simple args')
parser.add_argument('-m', '--model-name', required=True, default="microsoft/deberta-v3-small")
args = parser.parse_args()
model_name = args.model_name
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dds = get_dds(tokenizer=tokenizer)

bs = 16 // 4
epochs = 6
lr = 2e-5 / 4
gc.collect()
torch.cuda.empty_cache()
transformers.set_seed(1234)

args = TrainingArguments(
    "outputs_xxlarge_v3_more_training_data",
    learning_rate=lr,
    save_strategy="epoch",
    # warmup_ratio=0.1, lr_scheduler_type='cosine',
    load_best_model_at_end=False,
    fp16=False,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    metric_for_best_model="f1_macro_0.25",
    save_total_limit=1,
    report_to="none",
)
print(args)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(labs), problem_type="multi_label_classification"
)
model.resize_token_embeddings(len(tokenizer))

trainer = Trainer(
    model,
    args,
    train_dataset=dds["train"],
    eval_dataset=dds["valid"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
# Threshold analysis
sigmoid = torch.nn.Sigmoid()
valid_preds = trainer.predict(dds["valid"])
probs = sigmoid(torch.Tensor(valid_preds.predictions)).numpy()

y_true = valid_preds.label_ids
scores = []
ths = np.arange(0.05, 0.6, 0.05)
for threshold in ths:
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    scores.append(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))
print(list(zip(ths, scores)))

# Test preds population
test_probs = torch.nn.Sigmoid()(
    torch.Tensor(trainer.predict(dds["test"]).predictions)
).numpy()
test_probs
THRESHOLD = 0.20
y_pred = np.zeros(test_probs.shape)
y_pred[np.where(test_probs >= THRESHOLD)] = 1

for idx, val in enumerate(f.columns[1:]):
    f[val] = y_pred[:, idx].astype("int")
f.to_csv("run_A_v1.csv", index=None, sep="\t")
