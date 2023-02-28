import pandas as pd
import gc
import transformers
from transformers import TrainingArguments,Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset,DatasetDict
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,precision_score
from transformers import EvalPrediction
import torch
import numpy as np
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_true = labels
    metrics = {}
    # next, use threshold to turn them into integer predictions
    for threshold in [0.1, 0.15, 0.20, 0.25, 0.3, 0.5]:
      y_pred = np.zeros(probs.shape)
      y_pred[np.where(probs >= threshold)] = 1
      metrics[f'f1_macro_{threshold}'] = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


labs = ['Self-direction: thought', 'Self-direction: action',
       'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance',
       'Power: resources', 'Face', 'Security: personal', 'Security: societal',
       'Tradition', 'Conformity: rules', 'Conformity: interpersonal',
       'Humility', 'Benevolence: caring', 'Benevolence: dependability',
       'Universalism: concern', 'Universalism: nature',
       'Universalism: tolerance', 'Universalism: objectivity']


test = pd.read_csv("arguments-test.tsv", delimiter="\t")
f = pd.DataFrame(test['Argument ID'].copy())
for col in labs: f[col] = 0
f.to_csv("labels-test.tsv", index=None, sep='\t')


model_name = "microsoft/deberta-v3-large"
#model_name='roberta-large'
model_name ='microsoft/deberta-v2-xxlarge'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tok = AutoTokenizer.from_pretrained(model_name)

def tok_func(x): return tok(x["input"])
cols_to_remove = ('Argument ID', 'Conclusion', 'Stance', 'Premise',  'StanceUpdated')


def get_ds(fold):
  if fold == 'training':
    dfs = []
    for f in ['training', 'validation']:
      df = pd.read_csv(f"./arguments-{f}.tsv", delimiter="\t")
      labels = pd.read_csv(f"labels-{f}.tsv", delimiter="\t")
      labels['labels'] = labels.apply(lambda l: [float(l[lab]) for lab in labs], axis=1)
      df = df.merge(labels[['Argument ID', 'labels']], on=['Argument ID'])
      print(df.shape)
      dfs.append(df)
    df = pd.concat([dfs[0], dfs[1]])
    df['StanceUpdated'] = df['Stance'].apply(lambda l: 'against' if 'against' in l else "favour" )
    df['sectok'] = '[' + df.StanceUpdated + ']'
    sectoks = list(df.sectok.unique())
    tok.add_special_tokens({'additional_special_tokens': sectoks})
  if fold =='validation':
    df = pd.read_csv("arguments-validation-zhihu.tsv", delimiter='\t')
    l = pd.read_csv("labels-validation-zhihu.tsv", delimiter='\t')
    l['labels'] = l.apply(lambda l: [float(l[lab]) for lab in labs], axis=1)
    df = df.merge(l[['Argument ID', 'labels']], on=['Argument ID'])

    df['StanceUpdated'] = df['Stance'].apply(lambda l: 'against' if 'against' in l else "favour" )
    df['sectok'] = '[' + df.StanceUpdated + ']'
  if fold == 'test':
    df = pd.read_csv(f"./arguments-{fold}.tsv", delimiter="\t")
    labels = pd.read_csv(f"labels-{fold}.tsv", delimiter="\t")
    labels['labels'] = labels.apply(lambda l: [float(l[lab]) for lab in labs], axis=1)
    df = df.merge(labels[['Argument ID', 'labels']], on=['Argument ID'])
    df['StanceUpdated'] = df['Stance'].apply(lambda l: 'against' if 'against' in l else "favour" )
    df['sectok'] = '[' + df.StanceUpdated + ']'
  sep = ' [SEP] '
  df['input'] = df.sectok + sep + df.Premise + sep + df.Conclusion
  return Dataset.from_pandas(df)


def get_dds():
  tr = get_ds('training').map(tok_func, batched=True, remove_columns=cols_to_remove)
  val = get_ds('validation').map(tok_func, batched=True, remove_columns=cols_to_remove)
  te = get_ds('test').map(tok_func, batched=True, remove_columns=cols_to_remove)
  return DatasetDict({"train": tr, "valid":val, 'test':te})


dds = get_dds()

bs = 16//4
epochs = 6
lr = 2e-5/4
gc.collect()
torch.cuda.empty_cache()
transformers.set_seed(1234) 

args = TrainingArguments('outputs_xxlarge_v3_more_training_data', learning_rate=lr, 
                         save_strategy='epoch', 
                         # warmup_ratio=0.1, lr_scheduler_type='cosine',
                         load_best_model_at_end=False,
                         fp16=True, gradient_accumulation_steps=1,
    evaluation_strategy="epoch", 
    per_device_train_batch_size=bs, 
    per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, 
    weight_decay=0.01, 
    metric_for_best_model='f1_macro_0.25',
    save_total_limit = 1,
    report_to='none')
print(args)
model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                           num_labels=20, 
                                                           problem_type="multi_label_classification")
model.resize_token_embeddings(len(tok))

trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['valid'],
                  tokenizer=tok, compute_metrics=compute_metrics, )
trainer.train()
# Threshold analysis
sigmoid = torch.nn.Sigmoid()
valid_preds = trainer.predict(dds['valid'])
probs = sigmoid(torch.Tensor(valid_preds.predictions)).numpy()

y_true = valid_preds.label_ids
scores = []
ths = np.arange(0.05, 0.6,0.05 )
for threshold in ths:
  y_pred = np.zeros(probs.shape)
  y_pred[np.where(probs >= threshold)] = 1
  scores.append(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))
print(list(zip(ths, scores)))


# Test preds population
test_probs = torch.nn.Sigmoid()(torch.Tensor(trainer.predict(dds['test']).predictions)).numpy()
test_probs
THRESHOLD = 0.20
y_pred = np.zeros(test_probs.shape)
y_pred[np.where(test_probs >= THRESHOLD)] = 1



for idx, val in enumerate(f.columns[1:]):
  f[val] = y_pred[:, idx].astype('int')
f.to_csv("run_A_v1.csv", index=None, sep='\t')
