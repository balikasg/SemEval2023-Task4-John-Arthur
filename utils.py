import numpy as np
import torch
import pandas as pd
from transformers import EvalPrediction
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score

COLS_TO_REMOVE = ("Argument ID", "Conclusion", "Stance", "Premise", "StanceUpdated")


labs = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: caring",
    "Benevolence: dependability",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
    "Universalism: objectivity",
]


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
        metrics[f"f1_macro_{threshold}"] = f1_score(
            y_true=y_true, y_pred=y_pred, average="macro"
        )
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def get_ds_last_submission(fold, tokenizer):
    if fold == "training":
        dfs = []
        for f in ["training", "validation"]:
            df = pd.read_csv(f"./arguments-{f}.tsv", delimiter="\t")
            labels = pd.read_csv(f"labels-{f}.tsv", delimiter="\t")
            labels["labels"] = labels.apply(
                lambda l: [float(l[lab]) for lab in labs], axis=1
            )
            df = df.merge(labels[["Argument ID", "labels"]], on=["Argument ID"])
            print(df.shape)
            dfs.append(df)
        df = pd.concat([dfs[0], dfs[1]])
        df["StanceUpdated"] = df["Stance"].apply(
            lambda l: "against" if "against" in l else "favour"
        )
        df["sectok"] = "[" + df.StanceUpdated + "]"
        sectoks = list(df.sectok.unique())
        tokenizer.add_special_tokens({"additional_special_tokens": sectoks})
    if fold == "validation":
        df = pd.read_csv("arguments-validation-zhihu.tsv", delimiter="\t")
        l = pd.read_csv("labels-validation-zhihu.tsv", delimiter="\t")
        l["labels"] = l.apply(lambda l: [float(l[lab]) for lab in labs], axis=1)
        df = df.merge(l[["Argument ID", "labels"]], on=["Argument ID"])

        df["StanceUpdated"] = df["Stance"].apply(
            lambda l: "against" if "against" in l else "favour"
        )
        df["sectok"] = "[" + df.StanceUpdated + "]"
    if fold == "test":
        df = pd.read_csv(f"./arguments-{fold}.tsv", delimiter="\t")
        labels = pd.read_csv(f"labels-{fold}.tsv", delimiter="\t")
        labels["labels"] = labels.apply(
            lambda l: [float(l[lab]) for lab in labs], axis=1
        )
        df = df.merge(labels[["Argument ID", "labels"]], on=["Argument ID"])
        df["StanceUpdated"] = df["Stance"].apply(
            lambda l: "against" if "against" in l else "favour"
        )
        df["sectok"] = "[" + df.StanceUpdated + "]"
    sep = " [SEP] "
    df["input"] = df.sectok + sep + df.Premise + sep + df.Conclusion
    return Dataset.from_pandas(df)


def get_ds_vanilla(fold, tokenizer):
    """A lot of code duplication here"""
    df = pd.read_csv(f"./arguments-{fold}.tsv", delimiter="\t")
    labels = pd.read_csv(f"labels-{fold}.tsv", delimiter="\t")
    labels["labels"] = labels.apply(
        lambda l: [float(l[lab]) for lab in labs], axis=1
    )
    df = df.merge(labels[["Argument ID", "labels"]], on=["Argument ID"])
    print(df.shape)
    df["StanceUpdated"] = df["Stance"].apply(
        lambda l: "against" if "against" in l else "favour"
    )
    df["sectok"] = "[" + df.StanceUpdated + "]"
    sectoks = list(df.sectok.unique())
    if fold == 'training':
        tokenizer.add_special_tokens({"additional_special_tokens": sectoks})
    sep = " [SEP] "
    df["input"] = df.sectok + sep + df.Premise + sep + df.Conclusion
    return Dataset.from_pandas(df)


def get_dds(tokenizer, use_vanilla=True):
    """If use_vanilla=True, use standard splits. Else, merge validation/train (last submission)"""
    def tok_func(x):
        return tokenizer(x["input"])
    get_ds = get_ds_vanilla if use_vanilla else get_ds_last_submission
    tr = get_ds("training", tokenizer).map(tok_func, batched=True, remove_columns=COLS_TO_REMOVE)
    val = get_ds("validation", tokenizer).map(
        tok_func, batched=True, remove_columns=COLS_TO_REMOVE
    )
    te = get_ds("test", tokenizer).map(tok_func, batched=True, remove_columns=COLS_TO_REMOVE)
    return DatasetDict({"train": tr, "valid": val, "test": te})
