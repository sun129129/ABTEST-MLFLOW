# src/utils.py
import os, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

ART_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

def binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob), 1e-7, 1-1e-7)
    auc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    return {"auc": float(auc), "pr_auc": float(prauc), "logloss": float(ll)}

def plot_bar(values:dict, title:str, fname:str):
    keys = list(values.keys())
    vals = [values[k] for k in keys]
    plt.figure(figsize=(4.5,3))
    plt.bar(keys, vals)
    plt.title(title)
    plt.tight_layout()
    out = ART_DIR / fname
    plt.savefig(out, dpi=160)
    plt.close()
    return str(out)

def hr_ndcg_at_k(scores_pos, scores_neg, k=10):
    """
    per-user arrays of positive and negative scores
    returns HR@k, NDCG@k averaged
    """
    hr_list, ndcg_list = [], []
    for s_pos, s_neg in zip(scores_pos, scores_neg):
        scores = np.concatenate([np.asarray(s_pos), np.asarray(s_neg)])
        labels = np.concatenate([np.ones_like(s_pos), np.zeros_like(s_neg)])
        idx = np.argsort(-scores)
        topk = labels[idx][:k]
        hr = 1.0 if topk.sum() > 0 else 0.0
        # first relevant position
        dcg = 0.0
        for rank, lab in enumerate(topk, start=1):
            if lab == 1:
                dcg = 1.0 / math.log2(rank+1)
                break
        idcg = 1.0  # best case: relevant in rank1
        ndcg = dcg / idcg
        hr_list.append(hr); ndcg_list.append(ndcg)
    return float(np.mean(hr_list)), float(np.mean(ndcg_list))
