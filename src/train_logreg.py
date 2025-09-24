# src/train_logreg.py
import os, mlflow, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from features import load_split, build_logreg_features
from utils import binary_metrics

ART_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "abtest_movielens"

def main():
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="PolicyA_LogReg"):
        df_tr = load_split("train")
        df_va = load_split("valid")

        Xtr, ytr, enc = build_logreg_features(df_tr, enc=None, fit=True)
        Xva, yva, _   = build_logreg_features(df_va, enc=enc, fit=False)

        params = dict(penalty="elasticnet", l1_ratio=0.1, C=1.0, solver="saga", max_iter=200)
        mlflow.log_params(params)

        clf = LogisticRegression(**params, n_jobs=-1, random_state=42)
        clf.fit(Xtr, ytr)

        # valid metrics
        p_va = clf.predict_proba(Xva)[:,1]
        m_va = binary_metrics(yva, p_va)
        for k,v in m_va.items(): mlflow.log_metric(f"valid_{k}", v)

        # save artifacts
        joblib.dump(clf, ART_DIR / "logreg_model.pkl")
        mlflow.log_artifact(str(ART_DIR / "logreg_model.pkl"))
        mlflow.sklearn.log_model(clf, artifact_path="model")

        print("Policy A(LogReg) valid:", m_va)

if __name__ == "__main__":
    main()
