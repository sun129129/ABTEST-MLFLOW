# src/eval_cv.py
import numpy as np, mlflow, joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from features import load_split, build_logreg_features
from utils import binary_metrics, plot_bar

ART = Path(__file__).resolve().parent.parent / "data" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

def main(k=5):
    df = load_split("train")
    enc_path = ART / "logreg_ohe.pkl"
    # fold 마다 enc을 다시 fit하여 편향 방지
    aucA, aucB = [], []

    mlflow.set_experiment("abtest_movielens")
    with mlflow.start_run(run_name=f"CV_{k}fold"):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        for i, (_, idx) in enumerate(skf.split(df[["userId","movieId"]], df["label"])):
            valid = df.iloc[idx]
            # A
            XA, yA, encA = build_logreg_features(valid, enc=None, fit=True)
            clfA = LogisticRegression(penalty="elasticnet", l1_ratio=0.1, C=1.0, solver="saga", max_iter=200, n_jobs=-1).fit(XA, yA)
            pA = clfA.predict_proba(XA)[:,1]
            mA = binary_metrics(yA, pA); aucA.append(mA["auc"])
            # B (간이: LGBM을 same split으로)
            import lightgbm as lgb
            XB, yB, encB = build_logreg_features(valid, enc=None, fit=True)
            clfB = lgb.LGBMClassifier(objective="binary", n_estimators=200).fit(XB, yB)
            pB = clfB.predict_proba(XB)[:,1]
            mB = binary_metrics(yB, pB); aucB.append(mB["auc"])

        mlflow.log_metric("A_LogReg_auc_mean", float(np.mean(aucA)))
        mlflow.log_metric("A_LogReg_auc_std",  float(np.std(aucA)))
        mlflow.log_metric("B_LGBM_auc_mean",   float(np.mean(aucB)))
        mlflow.log_metric("B_LGBM_auc_std",    float(np.std(aucB)))

        # 박스플롯 저장
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4.5,3))
        plt.boxplot([aucA, aucB], labels=["A_LogReg","B_LGBM"])
        plt.title(f"AUC {k}-fold")
        plt.tight_layout(); out = ART / "cv_auc_box.png"; plt.savefig(out, dpi=160); plt.close()
        mlflow.log_artifact(out)

if __name__ == "__main__":
    main()
