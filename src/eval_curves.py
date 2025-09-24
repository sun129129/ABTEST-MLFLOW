# src/eval_curves.py
import numpy as np, mlflow, joblib
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from features import load_split, build_logreg_features

ART = Path(__file__).resolve().parent.parent / "data" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

def _plot_xy(x, y, title, xlabel, ylabel, fname):
    plt.figure(figsize=(5,4))
    plt.plot(x, y)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    out = ART / fname
    plt.savefig(out, dpi=160); plt.close()
    return str(out)

def _lift_curve(y_true, y_prob, bins=10):
    # 누적 gain/lift
    order = np.argsort(-y_prob)
    y = np.asarray(y_true)[order]
    pos = y.sum()
    pct = np.linspace(0,1,bins+1)[1:]
    gains = []
    for p in pct:
        k = int(len(y)*p)
        gains.append(y[:k].sum()/pos if pos>0 else 0.0)
    lift = np.asarray(gains)/pct
    return pct, gains, lift

def main():
    # 데이터 + 인코더/모델 로드
    df = load_split("test")
    enc = joblib.load(ART / "logreg_ohe.pkl")
    X, y, _ = build_logreg_features(df, enc=enc, fit=False)
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    mA = joblib.load(ART / "logreg_model.pkl")
    mB = joblib.load(ART / "lgbm_model.pkl")

    pA = mA.predict_proba(X)[:,1]
    pB = mB.predict_proba(X)[:,1]

    mlflow.set_experiment("abtest_movielens")
    with mlflow.start_run(run_name="Curves_PR_ROC_Calib_Lift"):
        # ROC
        fa, ta, _ = roc_curve(y, pA); fb, tb, _ = roc_curve(y, pB)
        rocA = _plot_xy(fa, ta, "ROC - A(LogReg)", "FPR","TPR","roc_A.png")
        rocB = _plot_xy(fb, tb, "ROC - B(LightGBM)", "FPR","TPR","roc_B.png")
        mlflow.log_artifact(rocA); mlflow.log_artifact(rocB)

        # PR
        pa, ra, _ = precision_recall_curve(y, pA)
        pb, rb, _ = precision_recall_curve(y, pB)
        prA = _plot_xy(ra, pa, "PR - A(LogReg)", "Recall","Precision","pr_A.png")
        prB = _plot_xy(rb, pb, "PR - B(LightGBM)", "Recall","Precision","pr_B.png")
        mlflow.log_artifact(prA); mlflow.log_artifact(prB)

        # Calibration
        for name, p, fname in [("A(LogReg)", pA, "calib_A.png"), ("B(LGBM)", pB, "calib_B.png")]:
            prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            plt.figure(figsize=(4.5,4))
            plt.plot([0,1],[0,1],'--')
            plt.plot(prob_pred, prob_true, marker='o')
            plt.title(f"Calibration - {name}")
            plt.xlabel("Predicted prob."); plt.ylabel("Observed freq.")
            plt.tight_layout(); out = ART / fname; plt.savefig(out, dpi=160); plt.close()
            mlflow.log_artifact(out)
            mlflow.log_metric(f"{'A' if 'LogReg' in name else 'B'}_brier", brier_score_loss(y, p))

        # Lift / Cumulative Gain
        for tag, p, f1, f2 in [("A", pA, "lift_A.png", "gain_A.png"), ("B", pB, "lift_B.png", "gain_B.png")]:
            pct, gains, lift = _lift_curve(y, p)
            _ = _plot_xy(pct, lift, f"Lift - {tag}", "Population %","Lift", f1)
            _ = _plot_xy(pct, gains, f"Cumulative Gain - {tag}", "Population %","Gain", f2)
            mlflow.log_artifact(ART/f1); mlflow.log_artifact(ART/f2)

if __name__ == "__main__":
    main()
