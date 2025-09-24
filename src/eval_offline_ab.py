# src/eval_offline_ab.py
import os
from pathlib import Path
import joblib
import numpy as np
import mlflow

from features import load_split, build_logreg_features
from utils import binary_metrics, plot_bar

ART_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "abtest_movielens"

LOGREG_MODEL_PATH = ART_DIR / "logreg_model.pkl"
LGBM_MODEL_PATH   = ART_DIR / "lgbm_model.pkl"
ENC_PATH          = ART_DIR / "logreg_ohe.pkl"

def _require(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}. 먼저 해당 학습 스크립트를 실행했는지 확인하세요.")

def main():
    # 준비물 체크
    _require(ENC_PATH, "Encoder (logreg_ohe.pkl)")
    _require(LOGREG_MODEL_PATH, "A(LogReg) model")
    _require(LGBM_MODEL_PATH, "B(LightGBM) model")

    # 데이터 로드
    df_te = load_split("test")

    # 인코더 로드 & 테스트 피처 생성 (A/B 동일 전처리)
    enc = joblib.load(ENC_PATH)
    Xte, yte, _ = build_logreg_features(df_te, enc=enc, fit=False)

    # 모델 로드
    from sklearn.linear_model import LogisticRegression
    clfA: LogisticRegression = joblib.load(LOGREG_MODEL_PATH)

    import lightgbm as lgb
    clfB: lgb.LGBMClassifier = joblib.load(LGBM_MODEL_PATH)

    # 예측확률
    pA = clfA.predict_proba(Xte)[:, 1]
    pB = clfB.predict_proba(Xte)[:, 1]

    # 메트릭
    mA = binary_metrics(yte, pA)  # {'auc', 'pr_auc', 'logloss'}
    mB = binary_metrics(yte, pB)

    # MLflow 로깅
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="Eval_Offline_AB"):
        # A 결과
        for k, v in mA.items():
            mlflow.log_metric(f"A_logreg_test_{k}", float(v))
        # B 결과
        for k, v in mB.items():
            mlflow.log_metric(f"B_lgbm_test_{k}", float(v))

        # 시각화(막대그래프) 저장 & 업로드
        chart_auc = plot_bar(
            {"A_LogReg": mA["auc"], "B_LightGBM": mB["auc"]},
            "AUC (Test)",
            "auc_bar.png",
        )
        chart_ll = plot_bar(
            {"A_LogReg": mA["logloss"], "B_LightGBM": mB["logloss"]},
            "LogLoss (lower is better)",
            "logloss_bar.png",
        )
        mlflow.log_artifact(chart_auc)
        mlflow.log_artifact(chart_ll)

        # 콘솔 요약
        print("\n=== Test Metrics ===")
        print(f"A(LogReg)    : AUC={mA['auc']:.4f}  PR-AUC={mA['pr_auc']:.4f}  LogLoss={mA['logloss']:.4f}")
        print(f"B(LightGBM)  : AUC={mB['auc']:.4f}  PR-AUC={mB['pr_auc']:.4f}  LogLoss={mB['logloss']:.4f}")
        print("\nArtifacts saved:", chart_auc, chart_ll)

if __name__ == "__main__":
    main()
