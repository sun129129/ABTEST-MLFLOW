# src/train_lgbm.py
import mlflow, joblib
from pathlib import Path
import lightgbm as lgb
from features import load_split, build_logreg_features
from utils import binary_metrics

ART_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT = "abtest_movielens"

def main():
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="PolicyB_LightGBM"):
        # 데이터 로드
        df_tr = load_split("train")
        df_va = load_split("valid")

        # 동일한 OHE 피처 사용 (A 모델과 동일 전처리)
        Xtr, ytr, enc = build_logreg_features(df_tr, enc=None, fit=True)
        Xva, yva, _   = build_logreg_features(df_va, enc=enc, fit=False)

        # LightGBM 파라미터
        params = dict(
            objective="binary",
            metric="binary_logloss",
            num_leaves=64,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=50,
            n_estimators=500,
            verbose=-1,
        )
        mlflow.log_params(params)

        # 모델 학습
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),  # 얼리스탑
                lgb.log_evaluation(period=0),            # 학습 로그 숨김
            ],
        )

        # 검증 성능 평가
        p_va = clf.predict_proba(Xva)[:, 1]
        m_va = binary_metrics(yva, p_va)
        for k, v in m_va.items():
            mlflow.log_metric(f"valid_{k}", float(v))

        # 아티팩트 저장
        joblib.dump(enc, ART_DIR / "logreg_ohe.pkl")
        joblib.dump(clf, ART_DIR / "lgbm_model.pkl")
        mlflow.log_artifact(str(ART_DIR / "lgbm_model.pkl"))

        print("Policy B (LightGBM) valid:", m_va)

if __name__ == "__main__":
    main()
