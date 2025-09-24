# src/eval_segments.py  (안전판: 장르 컬럼 없어도 동작)
import numpy as np, pandas as pd, mlflow, joblib
from pathlib import Path
from features import load_split, build_logreg_features
from utils import binary_metrics

ART = Path(__file__).resolve().parent.parent / "data" / "artifacts"
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
ART.mkdir(parents=True, exist_ok=True)

def _segment_report(df, y, pA, pB, name: str):
    """세그먼트별 A/B 지표 로깅 (장르 없으면 자동 스킵)"""
    def _log(seg_name: str, idx: np.ndarray):
        if idx.sum() == 0:
            return
        mA = binary_metrics(y[idx], pA[idx]); mB = binary_metrics(y[idx], pB[idx])
        for k, v in mA.items(): mlflow.log_metric(f"{name}_{seg_name}_A_{k}", float(v))
        for k, v in mB.items(): mlflow.log_metric(f"{name}_{seg_name}_B_{k}", float(v))

    # --- cold-start (train에 없던 유저/아이템) ---
    try:
        df_train = pd.read_parquet(PROCESSED / "train.parquet")
        seen_users  = set(df_train["userId"].unique())
        seen_items  = set(df_train["movieId"].unique())
        cold_user = ~df["userId"].isin(seen_users)
        cold_item = ~df["movieId"].isin(seen_items)
        _log("cold_user", cold_user.to_numpy())
        _log("cold_item", cold_item.to_numpy())
    except Exception:
        # train.parquet 없거나 스키마 다르면 스킵
        pass

    # --- 인기/롱테일 (test 내 출현빈도 기준 상위 10%) ---
    counts = df["movieId"].value_counts()
    if len(counts) > 0:
        top_k = int(max(1, 0.1 * len(counts)))
        popular_ids = set(counts.index[:top_k])
        popular = df["movieId"].isin(popular_ids).to_numpy()
        _log("popular_top10pct", popular)
        _log("long_tail", ~popular)

    # --- 장르별 (g0.. 형태가 실제로 있을 때만) ---
    genre_cols = [c for c in df.columns if c.startswith("g")]
    for g in genre_cols[:8]:  # 너무 많으면 상위 8개만
        idx = (df[g] == 1).to_numpy()
        _log(f"genre_{g}", idx)

def main():
    df = load_split("test")
    enc = joblib.load(ART / "logreg_ohe.pkl")
    X, y, _ = build_logreg_features(df, enc=enc, fit=False)

    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    mA: LogisticRegression = joblib.load(ART / "logreg_model.pkl")
    mB: lgb.LGBMClassifier = joblib.load(ART / "lgbm_model.pkl")

    pA = mA.predict_proba(X)[:, 1]
    pB = mB.predict_proba(X)[:, 1]

    mlflow.set_experiment("abtest_movielens")
    with mlflow.start_run(run_name="Segment_Analysis"):
        _segment_report(df, y, pA, pB, "test")
        print("Segment metrics logged to MLflow.")

if __name__ == "__main__":
    main()
