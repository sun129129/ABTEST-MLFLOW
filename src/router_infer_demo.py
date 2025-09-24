import pandas as pd
import mlflow
import os
from pathlib import Path
import numpy as np

MODEL_URI = "models:/movielens_ctr_router@router"
router_model = mlflow.pyfunc.load_model(MODEL_URI)

def main(n=20, seed=42):
    # 샘플 데이터 (label 있는 test.parquet에서 가져오기)
    test_path = Path("data/processed/test.parquet")
    df = pd.read_parquet(test_path).sample(n=n, random_state=seed).reset_index(drop=True)

    # Router 예측 실행
    preds = router_model.predict(df)

    # preds가 단순 float/array라면 → 임의 규칙으로 A/B 배정
    # (짝수 userId → PolicyA, 홀수 userId → PolicyB)
    if isinstance(preds, (np.ndarray, list)) and not isinstance(preds[0], dict):
        assigned = df["userId"].apply(lambda x: "PolicyA" if x % 2 == 0 else "PolicyB")
        df_out = df[["userId", "movieId", "label"]].copy()
        df_out["assigned"] = assigned
        df_out["score"] = preds
    else:
        # dict 구조인 경우 원래대로 처리
        pred_df = pd.DataFrame(preds)
        df_out = pd.concat([df[["userId","movieId","label"]], pred_df], axis=1)

    print("=== 샘플 Router 배정 결과 ===")
    print(df_out.head(10))

    print("\n=== Assignment 비율 ===")
    print(df_out["assigned"].value_counts(normalize=True).round(3))


# --- CSV 저장 ---
    os.makedirs("artifacts", exist_ok=True)
    csv_path = os.path.join("artifacts", "router_demo_results.csv")
    df_out.to_csv(csv_path, index=False)

# --- MLflow 로그는 메트릭만 남기고 artifact는 빼기 ---
    with mlflow.start_run(run_name="Router_Demo") as run:
        mlflow.log_metrics({
            "PolicyA_ratio": (df_out["assigned"] == "PolicyA").mean(),
            "PolicyB_ratio": (df_out["assigned"] == "PolicyB").mean()
        })
        print(f"\n[MLflow] Demo run logged under run_id={run.info.run_id}")
