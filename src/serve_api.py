# src/serve_api.py
from typing import List, Optional, Union
import os

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# -----------------------
# 설정
# -----------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
# 기본은 Registry alias. 필요 시 runs:/<run_id>/ab_router 로 교체 가능
MODEL_URI = os.getenv("ROUTER_MODEL_URI", "models:/movielens_ctr_router@router")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 모델 로드 (서버 기동 시 1회)
try:
    router_model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(
        f"[serve_api] Failed to load router model from '{MODEL_URI}'.\n"
        f"Hint: set ROUTER_MODEL_URI (e.g., runs:/<run_id>/ab_router) or ensure Registry alias exists.\n"
        f"Tracking URI: {MLFLOW_TRACKING_URI}\nError: {e}"
    )

# -----------------------
# FastAPI 앱
# -----------------------
app = FastAPI(title="MovieLens AB Router API", version="1.1.0")


# 요청/응답 스키마
class PredictIn(BaseModel):
    userId: int
    movieId: int
    label: Optional[int] = None  # 데모/검증용


class PredictOut(BaseModel):
    userId: int
    movieId: int
    assigned: str
    score: float
    label: Optional[int] = None


def _normalize_predictions(df: pd.DataFrame, preds: Union[List, pd.Series]) -> pd.DataFrame:
    """
    router pyfunc 출력 정규화:
      - [{'model': 'PolicyA', 'score': 0.7}, ...]
      - [0.12, 0.87, ...]
    """
    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
        pred_df = pd.DataFrame(preds)
        if "model" in pred_df.columns:
            pred_df.rename(columns={"model": "assigned"}, inplace=True)
        out = pd.concat([df.reset_index(drop=True), pred_df[["assigned", "score"]]], axis=1)
        return out

    # 점수만 나오는 경우: userId 짝/홀 임시 배정(데모)
    assigned = df["userId"].apply(lambda x: "PolicyA" if (x % 2 == 0) else "PolicyB")
    out = df.copy().reset_index(drop=True)
    out["assigned"] = assigned
    out["score"] = preds
    return out


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=PredictOut)
def predict_one(item: PredictIn):
    try:
        df = pd.DataFrame([item.dict()])
        preds = router_model.predict(df)
        out = _normalize_predictions(df, preds).iloc[0]
        return PredictOut(
            userId=int(out["userId"]),
            movieId=int(out["movieId"]),
            assigned=str(out["assigned"]),
            score=float(out["score"]),
            label=(int(out["label"]) if "label" in out and pd.notnull(out["label"]) else None),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[predict_one] {e}")


@app.post("/bulk_predict")
def bulk_predict(items: List[PredictIn] = Body(...)):
    try:
        df = pd.DataFrame([x.dict() for x in items])
        preds = router_model.predict(df)
        out_df = _normalize_predictions(df, preds)

        summary = {
            "PolicyA_ratio": float((out_df["assigned"] == "PolicyA").mean()),
            "PolicyB_ratio": float((out_df["assigned"] == "PolicyB").mean()),
            "PolicyA_mean_score": float(out_df.loc[out_df["assigned"] == "PolicyA", "score"].mean())
                                  if (out_df["assigned"] == "PolicyA").any() else None,
            "PolicyB_mean_score": float(out_df.loc[out_df["assigned"] == "PolicyB", "score"].mean())
                                  if (out_df["assigned"] == "PolicyB").any() else None,
        }

        return {"summary": summary, "rows": out_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[bulk_predict] {e}")
