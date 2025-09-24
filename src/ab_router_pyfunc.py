# src/ab_router_pyfunc.py
import mlflow
import mlflow.pyfunc
import pandas as pd
import hashlib
import joblib
from mlflow.models.signature import infer_signature

ART = "data/artifacts"

# 미리 학습된 A, B 모델 로드
modelA = joblib.load(f"{ART}/logreg_model.pkl")
modelB = joblib.load(f"{ART}/lgbm_model.pkl")

class ABRouter(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: pd.DataFrame):
        """userId 해시로 짝/홀 구분 → A 또는 B 선택"""
        outputs = []
        for _, row in model_input.iterrows():
            uid = int(row["userId"])
            hashed = int(hashlib.md5(str(uid).encode()).hexdigest(), 16)
            if hashed % 2 == 0:
                score = modelA.predict_proba([[row["movieId"]]])[0, 1]
                outputs.append({"assigned": "A", "score": score})
            else:
                score = modelB.predict_proba([[row["movieId"]]])[0, 1]
                outputs.append({"assigned": "B", "score": score})
        return pd.DataFrame(outputs)

if __name__ == "__main__":
    # 스키마 정의용 input/output 예시
    input_example = pd.DataFrame({"userId": [1], "movieId": [10]})
    output_example = pd.DataFrame({"assigned": ["A"], "score": [0.8]})
    signature = infer_signature(input_example, output_example)

    mlflow.set_experiment("abtest_movielens")
    with mlflow.start_run(run_name="AB_Router_Demo"):
        mlflow.pyfunc.log_model(
            artifact_path="ab_router",
            python_model=ABRouter(),
            input_example=input_example,
            signature=signature
        )
