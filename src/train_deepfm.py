# src/train_deepfm.py
import os, mlflow, torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import get_feature_names
from features import load_split, build_deepfm_inputs
from utils import binary_metrics

ART_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "abtest_movielens"

def to_tensor(x):
    return torch.tensor(x)

def make_loader(X_dict, y, batch_size=1024, shuffle_flag=True):
    tensors = [to_tensor(X_dict["user_id"]), to_tensor(X_dict["item_id"]), to_tensor(X_dict["genres"]), to_tensor(y)]
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag, drop_last=False)

def main():
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="PolicyB_DeepFM"):
        df_tr = load_split("train")
        df_va = load_split("valid")

        Xtr, feat_cols, ytr = build_deepfm_inputs(df_tr)
        Xva, _, yva = build_deepfm_inputs(df_va)

        linear_feature_columns  = feat_cols
        dnn_feature_columns     = feat_cols
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        model = DeepFM(linear_feature_columns, dnn_feature_columns,
                       task='binary', l2_reg_embedding=1e-6, dnn_hidden_units=(256,128,64), device='cpu')
        mlflow.log_params({"model":"DeepFM", "dnn_hidden_units":"256-128-64", "l2_reg_embedding":1e-6, "batch_size":1024, "epochs":3, "lr":1e-3})

        # DeepCTR는 내부 fit 기능이 있지만, 여기서는 명시적으로 예측만 수집
        model.compile("adam", "binary_crossentropy", metrics=["auc"], lr=1e-3)

        # fit
        history = model.fit(Xtr, ytr, batch_size=1024, epochs=3, verbose=2, validation_split=0.0)
        # predict on valid
        p_va = model.predict(Xva, batch_size=1024)
        m_va = binary_metrics(yva, p_va)
        for k,v in m_va.items(): mlflow.log_metric(f"valid_{k}", float(v))

        # 저장(DeepCTR은 state_dict 저장 권장)
        save_path = ART_DIR / "deepfm_state_dict.pt"
        torch.save(model.state_dict(), save_path)
        mlflow.log_artifact(str(save_path))

        print("Policy B(DeepFM) valid:", m_va)

if __name__ == "__main__":
    main()


# 0) 데모 스크립트 살짝 수정 (메트릭만 기록)

# src/router_infer_demo.py에서 artifact 로깅 부분을 제거하고 메트릭만 남기세요:

# # ... (상단 동일)
# with mlflow.start_run(run_name="Router_Demo") as run:
#     mlflow.log_metrics({
#         "PolicyA_ratio": (df_out["assigned"] == "PolicyA").mean(),
#         "PolicyB_ratio": (df_out["assigned"] == "PolicyB").mean()
#     })
#     print(f"\n[MLflow] Demo run logged under run_id={run.info.run_id}")


# 그리고 실행:

# python src/router_infer_demo.py

# 1) MLflow UI 여는 방법 (둘 중 하나)
# 방법 A: 파일 기반(local)로 기록 중이라면

# 프로젝트 루트에서:

# mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 8080


# 브라우저에서 http://127.0.0.1:8080 접속

# 방법 B: 이미 서버 모드(Registry까지)로 띄운 경우

# 이전에 이런 식으로 켰다면:

# mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 8080


# → 같은 주소(http://127.0.0.1:8080)로 접속하면 됩니다.

# 어떤 방식이든 UI는 똑같이 보이지만, backend-store-uri가 가리키는 위치(./mlruns 또는 sqlite)가 현재 프로젝트의 로그 저장소와 일치해야 합니다.

# 2) UI에서 PolicyA_ratio / PolicyB_ratio 보는 위치

# 왼쪽 Experiments → abtest_movielens 클릭

# Runs 목록에서 Router_Demo 선택

# 상단 탭에서 Metrics 선택

# 우측 검색창에 PolicyA_ratio, PolicyB_ratio 입력 → 값/차트 확인

# 또는 여러 Run을 체크박스로 선택 → Compare 버튼 →

# Metrics 탭에 두 지표가 나란히 보입니다.

# 3) 혹시 안 보이면?

# 마지막 실행이 에러 없이 끝났는지 콘솔 메시지에서 확인:
# "[MLflow] Demo run logged under run_id=..." 가 나와야 합니다.

# Router_Demo Run이 안 보이면, 실험 이름이 동일한지 확인 (mlflow.set_experiment("abtest_movielens")가 데모 코드에 없어도 기본 실험으로 기록됩니다. 필요하면 맨 위에 mlflow.set_experiment("abtest_movielens") 한 줄 추가해도 좋아요.)

# 여전히 artifact도 보고 싶다면, MLflow 서버 방식(방법 B) 로 전환한 뒤 log_artifact를 다시 켜면 됩니다.

# 4) CSV는 어디서 보나?

# 지금은 artifact 업로드를 끈 상태라 로컬 파일로 저장됩니다:
# artifacts/router_demo_results.csv

# 바로 엑셀/판다스로 열어보면 돼요.

# 요약

# 방금 에러가 났던 실행은 메트릭이 기록 안 됐을 것 → 수정본으로 재실행 필요

# UI에서는 Experiments → abtest_movielens → Router_Demo → Metrics 에서 확인

# artifact까지 UI에서 보고 싶으면 MLflow 서버 모드로 띄우세요