# ABTest-MLflow-MovieLens 🎬

MLflow 기반 **추천 모델 A/B 테스트 파이프라인** 예제 프로젝트입니다.  
MovieLens 데이터를 활용하여 **Logistic Regression (Policy A)**, **LightGBM (Policy B)**를 학습하고,  
오프라인 성능 평가 → 모델 레지스트리 등록 → Router 모델로 트래픽 분배까지 구현합니다.

---

## 📂 프로젝트 구조

```bash
abtest-mlflow-movielens/
│
├── data/                           # MovieLens 원본 + 전처리 데이터
│   ├── ml-1m.zip
│   ├── ml-100k.zip
│   └── processed/                  # prepare_movielens.py 실행 후 저장됨
│
├── src/
│   ├── prepare_movielens.py        # 데이터 전처리
│   ├── features.py                 # 피처 생성 함수
│   ├── train_logreg.py             # Policy A 학습 (로지스틱)
│   ├── train_lgbm.py               # Policy B 학습 (LightGBM)
│   ├── eval_offline_ab.py          # 오프라인 A/B 평가 (전체 비교)
│   ├── eval_curves.py              # ROC/PR 곡선 등 시각화
│   ├── eval_segments.py            # 세그먼트별 성능 비교
│   ├── eval_cv.py                  # K-Fold 교차검증 결과
│   ├── register_models.py          # A/B 모델 Registry 등록 (alias=PolicyA, PolicyB)
│   ├── ab_router_pyfunc.py         # Router 모델 (PyFunc, schema 포함) ✅
│   ├── ab_router_register.py       # Router 모델 Registry 등록 (alias=router)
│   └── utils.py                    # 공통 유틸 함수
│
├── mlruns/                         # MLflow 실험 로그 저장소
├── mlflow.db                       # SQLite 기반 MLflow backend store
├── requirements.txt
└── README.md
```

---

## 🚀 실행 단계

### 1️⃣ 데이터 준비
```bash
python src/prepare_movielens.py
```

### 2️⃣ 개별 모델 학습 (A/B)
```bash
python src/train_logreg.py   # Policy A
python src/train_lgbm.py     # Policy B
```

### 3️⃣ 오프라인 평가 + 시각화
```bash
python src/eval_offline_ab.py   # 성능 비교 + 막대그래프
python src/eval_curves.py       # ROC/PR/Calibration/Lift 곡선
python src/eval_segments.py     # Cold-start/장르별/인기 아이템 비교
python src/eval_cv.py           # K-Fold 교차검증 결과 (박스플롯)
```

### 4️⃣ Router(PyFunc) 등록
```bash
python src/ab_router_pyfunc.py
```

### 5️⃣ Model Registry 등록 + Alias 부여
```bash
python src/register_models.py       # PolicyA, PolicyB
python src/ab_router_register.py    # router
```

### 6️⃣ Router 데모 (배정 + Score 확인)
```bash
python src/router_infer_demo.py
```

---

## 🎯 ABTest 구현 포인트

- **Tracking**  
  - A와 B 모델 성능을 동일한 데이터셋에서 비교 (정량적 평가지표 기반)  
  - 오프라인 A/B Test 느낌

- **Model Registry**  
  - PolicyA, PolicyB 모델을 등록하고 alias로 관리  
  - 실제 서비스 환경에서 “두 정책이 병렬 존재” 구조 반영

- **Router (PyFunc)**  
  - 트래픽 분리 역할  
  - 예: `user_id % 2 == 0 → PolicyA`, `user_id % 2 == 1 → PolicyB`  
  - ABTEST의 핵심 원리인 **무작위 배분**을 코드로 구현

---

## 📑 MLflow와의 매핑

| 파일 / 단계                                                    | MLflow 컴포넌트         | 설명                                                                 |
|----------------------------------------------------------------|--------------------------|----------------------------------------------------------------------|
| `train_logreg.py`, `train_lgbm.py`                             | **Tracking**             | 파라미터, 메트릭, 아티팩트 기록 → "A vs B 비교 증거" 남김            |
| `eval_offline_ab.py`, `eval_curves.py`, `eval_segments.py`, `eval_cv.py` | **Tracking**             | ROC, PR, 세그먼트 비교 결과를 시각화 & 로그로 저장                    |
| `prepare_movielens.py`, `features.py`                          | **Projects**             | 데이터 전처리/피처 생성 코드. 재현 가능한 실험 단위                  |
| `register_models.py`                                           | **Model Registry**       | 학습 모델 등록 + alias: PolicyA, PolicyB                             |
| `ab_router_pyfunc.py`, `ab_router_register.py`                 | **Models + Registry**    | Router 모델 작성 & 등록 (alias=router) → 실제 온라인 배분 구조 반영   |
| 전체 파이프라인                                                | **MLflow Models**        | 모든 모델을 pyfunc/LightGBM 등으로 패키징 → 재사용 및 서비스 가능    |

---

## 📌 요약

- **ABTest = Tracking + Registry + Router**
  - Tracking → 성능 기록
  - Registry → 모델 버전 관리
  - Router → 실제 배분/할당

👉 MLflow로 실험을 추적하고, 모델을 관리하며, Router를 통해 실제 A/B 테스트의 기본 구조를 재현할 수 있습니다.

---

📎 더 자세한 설명과 시각화 결과는 `mlruns/`와 MLflow UI에서 확인할 수 있습니다.
