# ABTest-MLflow-MovieLens

MLflow 기반 추천 모델 **A/B 테스트 파이프라인** 예제입니다.  
MovieLens 데이터를 활용해 **Policy A(Logistic Regression)** vs **Policy B(LightGBM)** 를 학습하고,  
**오프라인 성능 비교 → Model Registry 등록 → Router(PyFunc) → 데모 실행**까지 재현합니다.

---

## 📊 데이터 설명

- **데이터셋**: MovieLens 100K (943명 사용자, 1,682편 영화, 평점 100,000건)
- **주요 컬럼**
  - `userId`: 사용자 ID
  - `movieId`: 영화 ID
  - `rating`: 1~5 평점
  - `timestamp`: 시청 시각
- **메타데이터**
  - 장르, 영화 제목, 사용자 나이/성별/직업
- **전처리 과정**
  - 평점 → binary feedback (좋아요=1, 비선호=0)
  - 사용자-영화 행렬 생성
  - 장르/나이대별 파생 피처 생성

---

## 📁 Project Structure

```
abtest-mlflow-movielens/
├─ data/                     
│  ├─ ml-100k.zip
│  └─ processed/              # prepare_movielens.py 실행 후 생성
├─ src/
│  ├─ prepare_movielens.py    # 데이터 전처리
│  ├─ features.py             # 피처 엔지니어링
│  ├─ train_logreg.py         # Policy A (Logistic Regression)
│  ├─ train_lgbm.py           # Policy B (LightGBM)
│  ├─ eval_offline_ab.py      # 오프라인 성능 비교
│  ├─ eval_curves.py          # ROC/PR/Calibration/Lift 곡선
│  ├─ eval_segments.py        # 세그먼트별 성능 평가
│  ├─ eval_cv.py              # 교차검증
│  ├─ register_models.py      # PolicyA/B 모델 Registry 등록
│  ├─ ab_router_pyfunc.py     # Router(PyFunc) 정의
│  ├─ ab_router_register.py   # Router 모델 Registry 등록
│  └─ router_infer_demo.py    # Router 데모 실행
├─ requirements.txt
└─ ...
```

---

## 🚀 Quickstart

### 0) 환경 구성
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

### 1) 데이터 준비
```bash
python src/prepare_movielens.py
```

### 2) 모델 학습 (A/B)
```bash
python src/train_logreg.py   # Policy A
python src/train_lgbm.py     # Policy B
```

### 3) 오프라인 평가
```bash
python src/eval_offline_ab.py
python src/eval_curves.py
python src/eval_segments.py
python src/eval_cv.py
```

### 4) 모델/라우터 등록
```bash
python src/register_models.py
python src/ab_router_pyfunc.py
python src/ab_router_register.py
```

### 5) Router 데모
```bash
python src/router_infer_demo.py
```

---

## 🧭 파이프라인 요약

```
[Data] → prepare_movielens → features
      → train_logreg / train_lgbm → MLflow Tracking
      → eval_* (offline/curves/segments/cv)
      → register_models (PolicyA/PolicyB)
      → ab_router_pyfunc (+ register)
      → router_infer_demo
```

---

## 🖥️ MLflow UI 실행
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
- Experiments: 학습 파라미터/메트릭, 시각화 아티팩트 확인
- Models: PolicyA, PolicyB, router alias 관리

---

## 🧪 실습 시나리오 요약
1. **데이터 전처리** (`prepare_movielens.py`, `features.py`)
2. **Policy A vs Policy B 학습** (`train_logreg.py`, `train_lgbm.py`)
3. **오프라인 평가** (`eval_offline_ab.py`, `eval_curves.py`, `eval_segments.py`, `eval_cv.py`)
4. **Registry 등록** (`register_models.py`, `ab_router_register.py`)
5. **Router 데모 실행** (`router_infer_demo.py`)

---

## ⚙️ 개선 방향
- 무작위 Router → MAB(Thompson Sampling, UCB) 확장
- Streamlit/FastAPI 대시보드 연동
- Kubernetes 기반 자동 스케일링

---

## 📦 데이터 & 라이선스
- MovieLens 데이터셋 사용 (GroupLens 라이선스 준수)
- 본 저장소 코드는 학습/실습용 예제
