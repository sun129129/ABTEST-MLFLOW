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
│   ├── register_models.py          # A/B 모델 Registry 등록(alias=PolicyA, PolicyB)
│   ├── ab_router_pyfunc.py         # Router 모델(pyfunc, schema 포함)   ✅
│   ├── ab_router_register.py       # Router 모델 Registry 등록(alias=router)
│   └── utils.py                    # 공통 유틸 함수
│
├── mlruns/                         # MLflow 실험 로그 저장소
├── mlflow.db                       # SQLite 기반 MLflow backend store
├── requirements.txt
└── README.md


# 1) 데이터 준비
python src/prepare_movielens.py

# 2) 개별 모델 학습 (A: Logistic, B: LightGBM)
python src/train_logreg.py
python src/train_lgbm.py

# 3) 오프라인 AB 평가 + 시각화
python src/eval_offline_ab.py     # A/B 테스트셋 성능 비교 + 막대그래프
python src/eval_curves.py         # ROC/PR/Calibration/Lift
python src/eval_segments.py       # cold-start/인기/장르별 비교
python src/eval_cv.py             # K-fold 분산(박스플롯)

# 4) Router(pyfunc) 등록 (스키마 포함)
python src/ab_router_pyfunc.py

# 5) Registry 등록 + alias 부여
python src/register_models.py       # movielens_ctr_ab → alias: PolicyA, PolicyB
python src/ab_router_register.py    # movielens_ctr_router → alias: router

# 6) 🔎 Router 동작 데모 (A/B 배정과 score 확인)
python src/router_infer_demo.py


🎯 ABTEST 이론이 들어간 지점

Tracking:
→ A와 B(두 모델)의 성능을 동일한 데이터셋에서 비교. (정량적 평가지표 기반 → 오프라인 A/B Test 느낌)

Model Registry:
→ PolicyA, PolicyB 두 모델을 버전 태깅으로 명시. 이것이 실제로 A/B Test 환경에서 “두 정책이 병렬로 존재한다”는 구조를 반영.

Router(pyfunc):
→ 실제 A/B Test “배분” 역할.
예: user_id % 2 == 0 → PolicyA, user_id % 2 == 1 → PolicyB
여기서 ABTEST의 트래픽 분리 원리가 구현됨.

즉, ABTEST = Tracking + Registry + Router

Tracking → 성능 기록

Registry → 모델 버전 관리

Router → 실제 배분/할당



| 파일/실습 단계                                                   | MLflow 컴포넌트 관련           | 설명                                                                                                             |
|:-----------------------------------------------------------------|:-------------------------------|:-----------------------------------------------------------------------------------------------------------------|
| train_logreg.py, train_lgbm.py                                   | MLflow Tracking                | 각 모델의 파라미터, 메트릭, 아티팩트(곡선, 로그 등)를 기록. 즉 'A vs B 성능 비교 증거'를 Tracking으로 남김       |
| eval_offline_ab.py, eval_curves.py, eval_segments.py, eval_cv.py | MLflow Tracking                | 추가적인 실험 결과(ROC, PR, 세그먼트별 비교)를 로그 & 시각화 아티팩트로 저장                                     |
| prepare_movielens.py, features.py                                | MLflow Projects                | 재현 가능한 데이터 전처리/특징 엔지니어링 코드. MLProject 개념에 포함 (코드/데이터 일관성 유지)                  |
| register_models.py                                               | Model Registry                 | 학습된 모델을 중앙 저장소에 등록하고 alias (PolicyA, PolicyB)를 붙임                                             |
| ab_router_pyfunc.py, ab_router_register.py                       | MLflow Models + Model Registry | Pyfunc 커스텀 모델을 작성 → Model Registry에 router alias로 등록 → 온라인 서비스라면 REST API 엔드포인트 제공    |
| 전체 파이프라인                                                  | MLflow Models                  | 등록된 모델(LogReg, LightGBM, Router)을 패키징된 형태(pyfunc, sklearn, lightgbm)로 관리, 어디서든 불러올 수 있음 |

