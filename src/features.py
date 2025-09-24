# src/features.py  (완전 교체본)
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
GENRE_COLS = [f"g{i}" for i in range(19)]  # prepare_movielens가 만드는 19개 장르

def load_split(split: str) -> pd.DataFrame:
    """Load split parquet file: 'train' | 'valid' | 'test'"""
    df = pd.read_parquet(DATA_DIR / f"{split}.parquet")
    return df

def _ensure_genre_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """장르 컬럼을 숫자형으로 강제 (결측은 0)"""
    cols = [c for c in GENRE_COLS if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if c.startswith("g")]
    if cols:
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.int8)
    return df

def build_logreg_features(df: pd.DataFrame, enc: OneHotEncoder = None, fit: bool = True):
    """
    입력: df(columns: userId, movieId, label, g0..g18)
    출력: X(희소 CSR), y(ndarray), enc(OneHotEncoder)
      - userId, movieId -> OneHot (희소)
      - genres g0..g18 -> float32 CSR로 변환 후 hstack
    """
    # 1) 장르 숫자형 보장
    df = _ensure_genre_numeric(df)

    # 2) ID 피처: 카테고리 캐스팅(안정적 OHE)
    X_id_base = df[["userId", "movieId"]].copy()
    X_id_base["userId"] = X_id_base["userId"].astype("int64").astype("category")
    X_id_base["movieId"] = X_id_base["movieId"].astype("int64").astype("category")

    # 3) OneHotEncoder (sklearn 1.5: sparse_output 사용)
    if enc is None:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        fit = True
    if fit:
        X_id = enc.fit_transform(X_id_base)   # scipy.sparse CSR/CSC
    else:
        X_id = enc.transform(X_id_base)

    # 4) 장르 -> float32 ndarray -> CSR
    genre_cols = [c for c in GENRE_COLS if c in df.columns] or [c for c in df.columns if c.startswith("g")]
    if genre_cols:
        X_genres_np = df[genre_cols].to_numpy(dtype=np.float32, copy=False)  # 👈 dtype 고정
        X_genres = sp.csr_matrix(X_genres_np)
        # 5) 희소 결합
        X = sp.hstack([X_id, X_genres], format="csr")
    else:
        X = X_id

    y = df["label"].astype(np.int64).to_numpy(copy=False)
    return X, y, enc
