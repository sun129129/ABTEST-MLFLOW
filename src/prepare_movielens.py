# src/prepare_movielens.py  (100K/1M 겸용 버전으로 교체)
import zipfile, os, csv
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR  = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MovieLens 표준 장르 19개 (알파벳/공백 형태 다양성 보정 가능)
GENRES_19 = [
    "Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama",
    "Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller",
    "War","Western","(no genres listed)"
]

def parse_ml100k(zf):
    # u.data: user\titem\trating\ttimestamp
    with zf.open("ml-100k/u.data") as f:
        df = pd.read_csv(f, sep="\t", names=["userId","movieId","rating","timestamp"])
    # u.item: movieId|title|release|video|imdb|19 flags
    with zf.open("ml-100k/u.item") as f:
        raw = f.read().decode("latin-1").strip().split("\n")
        rows = [r.split("|") for r in raw]
        cols = ["movieId","title","release_date","video_release_date","imdb_url"] + [f"g{i}" for i in range(19)]
        movies = pd.DataFrame(rows, columns=cols)
        movies["movieId"] = movies["movieId"].astype(int)
        for g in [f"g{i}" for i in range(19)]:
            movies[g] = pd.to_numeric(movies[g], errors="coerce").fillna(0).astype(int)
    return df, movies[[ "movieId","title" ] + [f"g{i}" for i in range(19)]]

def parse_ml1m(zf):
    # ratings.dat: UserID::MovieID::Rating::Timestamp
    with zf.open("ml-1m/ratings.dat") as f:
        df = pd.read_csv(
            f, sep="::", engine="python", header=None,
            names=["userId", "movieId", "rating", "timestamp"]
        )

    # movies.dat: MovieID::Title::Genres
    with zf.open("ml-1m/movies.dat") as f:
        movies = pd.read_csv(
            f, sep="::", engine="python", header=None,
            names=["movieId", "title", "genres"],
            encoding="latin-1"   # 👈 여기 추가
        )

    movies["movieId"] = movies["movieId"].astype(int)
    # ... (나머지 장르 원핫 처리 부분은 그대로 유지)
    return df, movies

def main(zip_path:str=None):
    # zip: ml-100k.zip 또는 ml-1m.zip
    if zip_path is None:
        # 기본 우선순위: ml-1m.zip 있으면 사용, 없으면 ml-100k.zip
        if (DATA_DIR / "ml-1m.zip").exists():
            zip_path = str(DATA_DIR / "ml-1m.zip")
        else:
            zip_path = str(DATA_DIR / "ml-100k.zip")
    assert os.path.exists(zip_path), f"Zip not found: {zip_path}"

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if any(n.startswith("ml-1m/") for n in names):
            df, movies = parse_ml1m(zf)
        elif any(n.startswith("ml-100k/") for n in names):
            df, movies = parse_ml100k(zf)
        else:
            raise ValueError("Unsupported dataset. Put ml-1m.zip or ml-100k.zip in data/")

    # implicit label
    df["label"] = (df["rating"] >= 4).astype(int)

    # join 19-genre one-hot
    df = df.merge(movies, on="movieId", how="left")

    # 시간순 정렬 후 split
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n*0.8); valid_end = int(n*0.9)
    df_train = df.iloc[:train_end].copy()
    df_valid = df.iloc[train_end:valid_end].copy()
    df_test  = df.iloc[valid_end:].copy()

    # 저장
    df_train.to_parquet(OUT_DIR / "train.parquet", index=False)
    df_valid.to_parquet(OUT_DIR / "valid.parquet", index=False)
    df_test.to_parquet(OUT_DIR / "test.parquet", index=False)

    users  = pd.DataFrame({"userId": np.sort(df["userId"].unique())})
    movies_ids = pd.DataFrame({"movieId": np.sort(df["movieId"].unique())})
    users.to_parquet(OUT_DIR / "users.parquet", index=False)
    movies_ids.to_parquet(OUT_DIR / "movies.parquet", index=False)

    print(f"Prepared {len(df)} rows. Saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
