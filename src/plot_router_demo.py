# src/plot_router_demo.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

CSV_PATH = Path("artifacts/router_demo_results.csv")   # router_infer_demo.py가 생성한 파일
OUTDIR = Path("artifacts/router_viz")                  # 시각화 이미지 저장 디렉토리
OUTDIR.mkdir(parents=True, exist_ok=True)

def _savefig(fig, name):
    out = OUTDIR / name
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Saved] {out}")

def _bar(values, labels, title, fname):
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    _savefig(fig, fname)

def _hist_two(df, col, group_col, title, fname, bins=20):
    fig, ax = plt.subplots()
    for g, sub in df.groupby(group_col):
        ax.hist(sub[col], bins=bins, alpha=0.5, label=str(g))
    ax.set_title(title)
    ax.legend()
    _savefig(fig, fname)

def _roc_pr_for_group(df, label_col, score_col, group_name):
    y = df[label_col].astype(int).to_numpy()
    s = df[score_col].astype(float).to_numpy()
    # ROC
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)
    # PR
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    return (fpr, tpr, roc_auc), (rec, prec, ap), group_name

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} 가 없습니다. 먼저 router_infer_demo.py 를 실행해 CSV를 생성하세요.")

    df = pd.read_csv(CSV_PATH)

    # 컬럼 체크
    required = {"userId","movieId","assigned","score"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다. 필요: {required}, 실제: {set(df.columns)}")

    has_label = "label" in df.columns and df["label"].notnull().any()
    if not has_label:
        print("[Info] label 컬럼이 없거나 전부 결측입니다. ROC/PR 곡선은 생략됩니다.")

    # 1) A/B 배정 비율
    ratio = df["assigned"].value_counts(normalize=True)
    _bar(
        values=ratio.values.tolist(),
        labels=ratio.index.tolist(),
        title="Assignment Ratio (A vs B)",
        fname="assignment_ratio_bar.png"
    )

    # 2) A/B 평균 score
    mean_score = df.groupby("assigned")["score"].mean()
    _bar(
        values=mean_score.values.tolist(),
        labels=mean_score.index.tolist(),
        title="Mean Predicted Score by Policy",
        fname="mean_score_bar.png"
    )

    # 3) 점수 분포 히스토그램
    _hist_two(
        df, col="score", group_col="assigned",
        title="Score Distribution by Policy",
        fname="score_hist.png", bins=25
    )

    # 4) (옵션) ROC/PR 곡선 (label 있을 때만)
    if has_label:
        groups = []
        for g, sub in df.groupby("assigned"):
            # label이 전부 0이거나 1이면 곡선 계산 불가 → 스킵
            if sub["label"].nunique() < 2:
                print(f"[Warn] '{g}' 그룹의 label이 단일값입니다. ROC/PR 스킵.")
                continue
            groups.append(_roc_pr_for_group(sub, "label", "score", g))

        # ROC
        if groups:
            fig, ax = plt.subplots()
            for (fpr, tpr, roc_auc), _, name in groups:
                ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
            ax.plot([0,1], [0,1], "--", lw=1)
            ax.set_title("ROC Curve (PolicyA vs PolicyB)")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend()
            _savefig(fig, "roc_A_vs_B.png")

        # PR
        if groups:
            fig, ax = plt.subplots()
            for _, (rec, prec, ap), name in groups:
                ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
            ax.set_title("Precision-Recall Curve (PolicyA vs PolicyB)")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend()
            _savefig(fig, "pr_A_vs_B.png")

    # 5) 요약 테이블 CSV (PPT용)
    summary = pd.DataFrame({
        "metric": ["PolicyA_ratio","PolicyB_ratio","PolicyA_mean_score","PolicyB_mean_score"],
        "value": [
            float((df["assigned"]=="PolicyA").mean()),
            float((df["assigned"]=="PolicyB").mean()),
            float(df.loc[df["assigned"]=="PolicyA","score"].mean()) if (df["assigned"]=="PolicyA").any() else np.nan,
            float(df.loc[df["assigned"]=="PolicyB","score"].mean()) if (df["assigned"]=="PolicyB").any() else np.nan,
        ]
    })
    out_csv = OUTDIR / "router_viz_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

    print("\n[Done] 다음 파일들을 PPT에 삽입하세요:")
    print(f"- {OUTDIR/'assignment_ratio_bar.png'} (A/B 배정 비율)")
    print(f"- {OUTDIR/'mean_score_bar.png'} (A/B 평균 점수)")
    print(f"- {OUTDIR/'score_hist.png'} (점수 분포)")
    if has_label:
        print(f"- {OUTDIR/'roc_A_vs_B.png'} (ROC 곡선)")
        print(f"- {OUTDIR/'pr_A_vs_B.png'} (PR 곡선)")
    print(f"- {out_csv} (요약 테이블)")

if __name__ == "__main__":
    main()
