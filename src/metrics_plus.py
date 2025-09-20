import argparse, os, pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

def precision_at_k(y, s, k):
    idx = s.argsort()[::-1][:k]
    return float(y[idx].mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--col", default="ens_score")
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    y = df["y_true"].values
    s = df[args.col].values

    ap_score = average_precision_score(y, s)
    p, r, t = precision_recall_curve(y, s)
    # choose threshold for ~90% recall (if possible)
    target_recall = 0.90
    best_thr = t[ (r[:-1] >= target_recall) ].min() if (r[:-1] >= target_recall).any() else t[len(t)//2]

    print(f"AP (PR-AUC) on {args.col}: {ap_score:.4f}")
    for k in [50, 100, 200, 500, 1000]:
        print(f"Precision@{k}: {precision_at_k(y, s, k):.4f}")
    print(f"Suggested threshold (≈90% recall if feasible): {best_thr:.6f}")
