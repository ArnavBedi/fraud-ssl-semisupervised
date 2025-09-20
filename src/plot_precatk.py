import argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def precision_at_k(y, s, k):
    idx = np.argsort(s)[::-1][:k]
    return float(y[idx].mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--col", default="ens_score")
    ap.add_argument("--out", default="reports/precision_at_k.png")
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    y = df["y_true"].values
    s = df[args.col].values

    ks = [50, 100, 200, 500, 1000]
    vals = [precision_at_k(y, s, k) for k in ks]

    plt.figure()
    plt.bar([str(k) for k in ks], vals)
    plt.title("Precision@K — Ensemble")
    plt.xlabel("K (top alerts)")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print("Wrote:", args.out)
