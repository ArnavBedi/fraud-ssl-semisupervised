import argparse, os, json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr(y, s, title, out_png, thr=None):
    p, r, t = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)

    plt.figure()
    plt.plot(r, p, label=f"PR curve (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    if thr is not None:
        # find the closest threshold index
        import numpy as np
        # sklearn's t is len-1 vs p/r len; align to r[:-1]
        idx = (abs(t - thr)).argmin()
        plt.scatter([r[idx]], [p[idx]], s=50)
        plt.annotate(f"thr={thr:.4f}\nP={p[idx]:.2f}, R={r[idx]:.2f}",
                     (r[idx], p[idx]), textcoords="offset points", xytext=(10, -15))
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True)   # e.g., models/run_.../scores_test.csv
    ap.add_argument("--col", default="ens_score")    # which score column to plot
    ap.add_argument("--title", default="Precision–Recall (Ensemble)")
    ap.add_argument("--out", default="reports/pr_curve.png")
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.scores_csv)
    y = df["y_true"].values
    s = df[args.col].values

    plot_pr(y, s, args.title, args.out, thr=args.thr)
    print("Wrote:", args.out)
