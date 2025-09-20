""" # --- begin: robust imports so unpickling works ---
import os, sys
import streamlit as st
import pandas as pd
import joblib
import torch

# Make the src/ folder importable as top-level modules
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Import using top-level names to match what was pickled during training
from autoencoder import AE, recon_error   # not "from src.autoencoder"
import pu_learning  # registers the module name for joblib pickle
# --- end: robust imports ---

from src.autoencoder import AE, recon_error

st.set_page_config(page_title="Fraud Risk Scoring (Semi-Supervised)", layout="wide")
st.title("Fraud Risk Scoring (AE + PU + Ensemble)")

# Path to your saved model run
run_dir = st.text_input("Models run_dir", "models/run_20250920_1813")
uploaded = st.file_uploader("Upload transactions CSV (same schema as training)", type=["csv"])

def rank_norm(a):
    rnk = pd.Series(a).rank(method="average").values
    return (rnk - rnk.min()) / (rnk.max() - rnk.min() + 1e-9)

if uploaded and os.path.isdir(run_dir):
    df = pd.read_csv(uploaded)

    # Load artifacts
    scaler = joblib.load(os.path.join(run_dir, "scaler.joblib"))
    Xs = pd.DataFrame(
        scaler.transform(df.drop(columns=["Class"], errors="ignore")),
        index=df.index
    )

    state = torch.load(os.path.join(run_dir, "ae.pt"), map_location="cpu")
    ae = AE(in_dim=Xs.shape[1])
    ae.load_state_dict(state)

    pu = joblib.load(os.path.join(run_dir, "pu.joblib"))

    # Scores
    ae_s = recon_error(ae, Xs)
    pu_s = pu.predict_proba(Xs.values)[:, 1]
    ens = 0.5*rank_norm(ae_s) + 0.5*rank_norm(pu_s)

    out = df.copy()
    out["ae_score"] = ae_s
    out["pu_prob"] = pu_s
    out["ensemble"] = ens

    st.subheader("Top Alerts (highest risk)")
    st.dataframe(out.sort_values("ensemble", ascending=False).head(50), height=420)
 """
# --- imports & setup (robust so unpickling works) ---
import os, sys
import streamlit as st
import pandas as pd
import joblib
import torch

# Make src/ importable at top-level (so pu_learning loads from pickle)
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from autoencoder import AE, recon_error   # top-level import (matches training)
import pu_learning  # registers module name for joblib

# ML metrics
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Risk Scoring (Semi-Supervised)", layout="wide")
st.title("Fraud Risk Scoring (AE + PU + Ensemble)")

# --- inputs ---
run_dir = st.text_input("Models run_dir", "models/run_20250920_1813")
uploaded = st.file_uploader("Upload transactions CSV (same schema as training)", type=["csv"])

def rank_norm(a):
    rnk = pd.Series(a).rank(method="average").values
    return (rnk - rnk.min()) / (rnk.max() - rnk.min() + 1e-9)

def precision_at_k(y, s, k):
    idx = np.argsort(s)[::-1][:k]
    return float(y[idx].mean())

# --- main ---
if uploaded and os.path.isdir(run_dir):
    df = pd.read_csv(uploaded)

    # Load artifacts
    scaler = joblib.load(os.path.join(run_dir, "scaler.joblib"))
    Xs = pd.DataFrame(
        scaler.transform(df.drop(columns=["Class"], errors="ignore")),
        index=df.index
    )

    state = torch.load(os.path.join(run_dir, "ae.pt"), map_location="cpu")
    ae = AE(in_dim=Xs.shape[1]); ae.load_state_dict(state)

    pu = joblib.load(os.path.join(run_dir, "pu.joblib"))

    # Scores
    ae_s = recon_error(ae, Xs)
    pu_s = pu.predict_proba(Xs.values)[:, 1]
    ens = 0.5*rank_norm(ae_s) + 0.5*rank_norm(pu_s)

    out = df.copy()
    out["ae_score"] = ae_s
    out["pu_prob"]  = pu_s
    out["ensemble"] = ens

    # ======= UI: Top alerts table =======
    st.subheader("Top Alerts (highest risk)")
    st.dataframe(out.sort_values("ensemble", ascending=False).head(50), height=420)

    # ======= UI: Metrics & Plots =======
    st.subheader("Model Metrics on Your Upload")

    # If the upload has labels, compute true metrics; otherwise show curves without truth
    has_labels = "Class" in df.columns
    if has_labels:
        y = df["Class"].astype(int).values
        s = ens

        # Precision–Recall curve
        p, r, t = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)

        col1, col2, col3 = st.columns(3)
        col1.metric("PR-AUC (AP)", f"{ap:.3f}")
        col2.metric("Precision@50", f"{precision_at_k(y, s, 50):.3f}")
        col3.metric("Precision@100", f"{precision_at_k(y, s, 100):.3f}")

        # Threshold slider (maps to closest PR point)
        st.caption("Adjust decision threshold to explore precision/recall trade-off.")
        thr = st.slider("Score threshold", float(s.min()), float(s.max()), float(np.quantile(s, 0.97)))
        # Find nearest threshold index (sklearn returns t of length len(p)-1)
        if len(t) > 0:
            idx = int(np.argmin(np.abs(t - thr)))
            pr_here = float(p[idx]); rc_here = float(r[idx])
        else:
            pr_here, rc_here = float("nan"), float("nan")

        # Plot PR
        fig = plt.figure()
        plt.plot(r, p, label=f"PR curve (AP={ap:.3f})")
        if len(t) > 0:
            plt.scatter([r[idx]], [p[idx]], s=50)
            plt.annotate(f"thr≈{thr:.4f}\nP={pr_here:.2f}, R={rc_here:.2f}",
                         (r[idx], p[idx]), textcoords="offset points", xytext=(10, -15))
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (Ensemble)")
        plt.grid(True); plt.legend(loc="lower left"); plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Precision@K bar chart
        ks = [50, 100, 200, 500, 1000]
        vals = [precision_at_k(y, s, k) for k in ks]
        fig2 = plt.figure()
        plt.bar([str(k) for k in ks], vals)
        plt.title("Precision@K — Ensemble"); plt.xlabel("K (top alerts)"); plt.ylabel("Precision")
        plt.ylim(0, 1); plt.grid(axis="y"); plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
    else:
        st.info("No labels (`Class`) found in your CSV — showing top alerts only. Add a `Class` column (0/1) to compute PR-AUC and Precision@K.")

    # ======= Download scored CSV =======
    scored = out.sort_values("ensemble", ascending=False)
    st.download_button(
        "Download scored CSV",
        data=scored.to_csv(index=False).encode(),
        file_name="scored_transactions.csv",
        mime="text/csv"
    )
else:
    st.info("Enter a valid `run_dir` and upload a CSV to see results.")
