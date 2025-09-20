import argparse, os, json
import numpy as np, pandas as pd
import joblib, torch
from data import load_data, make_splits, make_numeric_pipeline
from autoencoder import AE, recon_error
from pu_learning import PUElkanNoto

def rank_norm(a):
    r = pd.Series(a).rank(method='average').values
    return (r - r.min()) / (r.max() - r.min() + 1e-9)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="Class")
    ap.add_argument("--id", default="Time")
    args = ap.parse_args()

    # Load data & split (same seed as train.py)
    X, y, _ = load_data(args.data, target=args.target, id_col=args.id)
    (Xtr, ytr), (Xval, yval), (Xte, yte) = make_splits(X, y, test_size=0.2, val_size=0.2, seed=42)

    # Load artifacts
    scaler = joblib.load(os.path.join(args.run_dir, "scaler.joblib"))
    Xte_s = pd.DataFrame(scaler.transform(Xte), index=Xte.index)

    state = torch.load(os.path.join(args.run_dir, "ae.pt"), map_location="cpu")
    ae = AE(in_dim=Xte_s.shape[1]); ae.load_state_dict(state)

    pu = joblib.load(os.path.join(args.run_dir, "pu.joblib"))

    # Scores
    ae_te = recon_error(ae, Xte_s)
    pu_te = pu.predict_proba(Xte_s.values)[:, 1]
    ens_te = 0.5*rank_norm(ae_te) + 0.5*rank_norm(pu_te)

    out = pd.DataFrame({
        "y_true": yte.values,
        "ae_score": ae_te,
        "pu_prob": pu_te,
        "ens_score": ens_te
    }, index=Xte.index)

    csv_path = os.path.join(args.run_dir, "scores_test.csv")
    out.to_csv(csv_path, index=False)
    print("Wrote:", csv_path)
