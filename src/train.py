import argparse, os, json, time
import numpy as np, pandas as pd
from joblib import dump
from data import load_data, make_splits, make_numeric_pipeline
from autoencoder import train_autoencoder, recon_error
from pu_learning import PUElkanNoto
from sklearn.metrics import average_precision_score, roc_auc_score

def evaluate(y_true, score):
    return {
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score))
    }

def main(args):
    X, y, _ = load_data(args.data, target=args.target, id_col=args.id)
    (Xtr, ytr), (Xval, yval), (Xte, yte) = make_splits(X, y, test_size=0.2, val_size=0.2, seed=42)

    pipe = make_numeric_pipeline("standard")
    Xtr_s = pd.DataFrame(pipe.fit_transform(Xtr), index=Xtr.index)
    Xval_s = pd.DataFrame(pipe.transform(Xval), index=Xval.index)
    Xte_s  = pd.DataFrame(pipe.transform(Xte),  index=Xte.index)

    normals = Xtr_s[ytr==0]
    ae_cfg = {"epochs":20, "batch_size":1024, "lr":1e-3, "hidden_dims":[64,32,8], "dropout":0.05}
    ae = train_autoencoder(normals, ae_cfg)

    ae_te = recon_error(ae, Xte_s)

    # PU learning with a small seed of labeled positives
    pos_idx = np.where(ytr==1)[0]
    rng = np.random.default_rng(42)
    labeled_pos = rng.choice(pos_idx, size=min(50, len(pos_idx)), replace=False)
    s = np.zeros(len(ytr), dtype=int)
    s[labeled_pos] = 1

    pu = PUElkanNoto(max_iter=200)
    pu.fit(Xtr_s.values, s)
    pu_te  = pu.predict_proba(Xte_s.values)[:,1]

    def rank_norm(a):
        r = pd.Series(a).rank(method='average').values
        return (r - r.min()) / (r.max() - r.min() + 1e-9)

    ens_te  = 0.5*rank_norm(ae_te)  + 0.5*rank_norm(pu_te)

    base_metrics = {
        "AE": evaluate(yte.values, ae_te),
        "PU": evaluate(yte.values, pu_te),
        "ENS": evaluate(yte.values, ens_te)
    }

    run_dir = os.path.join("models", f"run_{time.strftime('%Y%m%d_%H%M')}")
    os.makedirs(run_dir, exist_ok=True)
    dump(pipe, os.path.join(run_dir, "scaler.joblib"))
    import torch; torch.save(ae.state_dict(), os.path.join(run_dir, "ae.pt"))
    dump(pu, os.path.join(run_dir, "pu.joblib"))

    with open(os.path.join(run_dir, "metrics.json"), 'w') as f:
        json.dump({"base": base_metrics}, f, indent=2)

    print("Saved:", run_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--target', type=str, default='Class')
    parser.add_argument('--id', type=str, default=None)
    args = parser.parse_args()
    main(args)
