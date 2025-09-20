import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class AE(nn.Module):
    def __init__(self, in_dim, hidden_dims=(64, 32, 8), dropout=0.05):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(h3, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_autoencoder(X_train_normals, cfg):
    Xn = torch.tensor(X_train_normals.values, dtype=torch.float32)
    ds = TensorDataset(Xn)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    model = AE(Xn.shape[1], tuple(cfg["hidden_dims"]), cfg.get("dropout", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(cfg["epochs"]):
        tot = 0
        for (xb,) in dl:
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        print(f"[AE] epoch {ep+1}/{cfg['epochs']} loss={tot/len(ds):.6f}")

    return model

@torch.no_grad()
def recon_error(model, X):
    model.eval()
    X_t = torch.tensor(X.values, dtype=torch.float32)
    R = model(X_t) - X_t
    return ((R**2).mean(dim=1)).numpy()
