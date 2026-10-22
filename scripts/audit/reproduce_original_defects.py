"""Reproduce the ORIGINAL notebook's defects (see notebooks/original_bnypi.ipynb, git ad30d10).

Runs four ablations: original leaky split + broadcast loss (A), leaky split + fixed loss (B),
chronological train-only fit + broadcast loss (C), chronological + fixed loss (D), and prints
the legacy PS-L statistics. Reads data/raw/secom_telemetry_modified.csv; writes nothing.

Usage (from the repository root):  python scripts/audit/reproduce_original_defects.py
Requires imbalanced-learn (the original notebook used SMOTE). Takes about 3 minutes on CPU.
"""
import random, warnings, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
SEED = 1337
HP = dict(batch_size=64, epochs=40, lr=1e-3, weight_decay=1e-5, hidden=256, latent=32, lambda_clf=1.0)
CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "secom_telemetry_modified.csv"


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


class WaferSAE(nn.Module):  # verbatim from original cell 12
    def __init__(self, d_in, d_hid, d_lat):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(d_hid, d_hid // 2), nn.LayerNorm(d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_lat))
        self.decoder = nn.Sequential(
            nn.Linear(d_lat, d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_hid), nn.GELU(), nn.Linear(d_hid, d_in))
        self.head = nn.Linear(d_lat, 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        z = self.encoder(x)
        return self.decoder(z), self.head(z).squeeze(-1), z


def train(X, y, buggy_loss):
    seed_all()
    F = X.shape[1]
    Xt = torch.from_numpy(X).float().unsqueeze(1)          # original: [N,1,F]
    yt = torch.from_numpy(y).float()
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=HP["batch_size"], shuffle=True, generator=g, drop_last=True)
    model = WaferSAE(F, HP["hidden"], HP["latent"])
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"])
    bce = nn.BCEWithLogitsLoss()
    for ep in range(HP["epochs"]):
        model.train()
        for bx, by in loader:
            rec, logits, _ = model(bx)
            if buggy_loss:
                recon = torch.mean((rec - bx) ** 2)                 # original line: [B,F]-[B,1,F] -> [B,B,F]
            else:
                recon = torch.mean((rec - bx.flatten(1)) ** 2)      # fixed
            loss = recon + HP["lambda_clf"] * bce(logits, by)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
    return model


def scores(model, X):
    model.eval()
    with torch.no_grad():
        _, logits, _ = model(torch.from_numpy(X).float().unsqueeze(1))
    return torch.sigmoid(logits).numpy()


def recon_diag(model, X):
    """How much of each sample's own structure does the reconstruction keep?"""
    model.eval()
    xb = torch.from_numpy(X[:256]).float()
    with torch.no_grad():
        rec, _, _ = model(xb.unsqueeze(1))
    rec, xb = rec.numpy(), xb.numpy()
    per_sample_mse = float(np.mean((rec - xb) ** 2))
    mean_pred_mse = float(np.mean((xb.mean(0, keepdims=True) - xb) ** 2))   # always predicting the batch mean
    spread_ratio = float(rec.std(0).mean() / xb.std(0).mean())              # cross-sample spread kept
    return dict(per_sample_mse=round(per_sample_mse, 4), mse_of_batch_mean_predictor=round(mean_pred_mse, 4),
                cross_sample_spread_ratio=round(spread_ratio, 4))


def report(name, y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(setting=name, n=len(y), prevalence=round(float(y.mean()), 4),
               F1=round(f1_score(y, pred, zero_division=0), 4), precision=round(precision_score(y, pred, zero_division=0), 4),
               recall=round(recall_score(y, pred, zero_division=0), 4), ROC_AUC=round(roc_auc_score(y, p), 4),
               AP=round(average_precision_score(y, p), 4), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    print(json.dumps(out), flush=True)
    return out


df = pd.read_csv(CSV)
S = [c for c in df.columns if c.startswith("sensor_")]
X_raw = df[S].to_numpy(dtype=np.float64)
y_raw = (df["label"].to_numpy() == 1).astype(np.int64)
print("defect rate", round(y_raw.mean(), 4), int(y_raw.sum()), "of", len(y_raw))

# ---------- Invariant evidence (legacy PS-L) ----------
leg = df["sensor_0"].eq(0.0).to_numpy()
s1 = df["sensor_1"].to_numpy()
print("legacy rows", int(leg.sum()), "| defects among legacy", int(y_raw[leg].sum()),
      "| codes", pd.Series(s1[leg]).value_counts().sort_index().to_dict())
non = s1[~leg]; non = non[~np.isnan(non)]
print("sensor_1 non-legacy mean/std", round(non.mean(), 1), round(non.std(), 1))
allv = s1[~np.isnan(s1)]
print("sensor_1 all rows mean/std (what the original scaler used)", round(allv.mean(), 1), round(allv.std(), 1))
for code in (100, 200, 300):
    print(f"  code {code}: z-score under original scaler = {(code - allv.mean()) / allv.std():.1f}; "
          f"under non-legacy stats = {(code - non.mean()) / non.std():.1f}")
s0 = df["sensor_0"].to_numpy(); s0n = s0[(~leg) & ~np.isnan(s0)]
print("sensor_0 non-legacy mean/std", round(s0n.mean(), 3), round(s0n.std(), 3), "| legacy value 0.0 z =",
      round((0 - s0n.mean()) / s0n.std(), 1))

# ---------- A / B: original leaky pipeline ----------
imp = SimpleImputer(strategy="mean", keep_empty_features=True)
Xs = StandardScaler().fit_transform(imp.fit_transform(X_raw))                   # fit on ALL rows
Xb, yb = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=SEED).fit_resample(Xs, y_raw)   # before split
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.20, stratify=yb, random_state=SEED)
print("after SMOTE", Xb.shape, "test defect rate", round(float(yte.mean()), 3))
res = []
mA = train(Xtr, ytr, buggy_loss=True)
res.append(report("A original (leaky split + broadcast loss), balanced SMOTE test, thr 0.5", yte, scores(mA, Xte)))
print("  recon diag A:", recon_diag(mA, Xtr))
mB = train(Xtr, ytr, buggy_loss=False)
res.append(report("B leaky split + FIXED loss, balanced SMOTE test, thr 0.5", yte, scores(mB, Xte)))
print("  recon diag B:", recon_diag(mB, Xtr))

# ---------- C / D: causal pipeline, natural prevalence ----------
d2 = df.copy(); d2["timestamp"] = pd.to_datetime(d2["timestamp"])
d2 = d2.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
n = len(d2); a = int(n * 0.70); b = a + int(n * 0.15)
Xr = d2[S].to_numpy(dtype=np.float64); yr = (d2["label"].to_numpy() == 1).astype(np.int64)
imp2 = SimpleImputer(strategy="mean", keep_empty_features=True).fit(Xr[:a])
sc2 = StandardScaler().fit(imp2.transform(Xr[:a]))
f = lambda M: np.nan_to_num(sc2.transform(imp2.transform(M)))
Xc_tr, Xc_te, yc_tr, yc_te = f(Xr[:a]), f(Xr[b:]), yr[:a], yr[b:]
mC = train(Xc_tr, yc_tr, buggy_loss=True)
res.append(report("C chronological, train-only fit, no SMOTE, broadcast loss, natural test, thr 0.5", yc_te, scores(mC, Xc_te)))
mD = train(Xc_tr, yc_tr, buggy_loss=False)
res.append(report("D chronological, train-only fit, no SMOTE, FIXED loss, natural test, thr 0.5", yc_te, scores(mD, Xc_te)))
print("  recon diag D:", recon_diag(mD, Xc_tr))
