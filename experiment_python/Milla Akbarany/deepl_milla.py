"""
Pipeline MLP untuk Datathon (dataset A / B / C).

Pemakaian (dari folder repo, sejajar dengan folder Dataset/):
    python run_dataset.py A --team T017
    python run_dataset.py B --team T017
    python run_dataset.py C --team T017

Alur per dataset:
  1. Split stratified 80/20 dari train.csv (seed tetap).
  2. Imputer + scaler di-fit HANYA pada 80% training internal.
  3. Jalankan beberapa eksperimen terkontrol (depth, dropout, width), semua dicatat ke logs/experiments.csv.
  4. Pilih konfigurasi terbaik berdasarkan val accuracy (tie -> val loss lebih kecil).
  5. Simpan kurva loss, confusion matrix validasi, dan classification report.
  6. Latih ulang di seluruh train.csv dengan konfigurasi terpilih (jumlah epoch = best epoch), lalu tulis submission.
"""
import argparse
import os
import platform
import random
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # untuk outputs/ dan logs/ (di folder skrip)
DATA_ROOT = BASE_DIR                                   # untuk mencari folder Dataset/
while not os.path.isdir(os.path.join(DATA_ROOT, "Dataset")) and os.path.dirname(DATA_ROOT) != DATA_ROOT:
    DATA_ROOT = os.path.dirname(DATA_ROOT)

# Eksperimen terkontrol: ubah satu faktor per eksperimen dari "base".
EXPERIMENTS = [
    dict(name="base", hidden=[128, 64], dropout=0.2),
    dict(name="depth1", hidden=[128], dropout=0.2),
    dict(name="depth3", hidden=[256, 128, 64], dropout=0.2),
    dict(name="wide", hidden=[512, 256], dropout=0.2),
    dict(name="drop0", hidden=[128, 64], dropout=0.0),
    dict(name="drop4", hidden=[128, 64], dropout=0.4),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(d_in, n_cls, hidden, dropout):
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_cls))
    return nn.Sequential(*layers)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, X, y, loss_fn):
    model.eval()
    logits = model(X)
    return loss_fn(logits, y).item(), logits.argmax(1)


def fit(model, Xtr, ytr, Xva, yva, lr, wd, bs, max_epochs, patience, epochs=None):
    """Kalau Xva ada: early stopping pada val loss, bobot terbaik dipulihkan.
    Kalau Xva None: latih tepat `epochs` epoch (dipakai untuk refit di seluruh train)."""
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = Xtr.shape[0]
    hist = {"train_loss": [], "val_loss": []}
    best_val, best_epoch, best_state, bad = float("inf"), 0, None, 0
    total = epochs if Xva is None else max_epochs

    for ep in range(1, total + 1):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        run_loss, seen = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:  # BatchNorm butuh minimal 2 sampel
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            seen += len(idx)
        hist["train_loss"].append(run_loss / seen)

        if Xva is not None:
            vloss, _ = evaluate(model, Xva, yva, loss_fn)
            hist["val_loss"].append(vloss)
            if vloss < best_val - 1e-5:
                best_val, best_epoch, bad = vloss, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

    if Xva is not None and best_state is not None:
        model.load_state_dict(best_state)
    return hist, best_epoch


def save_loss_curve(hist, best_epoch, title, path):
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.axvline(best_epoch - 1, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["A", "B", "C"])
    ap.add_argument("--team", default="T017", help="ID tim untuk submission")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no-refit", action="store_true", help="lewati retrain di seluruh train.csv")
    args = ap.parse_args()
    D = args.dataset

    out_dir = os.path.join(BASE_DIR, "outputs")
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU {platform.processor()}"
    print(f"Dataset {D} | device: {hw}")

    # ---------- Load ----------
    data_dir = os.path.join(DATA_ROOT, "Dataset", f"dataset_{D}")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    feats = [c for c in train.columns if c not in ("sample_id", "target")]  # sample_id BUKAN fitur
    classes = np.sort(train["target"].unique())
    cmap = {c: i for i, c in enumerate(classes)}
    X_all = train[feats].values.astype(np.float32)
    y_all = train["target"].map(cmap).values
    X_test_raw = test[feats].values.astype(np.float32)
    n_cls = len(classes)

    # ---------- Split + preprocessing (fit hanya di training internal) ----------
    Xtr_raw, Xva_raw, ytr, yva = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED
    )
    imp = SimpleImputer(strategy="median").fit(Xtr_raw)
    sc = StandardScaler().fit(imp.transform(Xtr_raw))

    def prep(X):
        return sc.transform(imp.transform(X)).astype(np.float32)

    to_t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    Xtr, Xva = to_t(prep(Xtr_raw)), to_t(prep(Xva_raw))
    ytr_t, yva_t = to_t(ytr, torch.long), to_t(yva, torch.long)
    loss_fn = nn.CrossEntropyLoss()

    # ---------- Eksperimen ----------
    results = []
    log_path = os.path.join(log_dir, "experiments.csv")
    for cfg in EXPERIMENTS:
        set_seed(SEED)
        model = make_model(Xtr.shape[1], n_cls, cfg["hidden"], cfg["dropout"]).to(device)
        t0 = time.time()
        hist, best_ep = fit(model, Xtr, ytr_t, Xva, yva_t, args.lr, args.wd, args.bs,
                            args.max_epochs, args.patience)
        secs = time.time() - t0
        vloss, vpred = evaluate(model, Xva, yva_t, loss_fn)
        vacc = (vpred == yva_t).float().mean().item()
        print(f"[{D}/{cfg['name']:7s}] val_acc={vacc:.4f} val_loss={vloss:.4f} "
              f"best_ep={best_ep} params={n_params(model)} time={secs:.1f}s")
        save_loss_curve(hist, best_ep, f"Dataset {D} - {cfg['name']}",
                        os.path.join(out_dir, f"{D}_{cfg['name']}_loss.png"))
        row = dict(timestamp=datetime.now().isoformat(timespec="seconds"), dataset=D, exp=cfg["name"],
                   hidden=str(cfg["hidden"]), dropout=cfg["dropout"], optimizer="AdamW", lr=args.lr,
                   weight_decay=args.wd, batch_size=args.bs, max_epochs=args.max_epochs,
                   early_stop_patience=args.patience, best_epoch=best_ep, seed=SEED,
                   params=n_params(model), val_loss=round(vloss, 5), val_acc=round(vacc, 5),
                   train_time_s=round(secs, 1), hardware=hw, phase="validation")
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
        results.append((cfg, vacc, vloss, best_ep, vpred.cpu().numpy()))

    # ---------- Pilih terbaik ----------
    best_cfg, best_acc, best_loss, best_ep, best_pred = sorted(results, key=lambda r: (-r[1], r[2]))[0]
    print(f"\n>>> Terbaik: {best_cfg['name']} {best_cfg['hidden']} dropout={best_cfg['dropout']} "
          f"val_acc={best_acc:.4f} best_epoch={best_ep}")

    cm = confusion_matrix(yva, best_pred)
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d")
    plt.title(f"Dataset {D} - confusion matrix validasi ({best_cfg['name']})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{D}_confusion.png"), dpi=150)
    plt.close()
    rep = classification_report(yva, best_pred, target_names=[str(c) for c in classes], digits=4)
    with open(os.path.join(log_dir, f"{D}_classification_report.txt"), "w") as f:
        f.write(f"Konfigurasi: {best_cfg}\n\n{rep}\nConfusion matrix:\n{cm}\n")
    print(rep)

    if args.no_refit:
        print("Refit dilewati (--no-refit).")
        return

    # ---------- Refit di seluruh train.csv ----------
    set_seed(SEED)
    imp_f = SimpleImputer(strategy="median").fit(X_all)
    sc_f = StandardScaler().fit(imp_f.transform(X_all))
    prep_f = lambda X: sc_f.transform(imp_f.transform(X)).astype(np.float32)
    Xf, yf = to_t(prep_f(X_all)), to_t(y_all, torch.long)
    Xt = to_t(prep_f(X_test_raw))

    model = make_model(Xf.shape[1], n_cls, best_cfg["hidden"], best_cfg["dropout"]).to(device)
    t0 = time.time()
    fit(model, Xf, yf, None, None, args.lr, args.wd, args.bs, args.max_epochs, args.patience,
        epochs=max(best_ep, 1))
    secs = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred_idx = model(Xt).argmax(1).cpu().numpy()
    pred_label = classes[pred_idx]
    torch.save(model.state_dict(), os.path.join(out_dir, f"{D}_final.pt"))

    sub = pd.DataFrame({"sample_id": test["sample_id"], "target": pred_label.astype(int)})
    # validasi format submission
    assert list(sub.columns) == ["sample_id", "target"]
    assert sub["sample_id"].is_unique and len(sub) == len(test)
    assert not sub.isna().any().any() and set(sub["target"]).issubset(set(classes))
    sub_path = os.path.join(out_dir, f"{args.team}_{D}_01.csv")
    sub.to_csv(sub_path, index=False)

    row = dict(timestamp=datetime.now().isoformat(timespec="seconds"), dataset=D, exp=best_cfg["name"] + "_refit",
               hidden=str(best_cfg["hidden"]), dropout=best_cfg["dropout"], optimizer="AdamW", lr=args.lr,
               weight_decay=args.wd, batch_size=args.bs, max_epochs=best_ep, early_stop_patience="-",
               best_epoch=best_ep, seed=SEED, params=n_params(model), val_loss="-", val_acc="-",
               train_time_s=round(secs, 1), hardware=hw, phase="refit_full_train")
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
    print(f"\nSubmission tersimpan & lolos validasi: {sub_path}")
    print(sub["target"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
