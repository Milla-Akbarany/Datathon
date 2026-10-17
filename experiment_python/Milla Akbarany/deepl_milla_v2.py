"""
deepl_milla_v2.py - MLP untuk Datathon, satu file untuk semua dataset.

Pemakaian (dari folder tempat file ini berada):
    python deepl_milla_v2.py A --team T017 --attempt 02

Perbedaan dari v1:
  - Pengaturan per dataset ada di DATASET_CFG (sekarang baru A; B dan C tinggal tambah blok).
  - Epoch maksimal lebih panjang (v1: best_epoch hampir menyentuh batas 200) dan ada
    eksperimen learning rate / batch size, selain depth, lebar, dan dropout.
  - Nama output mengikuti --attempt, mis. T017_A_02.csv. Skrip MENOLAK menimpa file yang sudah ada
    (cek di awal, sebelum training), kecuali dengan --force.
  - Kalau run terputus, eksperimen yang sudah selesai tidak diulang (cache di logs_v2/).
    Pakai --restart untuk mengulang semuanya dari awal.
  - Semua hasil v2 disimpan di outputs_v2/ dan logs_v2/, terpisah dari v1.

Alur: split stratified 80/20 -> imputer+scaler di-fit hanya di 80% training internal ->
eksperimen terkontrol (satu faktor berubah per eksperimen dari 'base') -> pilih val accuracy
tertinggi (tie: val loss lebih kecil) -> latih ulang di seluruh train.csv sebanyak best_epoch epoch
-> tulis submission dan validasi formatnya.
"""
import argparse
import json
import os
import platform
import random
import shutil
import sys
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

# ---------------------------------------------------------------------------
# Pengaturan per dataset. Untuk B dan C: salin blok "A", ubah isinya, beri kunci "B" / "C".
# Eksperimen: tiap entri hanya mengubah SATU faktor dari 'base'.
# Kunci opsional per eksperimen: lr, wd, bs (menimpa nilai default dataset).
# ---------------------------------------------------------------------------
DATASET_CFG = {
    "A": dict(
        lr=1e-3, wd=1e-4, bs=256, max_epochs=600, patience=30,
        experiments=[
            dict(name="base", hidden=[128, 64], dropout=0.0),
            dict(name="lr3e-3", hidden=[128, 64], dropout=0.0, lr=3e-3),
            dict(name="bs128", hidden=[128, 64], dropout=0.0, bs=128),
            dict(name="wide", hidden=[256, 128], dropout=0.0),
            dict(name="deep3", hidden=[256, 128, 64], dropout=0.0),
            dict(name="drop1", hidden=[128, 64], dropout=0.1),
        ],
    ),
    # "B": dict(lr=..., wd=..., bs=..., max_epochs=..., patience=..., experiments=[...]),
    # "C": dict(lr=..., wd=..., bs=..., max_epochs=..., patience=..., experiments=[...]),
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # outputs_v2/ dan logs_v2/ dibuat di sini
DATA_ROOT = BASE_DIR                                    # cari folder Dataset/ ke atas
while not os.path.isdir(os.path.join(DATA_ROOT, "Dataset")) and os.path.dirname(DATA_ROOT) != DATA_ROOT:
    DATA_ROOT = os.path.dirname(DATA_ROOT)


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
    """Xva ada: early stopping pada val loss, bobot terbaik dipulihkan.
    Xva None: latih tepat `epochs` epoch (refit di seluruh train)."""
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
        if ep % 25 == 0:
            msg = f"    epoch {ep}/{total} train_loss={hist['train_loss'][-1]:.4f}"
            if Xva is not None:
                msg += f" val_loss={hist['val_loss'][-1]:.4f} (best {best_val:.4f} @ {best_epoch})"
            print(msg, flush=True)

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


def resolve(dcfg, exp):
    """Gabungkan default dataset dengan override per eksperimen."""
    hp = {k: dcfg[k] for k in ("lr", "wd", "bs", "max_epochs", "patience")}
    hp.update({k: v for k, v in exp.items() if k in hp})
    hp["hidden"], hp["dropout"] = list(exp["hidden"]), exp["dropout"]
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--team", default="T017")
    ap.add_argument("--attempt", default="02", help="nomor attempt untuk nama file, mis. 02")
    ap.add_argument("--force", action="store_true", help="boleh menimpa CSV submission yang sudah ada")
    ap.add_argument("--restart", action="store_true", help="abaikan cache, ulangi semua eksperimen")
    ap.add_argument("--no-refit", action="store_true", help="hanya eksperimen validasi, tanpa CSV submission")
    args = ap.parse_args()

    D = args.dataset.upper()
    if D not in DATASET_CFG:
        sys.exit(f"Dataset {D} belum diisi di DATASET_CFG (baru: {', '.join(DATASET_CFG)}). "
                 f"Tambahkan blok pengaturannya dulu.")
    dcfg = DATASET_CFG[D]

    out_dir = os.path.join(BASE_DIR, "outputs_v2")
    log_dir = os.path.join(BASE_DIR, "logs_v2")
    cache_dir = os.path.join(log_dir, f"cache_{D}_{args.attempt}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Pengaman: cek lebih dulu, sebelum training berjam-jam.
    sub_path = os.path.join(out_dir, f"{args.team}_{D}_{args.attempt}.csv")
    if not args.no_refit and os.path.exists(sub_path) and not args.force:
        sys.exit(f"{sub_path} sudah ada. Ganti --attempt (mis. 03) atau pakai --force untuk menimpa.")
    if args.restart and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU {platform.processor()}"
    print(f"Dataset {D} | attempt {args.attempt} | device: {hw}")
    print(f"Data   : {os.path.join(DATA_ROOT, 'Dataset', f'dataset_{D}')}")
    print(f"Output : {out_dir}")

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
    prep = lambda X: sc.transform(imp.transform(X)).astype(np.float32)
    to_t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    Xtr, Xva = to_t(prep(Xtr_raw)), to_t(prep(Xva_raw))
    ytr_t, yva_t = to_t(ytr, torch.long), to_t(yva, torch.long)
    loss_fn = nn.CrossEntropyLoss()

    # ---------- Eksperimen ----------
    log_path = os.path.join(log_dir, "experiments_v2.csv")
    results = []
    for exp in dcfg["experiments"]:
        hp = resolve(dcfg, exp)
        jpath = os.path.join(cache_dir, f"{exp['name']}.json")
        npath = os.path.join(cache_dir, f"{exp['name']}_valpred.npy")

        if os.path.exists(jpath) and os.path.exists(npath):
            cached = json.load(open(jpath))
            if cached["hp"] == hp:
                print(f"[{D}/{exp['name']:7s}] SKIP (sudah selesai) val_acc={cached['val_acc']:.4f} "
                      f"best_ep={cached['best_epoch']}")
                results.append((exp, hp, cached["val_acc"], cached["val_loss"], cached["best_epoch"],
                                np.load(npath)))
                continue

        print(f"[{D}/{exp['name']:7s}] mulai: {hp}", flush=True)
        set_seed(SEED)
        model = make_model(Xtr.shape[1], n_cls, hp["hidden"], hp["dropout"]).to(device)
        t0 = time.time()
        hist, best_ep = fit(model, Xtr, ytr_t, Xva, yva_t, hp["lr"], hp["wd"], hp["bs"],
                            hp["max_epochs"], hp["patience"])
        secs = time.time() - t0
        vloss, vpred = evaluate(model, Xva, yva_t, loss_fn)
        vacc = (vpred == yva_t).float().mean().item()
        print(f"[{D}/{exp['name']:7s}] val_acc={vacc:.4f} val_err={1 - vacc:.4f} val_loss={vloss:.4f} "
              f"best_ep={best_ep} epoch_jalan={len(hist['train_loss'])} params={n_params(model)} time={secs:.1f}s")

        save_loss_curve(hist, best_ep, f"Dataset {D} - {exp['name']} (v2)",
                        os.path.join(out_dir, f"{D}_{args.attempt}_{exp['name']}_loss.png"))
        row = dict(timestamp=datetime.now().isoformat(timespec="seconds"), dataset=D, attempt=args.attempt,
                   exp=exp["name"], hidden=str(hp["hidden"]), dropout=hp["dropout"], optimizer="AdamW",
                   lr=hp["lr"], weight_decay=hp["wd"], batch_size=hp["bs"], max_epochs=hp["max_epochs"],
                   early_stop_patience=hp["patience"], best_epoch=best_ep, epochs_run=len(hist["train_loss"]),
                   seed=SEED, params=n_params(model), val_loss=round(vloss, 5), val_acc=round(vacc, 5),
                   val_err=round(1 - vacc, 5), train_time_s=round(secs, 1), hardware=hw, phase="validation")
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

        vpred_np = vpred.cpu().numpy()
        np.save(npath, vpred_np)
        json.dump(dict(hp=hp, val_acc=vacc, val_loss=vloss, best_epoch=best_ep), open(jpath, "w"))
        results.append((exp, hp, vacc, vloss, best_ep, vpred_np))

    # ---------- Ringkasan + pilih terbaik ----------
    print("\nRingkasan eksperimen (urut val accuracy):")
    for exp, hp, vacc, vloss, bep, _ in sorted(results, key=lambda r: (-r[2], r[3])):
        print(f"  {exp['name']:8s} val_acc={vacc:.4f} val_err={1 - vacc:.4f} val_loss={vloss:.4f} best_ep={bep}")
    best_exp, best_hp, best_acc, best_loss, best_ep, best_pred = sorted(results, key=lambda r: (-r[2], r[3]))[0]
    print(f"\n>>> Terbaik {D}: {best_exp['name']} {best_hp['hidden']} dropout={best_hp['dropout']} "
          f"lr={best_hp['lr']} bs={best_hp['bs']} val_acc={best_acc:.4f} best_epoch={best_ep}")

    cm = confusion_matrix(yva, best_pred)
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d")
    plt.title(f"Dataset {D} - confusion matrix validasi ({best_exp['name']}, v2)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{D}_{args.attempt}_confusion.png"), dpi=150)
    plt.close()
    rep = classification_report(yva, best_pred, target_names=[str(c) for c in classes], digits=4)
    with open(os.path.join(log_dir, f"{D}_{args.attempt}_classification_report.txt"), "w") as f:
        f.write(f"Konfigurasi: {best_exp['name']} {best_hp}\n\n{rep}\nConfusion matrix:\n{cm}\n")
    print(rep)

    if args.no_refit:
        print("Refit dilewati (--no-refit).")
        return

    # ---------- Refit di seluruh train.csv ----------
    print(f"Refit di seluruh train.csv: {best_ep} epoch ...", flush=True)
    set_seed(SEED)
    imp_f = SimpleImputer(strategy="median").fit(X_all)
    sc_f = StandardScaler().fit(imp_f.transform(X_all))
    prep_f = lambda X: sc_f.transform(imp_f.transform(X)).astype(np.float32)
    Xf, yf = to_t(prep_f(X_all)), to_t(y_all, torch.long)
    Xt = to_t(prep_f(X_test_raw))

    model = make_model(Xf.shape[1], n_cls, best_hp["hidden"], best_hp["dropout"]).to(device)
    t0 = time.time()
    fit(model, Xf, yf, None, None, best_hp["lr"], best_hp["wd"], best_hp["bs"],
        best_hp["max_epochs"], best_hp["patience"], epochs=max(best_ep, 1))
    secs = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred_idx = model(Xt).argmax(1).cpu().numpy()
    torch.save(model.state_dict(), os.path.join(out_dir, f"{D}_{args.attempt}_final.pt"))

    sub = pd.DataFrame({"sample_id": test["sample_id"], "target": classes[pred_idx].astype(int)})
    assert list(sub.columns) == ["sample_id", "target"]
    assert sub["sample_id"].is_unique and len(sub) == len(test)
    assert not sub.isna().any().any() and set(sub["target"]).issubset(set(classes.astype(int)))
    sub.to_csv(sub_path, index=False)

    row = dict(timestamp=datetime.now().isoformat(timespec="seconds"), dataset=D, attempt=args.attempt,
               exp=best_exp["name"] + "_refit", hidden=str(best_hp["hidden"]), dropout=best_hp["dropout"],
               optimizer="AdamW", lr=best_hp["lr"], weight_decay=best_hp["wd"], batch_size=best_hp["bs"],
               max_epochs=best_ep, early_stop_patience="-", best_epoch=best_ep, epochs_run=best_ep, seed=SEED,
               params=n_params(model), val_loss="-", val_acc="-", val_err="-", train_time_s=round(secs, 1),
               hardware=hw, phase="refit_full_train")
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
    print(f"\nSubmission tersimpan & lolos validasi: {sub_path}")
    print(sub["target"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
