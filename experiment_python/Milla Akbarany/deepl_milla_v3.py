"""
deepl_milla_v3.py - upaya menaikkan akurasi dataset B (bisa dipakai A/C, tapi eksperimennya disetel untuk B).

SYARAT: file ini harus satu folder dengan deepl_milla_v2.py (fungsi dasarnya di-import dari sana).

Pemakaian (dari folder Milla):
    python deepl_milla_v3.py B --team T017 --attempt 03

Beda dari v2:
  1. Tiap eksperimen dilatih dengan beberapa seed (default 3) di split validasi YANG SAMA dengan v2,
     lalu dibandingkan dari RATA-RATA seed. Di v2 seed 42 untung (60,03%), padahal 4 seed lain 59,35-59,58%.
  2. Eksperimen baru: regularisasi lebih kuat, label smoothing, jadwal LR cosine, fitur polinomial derajat 2
     (interaksi antar fitur), model besar + regularisasi kuat, dan kombinasinya.
  3. Submission akhir = ensemble beberapa seed dari konfigurasi terbaik (lebih stabil daripada satu seed).
  4. Diagnosis cepat (LDA/QDA/LogReg-poly2) untuk melihat struktur data; hanya untuk laporan.
Semua hasil: outputs_v2/ dan logs_v2/ (nama berawalan attempt, mis. B_03_...). Resume otomatis per (eksperimen, seed).
"""
import argparse
import json
import os
import platform
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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from deepl_milla_v2 import BASE_DIR, DATA_ROOT, SEED, evaluate, make_model, n_params, save_loss_curve, set_seed

# Acuan dari v2 (dataset B): seed 42 = 0.6003; rata-rata 5 seed = 0.5959; ensemble 5 seed = 0.6000
REF_MEAN, REF_ENS = 0.5959, 0.6000

DEFAULT = dict(hidden=[256, 128, 64], dropout=0.2, lr=1e-3, wd=1e-3, bs=256,
               ls=0.0, sched="es", epochs=200, patience=15, poly=False)
# sched="es": early stopping pada val loss (epoch maksimal = epochs)
# sched="cosine": LR cosine, latih tepat `epochs` epoch, bobot epoch terakhir dipakai
EXPERIMENTS = [
    dict(name="base"),                                                   # = konfigurasi terbaik v2
    dict(name="reg", dropout=0.3, wd=1e-2),                              # regularisasi lebih kuat
    dict(name="ls", ls=0.1),                                             # label smoothing
    dict(name="cos", sched="cosine", epochs=100),                        # jadwal LR cosine
    dict(name="poly2", poly=True),                                       # fitur kuadrat + interaksi
    dict(name="bigreg", hidden=[512, 256, 128], dropout=0.4, wd=1e-2),   # model besar + regularisasi kuat
    dict(name="combo", dropout=0.3, wd=1e-2, ls=0.1, sched="cosine", epochs=100),
]


def build_prep(Xfit, poly):
    """Imputer (+ fitur polinomial) + scaler, semuanya di-fit HANYA pada Xfit."""
    imp = SimpleImputer(strategy="median").fit(Xfit)
    pf = PolynomialFeatures(2, include_bias=False) if poly else None
    Z = imp.transform(Xfit)
    if pf is not None:
        Z = pf.fit_transform(Z)
    sc = StandardScaler().fit(Z)

    def f(X):
        Z = imp.transform(X)
        if pf is not None:
            Z = pf.transform(Z)
        return sc.transform(Z).astype(np.float32)

    return f


def fit2(model, Xtr, ytr, Xva, yva, hp, epochs=None):
    """Xva ada: mode validasi. Xva None: refit tepat `epochs` epoch (cosine: hp['epochs'])."""
    ce_val = nn.CrossEntropyLoss()
    ce_tr = nn.CrossEntropyLoss(label_smoothing=hp["ls"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    cosine = hp["sched"] == "cosine"
    total = epochs if Xva is None else hp["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total) if cosine else None
    n, bs = Xtr.shape[0], hp["bs"]
    hist = {"train_loss": [], "val_loss": []}
    best_val, best_epoch, best_state, bad = float("inf"), 0, None, 0

    for ep in range(1, total + 1):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        run, seen = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:  # BatchNorm butuh minimal 2 sampel
                continue
            opt.zero_grad()
            loss = ce_tr(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            run += loss.item() * len(idx)
            seen += len(idx)
        if sched is not None:
            sched.step()
        hist["train_loss"].append(run / seen)

        if Xva is not None:
            vloss, _ = evaluate(model, Xva, yva, ce_val)
            hist["val_loss"].append(vloss)
            if not cosine:
                if vloss < best_val - 1e-5:
                    best_val, best_epoch, bad = vloss, ep, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= hp["patience"]:
                        break
        if ep % 25 == 0:
            msg = f"    epoch {ep}/{total} train_loss={hist['train_loss'][-1]:.4f}"
            if Xva is not None:
                msg += f" val_loss={hist['val_loss'][-1]:.4f}"
            print(msg, flush=True)

    if Xva is not None and not cosine and best_state is not None:
        model.load_state_dict(best_state)
    if cosine:
        best_epoch = total
    return hist, best_epoch


def get_probs(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), 1)


def run_diag(Xtr_raw, ytr, Xva_raw, yva, D, log_dir):
    """Diagnosis struktur data (hanya untuk laporan, bukan sumber prediksi)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
        "LogReg+poly2": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                      StandardScaler(), LogisticRegression(max_iter=300, C=0.1)),
    }
    rows = []
    print("\nDiagnosis struktur data (hanya untuk laporan):")
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr_raw, ytr)
        acc = float((m.predict(Xva_raw) == yva).mean())
        print(f"  {name:14s} val_acc={acc:.4f} time={time.time() - t0:.1f}s", flush=True)
        rows.append(dict(dataset=D, model=name, val_acc=round(acc, 5)))
    pd.DataFrame(rows).to_csv(os.path.join(log_dir, f"diagnosis_{D}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--team", default="T017")
    ap.add_argument("--attempt", default="03")
    ap.add_argument("--seeds", type=int, default=3, help="jumlah seed per eksperimen (validasi)")
    ap.add_argument("--final-seeds", type=int, default=5, help="jumlah model di ensemble submission akhir")
    ap.add_argument("--no-diag", action="store_true", help="lewati diagnosis LDA/QDA/LogReg")
    ap.add_argument("--force", action="store_true", help="boleh menimpa CSV submission yang sudah ada")
    ap.add_argument("--restart", action="store_true", help="abaikan cache")
    ap.add_argument("--no-refit", action="store_true", help="hanya validasi, tanpa CSV submission")
    args = ap.parse_args()
    D = args.dataset.upper()
    A = args.attempt

    out_dir = os.path.join(BASE_DIR, "outputs_v2")
    log_dir = os.path.join(BASE_DIR, "logs_v2")
    cache_dir = os.path.join(log_dir, f"cache_{D}_{A}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    sub_path = os.path.join(out_dir, f"{args.team}_{D}_{A}.csv")
    if not args.no_refit and os.path.exists(sub_path) and not args.force:
        sys.exit(f"{sub_path} sudah ada. Ganti --attempt atau pakai --force.")
    if args.restart and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU {platform.processor()}"
    print(f"Dataset {D} | attempt {A} | device: {hw} | seed/eksperimen: {args.seeds}")

    # ---------- Load + split (sama persis dengan v2) ----------
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
    Xtr_raw, Xva_raw, ytr, yva = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)

    to_t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    ytr_t, yva_t = to_t(ytr, torch.long), to_t(yva, torch.long)
    tensors = {}

    def get_tensors(poly):
        if poly not in tensors:
            f = build_prep(Xtr_raw, poly)  # fit hanya di training internal
            tensors[poly] = (to_t(f(Xtr_raw)), to_t(f(Xva_raw)))
        return tensors[poly]

    if not args.no_diag:
        run_diag(Xtr_raw, ytr, Xva_raw, yva, D, log_dir)

    # ---------- Eksperimen (multi-seed) ----------
    log_path = os.path.join(log_dir, "experiments_v3.csv")
    summary = []
    for exp in EXPERIMENTS:
        hp = {**DEFAULT, **{k: v for k, v in exp.items() if k != "name"}}
        Xtr, Xva = get_tensors(hp["poly"])
        runs = []
        for i in range(args.seeds):
            seed = SEED + i
            tag = f"{exp['name']}_s{seed}"
            jp = os.path.join(cache_dir, tag + ".json")
            npth = os.path.join(cache_dir, tag + "_valprob.npy")
            if os.path.exists(jp) and os.path.exists(npth):
                c = json.load(open(jp))
                if c["hp"] == hp:
                    print(f"[{D}/{tag:14s}] SKIP val_acc={c['val_acc']:.4f} best_ep={c['best_epoch']}")
                    runs.append(dict(acc=c["val_acc"], loss=c["val_loss"], ep=c["best_epoch"], prob=np.load(npth)))
                    continue
            print(f"[{D}/{tag:14s}] mulai: {hp}", flush=True)
            set_seed(seed)
            model = make_model(Xtr.shape[1], n_cls, hp["hidden"], hp["dropout"]).to(device)
            t0 = time.time()
            hist, best_ep = fit2(model, Xtr, ytr_t, Xva, yva_t, hp)
            secs = time.time() - t0
            vloss, _ = evaluate(model, Xva, yva_t, nn.CrossEntropyLoss())
            prob = get_probs(model, Xva)
            vacc = (prob.argmax(1) == yva_t).float().mean().item()
            print(f"[{D}/{tag:14s}] val_acc={vacc:.4f} val_err={1 - vacc:.4f} val_loss={vloss:.4f} "
                  f"best_ep={best_ep} params={n_params(model)} time={secs:.1f}s", flush=True)
            if i == 0:
                save_loss_curve(hist, best_ep, f"Dataset {D} - {exp['name']} (v3, seed {seed})",
                                os.path.join(out_dir, f"{D}_{A}_{exp['name']}_loss.png"))
            row = dict(timestamp=datetime.now().isoformat(timespec="seconds"), dataset=D, attempt=A,
                       exp=exp["name"], seed=seed, hidden=str(hp["hidden"]), dropout=hp["dropout"],
                       optimizer="AdamW", lr=hp["lr"], weight_decay=hp["wd"], batch_size=hp["bs"],
                       label_smoothing=hp["ls"], schedule=hp["sched"], poly2=hp["poly"],
                       max_epochs=hp["epochs"], patience=hp["patience"], best_epoch=best_ep,
                       params=n_params(model), val_loss=round(vloss, 5), val_acc=round(vacc, 5),
                       val_err=round(1 - vacc, 5), train_time_s=round(secs, 1), hardware=hw, phase="validation")
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
            prob_np = prob.cpu().numpy()
            np.save(npth, prob_np)
            json.dump(dict(hp=hp, val_acc=vacc, val_loss=vloss, best_epoch=best_ep), open(jp, "w"))
            runs.append(dict(acc=vacc, loss=vloss, ep=best_ep, prob=prob_np))

        accs = np.array([r["acc"] for r in runs])
        ens_prob = np.mean([r["prob"] for r in runs], axis=0)
        ens_acc = float((ens_prob.argmax(1) == yva).mean())
        summary.append(dict(exp=exp, hp=hp, mean=accs.mean(), std=accs.std(ddof=1) if len(accs) > 1 else 0.0,
                            ens=ens_acc, loss=float(np.mean([r["loss"] for r in runs])),
                            ep=float(np.mean([r["ep"] for r in runs])), ens_prob=ens_prob))

    # ---------- Ringkasan ----------
    summary.sort(key=lambda r: (-r["mean"], r["loss"]))
    print(f"\nRingkasan (urut rata-rata val acc {args.seeds} seed; acuan v2: rata-rata {REF_MEAN:.4f}, ensemble {REF_ENS:.4f}):")
    for r in summary:
        print(f"  {r['exp']['name']:8s} mean={r['mean']:.4f} +-{r['std']:.4f}  ens_seed={r['ens']:.4f}  "
              f"val_loss={r['loss']:.4f}  best_ep~{r['ep']:.0f}")
    best = summary[0]
    hp = best["hp"]
    print(f"\n>>> Terbaik {D}: {best['exp']['name']} {hp['hidden']} dropout={hp['dropout']} wd={hp['wd']} "
          f"ls={hp['ls']} sched={hp['sched']} poly2={hp['poly']}")
    print(f"    rata-rata seed {best['mean']:.4f} (v2: {REF_MEAN:.4f}) | ensemble seed {best['ens']:.4f} (v2: {REF_ENS:.4f})")

    pred_va = best["ens_prob"].argmax(1)
    cm = confusion_matrix(yva, pred_va)
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d")
    plt.title(f"Dataset {D} - confusion matrix validasi ({best['exp']['name']}, ensemble seed)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{D}_{A}_confusion.png"), dpi=150)
    plt.close()
    rep = classification_report(yva, pred_va, target_names=[str(c) for c in classes], digits=4)
    with open(os.path.join(log_dir, f"{D}_{A}_classification_report.txt"), "w") as f:
        f.write(f"Konfigurasi: {best['exp']['name']} {hp}\n\n{rep}\nConfusion matrix:\n{cm}\n")
    print(rep)

    if args.no_refit:
        print("Refit dilewati (--no-refit).")
        return

    # ---------- Refit di seluruh train.csv: ensemble beberapa seed ----------
    epochs_refit = hp["epochs"] if hp["sched"] == "cosine" else max(int(round(best["ep"])), 1)
    print(f"\nRefit di seluruh train.csv: {args.final_seeds} model x {epochs_refit} epoch ...", flush=True)
    f_full = build_prep(X_all, hp["poly"])
    Xf, Xt = to_t(f_full(X_all)), to_t(f_full(X_test_raw))
    yf = to_t(y_all, torch.long)
    t0 = time.time()
    test_probs = []
    for k in range(args.final_seeds):
        set_seed(SEED + k)
        model = make_model(Xf.shape[1], n_cls, hp["hidden"], hp["dropout"]).to(device)
        fit2(model, Xf, yf, None, None, hp, epochs=epochs_refit)
        test_probs.append(get_probs(model, Xt))
        suffix = "" if k == 0 else f"_s{k}"
        torch.save(model.state_dict(), os.path.join(out_dir, f"{D}_{A}_final{suffix}.pt"))
        print(f"  model {k + 1}/{args.final_seeds} selesai", flush=True)
    secs = time.time() - t0
    pred_idx = torch.stack(test_probs).mean(0).argmax(1).cpu().numpy()

    sub = pd.DataFrame({"sample_id": test["sample_id"], "target": classes[pred_idx].astype(int)})
    assert list(sub.columns) == ["sample_id", "target"]
    assert sub["sample_id"].is_unique and len(sub) == len(test)
    assert not sub.isna().any().any() and set(sub["target"]).issubset(set(classes.astype(int)))
    sub.to_csv(sub_path, index=False)

    row = dict(timestamp=datetime.now().isoformat(timespec="seconds"), dataset=D, attempt=A,
               exp=best["exp"]["name"] + f"_refit_ens{args.final_seeds}", seed=f"{SEED}..{SEED + args.final_seeds - 1}",
               hidden=str(hp["hidden"]), dropout=hp["dropout"], optimizer="AdamW", lr=hp["lr"],
               weight_decay=hp["wd"], batch_size=hp["bs"], label_smoothing=hp["ls"], schedule=hp["sched"],
               poly2=hp["poly"], max_epochs=epochs_refit, patience="-", best_epoch=epochs_refit,
               params=n_params(model), val_loss="-", val_acc="-", val_err="-", train_time_s=round(secs, 1),
               hardware=hw, phase="refit_full_train_ensemble")
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

    print(f"\nSubmission tersimpan & lolos validasi: {sub_path}")
    print(sub["target"].value_counts(normalize=True).sort_index())
    gain = best["ens"] - REF_ENS
    print(f"\nEnsemble val {best['ens']:.4f} vs acuan v2 {REF_ENS:.4f} (selisih {gain * 100:+.2f} poin). "
          f"Noise val B sekitar +-0.4 poin: unggah CSV ini hanya kalau selisihnya jelas positif (>= +0.4).")


if __name__ == "__main__":
    main()