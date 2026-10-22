"""
deepl_milla_v4.py - ensemble BERAGAM (beberapa konfigurasi x beberapa seed) untuk dataset B.

SYARAT: satu folder dengan deepl_milla_v2.py dan deepl_milla_v3.py, dan folder logs_v2/cache_B_03/
(hasil run v3) harus masih ada.

Pemakaian:
    python deepl_milla_v4.py B --team T017 --attempt 04 --no-refit    # cek estimasi val saja (detik)
    python deepl_milla_v4.py B --team T017 --attempt 04               # lengkap: refit + CSV submission

Cara kerja:
  1. Memuat probabilitas validasi yang sudah tersimpan dari run v3 (model dilatih di 80% data, split sama).
     Anggota ensemble ditetapkan di awal (bukan dipilih dari skor val), jadi estimasi val tidak bias oleh seleksi.
  2. Menghitung val accuracy ensemble dan kesalahan standarnya.
  3. Kalau tanpa --no-refit: melatih ulang SEMUA anggota di seluruh train.csv, merata-ratakan probabilitas test,
     lalu menulis outputs_v2/TEAMID_B_04.csv. Hasil refit tiap anggota disimpan (resume otomatis).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from deepl_milla_v2 import BASE_DIR, DATA_ROOT, SEED, make_model, n_params, set_seed
from deepl_milla_v3 import build_prep, fit2, get_probs

REF_ENS = 0.6000   # ensemble 5 seed v2
TARGET = 0.6050    # sasaran val acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="B")
    ap.add_argument("--team", default="T017")
    ap.add_argument("--attempt", default="04")
    ap.add_argument("--src-attempt", default="03", help="attempt v3 yang cache-nya dipakai")
    ap.add_argument("--members", default="base,cos,reg,bigreg,ls", help="nama eksperimen v3, dipisah koma")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-refit", action="store_true", help="hanya hitung estimasi val ensemble")
    args = ap.parse_args()
    D, A = args.dataset.upper(), args.attempt
    members = [m.strip() for m in args.members.split(",") if m.strip()]

    out_dir = os.path.join(BASE_DIR, "outputs_v2")
    log_dir = os.path.join(BASE_DIR, "logs_v2")
    src_dir = os.path.join(log_dir, f"cache_{D}_{args.src_attempt}")
    cache_dir = os.path.join(log_dir, f"cache_{D}_{A}")
    sub_path = os.path.join(out_dir, f"{args.team}_{D}_{A}.csv")
    if not args.no_refit and os.path.exists(sub_path) and not args.force:
        sys.exit(f"{sub_path} sudah ada. Ganti --attempt atau pakai --force.")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # ---------- Data + split (sama persis dengan v2/v3) ----------
    data_dir = os.path.join(DATA_ROOT, "Dataset", f"dataset_{D}")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    feats = [c for c in train.columns if c not in ("sample_id", "target")]
    classes = np.sort(train["target"].unique())
    cmap = {c: i for i, c in enumerate(classes)}
    X_all = train[feats].values.astype(np.float32)
    y_all = train["target"].map(cmap).values
    X_test_raw = test[feats].values.astype(np.float32)
    n_cls = len(classes)
    _, _, _, yva = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)

    # ---------- Muat cache v3: probabilitas validasi + konfigurasi ----------
    hps, eps, val_probs = {}, {}, {}
    for name in members:
        ep_list = []
        for i in range(args.seeds):
            tag = f"{name}_s{SEED + i}"
            jp, npth = os.path.join(src_dir, tag + ".json"), os.path.join(src_dir, tag + "_valprob.npy")
            if not (os.path.exists(jp) and os.path.exists(npth)):
                sys.exit(f"Cache {tag} tidak ditemukan di {src_dir}. Jalankan dulu deepl_milla_v3.py.")
            c = json.load(open(jp))
            hps[name] = c["hp"]
            ep_list.append(c["best_epoch"])
            val_probs[(name, SEED + i)] = np.load(npth)
        eps[name] = float(np.mean(ep_list))

    # ---------- Estimasi val ensemble ----------
    n = len(yva)
    acc = lambda p: float((p.argmax(1) == yva).mean())
    print(f"Dataset {D} | anggota: {members} x {args.seeds} seed = {len(val_probs)} model\n")
    for name in members:
        p = np.mean([val_probs[(name, SEED + i)] for i in range(args.seeds)], axis=0)
        singles = [acc(val_probs[(name, SEED + i)]) for i in range(args.seeds)]
        print(f"  {name:8s} rata-rata seed={np.mean(singles):.4f}  ensemble {args.seeds} seed={acc(p):.4f}")
    all_prob = np.mean(list(val_probs.values()), axis=0)
    ens_acc = acc(all_prob)
    se = float(np.sqrt(ens_acc * (1 - ens_acc) / n))
    print(f"\nEnsemble beragam ({len(val_probs)} model): val_acc={ens_acc:.4f} val_err={1 - ens_acc:.4f} "
          f"(kesalahan standar sekitar +-{se:.4f})")
    print(f"Acuan: ensemble 5 seed v2 = {REF_ENS:.4f} | sasaran = {TARGET:.4f}")
    if ens_acc >= TARGET:
        print("Sasaran tercapai di val. Catatan: selisih kecil seperti ini masih dalam batas noise, jadi belum "
              "tentu berarti skor private lebih baik.")
    else:
        print("Sasaran belum tercapai di val.")

    if args.no_refit:
        print("\n--no-refit: berhenti di sini.")
        return

    # ---------- Refit semua anggota di seluruh train.csv ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    f_full = build_prep(X_all, False)
    Xf, Xt, yf = to_t(f_full(X_all)), to_t(f_full(X_test_raw)), to_t(y_all, torch.long)

    test_probs = []
    t0 = time.time()
    total = len(members) * args.seeds
    done = 0
    for name in members:
        hp = hps[name]
        epochs = hp["epochs"] if hp["sched"] == "cosine" else max(int(round(eps[name])), 1)
        for i in range(args.seeds):
            seed = SEED + i
            done += 1
            tp = os.path.join(cache_dir, f"refit_{name}_s{seed}_test.npy")
            if os.path.exists(tp):
                print(f"[{done}/{total}] {name}_s{seed} SKIP (sudah selesai)")
                test_probs.append(np.load(tp))
                continue
            print(f"[{done}/{total}] refit {name}_s{seed}: {epochs} epoch ...", flush=True)
            set_seed(seed)
            model = make_model(Xf.shape[1], n_cls, hp["hidden"], hp["dropout"]).to(device)
            fit2(model, Xf, yf, None, None, hp, epochs=epochs)
            p = get_probs(model, Xt).cpu().numpy()
            np.save(tp, p)
            test_probs.append(p)
    secs = time.time() - t0

    pred_idx = np.mean(test_probs, axis=0).argmax(1)
    sub = pd.DataFrame({"sample_id": test["sample_id"], "target": classes[pred_idx].astype(int)})
    assert list(sub.columns) == ["sample_id", "target"]
    assert sub["sample_id"].is_unique and len(sub) == len(test)
    assert not sub.isna().any().any() and set(sub["target"]).issubset(set(classes.astype(int)))
    sub.to_csv(sub_path, index=False)

    log_path = os.path.join(log_dir, "experiments_v4.csv")
    row = dict(timestamp=datetime.now().isoformat(timespec="seconds"), dataset=D, attempt=A,
               exp="diverse_ensemble", members="+".join(members), seeds=args.seeds, n_models=len(test_probs),
               val_acc_ensemble=round(ens_acc, 5), val_err_ensemble=round(1 - ens_acc, 5),
               train_time_s_new=round(secs, 1), phase="refit_full_train_ensemble")
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

    print(f"\nSubmission tersimpan & lolos validasi: {sub_path}")
    print(sub["target"].value_counts(normalize=True).sort_index())
    print(f"\nEstimasi val ensemble {ens_acc:.4f} (acuan {REF_ENS:.4f}, sasaran {TARGET:.4f}).")


if __name__ == "__main__":
    main()
