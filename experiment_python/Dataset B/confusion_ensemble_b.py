"""
confusion_ensemble_b.py - confusion matrix + classification report validasi untuk ENSEMBLE 15 model B
(base, cos, reg, bigreg, ls x seed 42-44). Tanpa training: hanya membaca cache hasil v3.

Taruh di folder Milla (sejajar dengan deepl_milla_v3.py), lalu:
    python confusion_ensemble_b.py

Hasil:
    outputs_v2/B_04_confusion_ensemble.png
    logs_v2/B_04_classification_report_ensemble.txt
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
D = "B"
MEMBERS = ["base", "cos", "reg", "bigreg", "ls"]
N_SEEDS = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = BASE_DIR
while not os.path.isdir(os.path.join(DATA_ROOT, "Dataset")) and os.path.dirname(DATA_ROOT) != DATA_ROOT:
    DATA_ROOT = os.path.dirname(DATA_ROOT)

src_dir = os.path.join(BASE_DIR, "logs_v2", f"cache_{D}_03")
out_dir = os.path.join(BASE_DIR, "outputs_v2")
log_dir = os.path.join(BASE_DIR, "logs_v2")
os.makedirs(out_dir, exist_ok=True)

# Split yang sama persis dengan v2/v3/v4
train = pd.read_csv(os.path.join(DATA_ROOT, "Dataset", f"dataset_{D}", "train.csv"))
classes = np.sort(train["target"].unique())
cmap = {c: i for i, c in enumerate(classes)}
feats = [c for c in train.columns if c not in ("sample_id", "target")]
X, y = train[feats].values, train["target"].map(cmap).values
_, _, _, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

probs = []
for name in MEMBERS:
    for i in range(N_SEEDS):
        p = os.path.join(src_dir, f"{name}_s{SEED + i}_valprob.npy")
        if not os.path.exists(p):
            sys.exit(f"Cache tidak ditemukan: {p}")
        probs.append(np.load(p))
pred = np.mean(probs, axis=0).argmax(1)

acc = float((pred == yva).mean())
cm = confusion_matrix(yva, pred)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d", colorbar=False)
plt.title(f"Dataset {D} - confusion matrix validasi (ensemble {len(probs)} MLP), acc={acc:.4f}")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{D}_04_confusion_ensemble.png"), dpi=200)
plt.close()

rep = classification_report(yva, pred, target_names=[str(c) for c in classes], digits=4)
with open(os.path.join(log_dir, f"{D}_04_classification_report_ensemble.txt"), "w") as f:
    f.write(f"Ensemble {len(probs)} model, val_acc={acc:.4f}\n\n{rep}\nConfusion matrix:\n{cm}\n")
print(f"Val acc ensemble: {acc:.4f}  val_err: {1 - acc:.4f}\n")
print(rep)
print("Confusion matrix:\n", cm)
print("\nTertukar paling sering (baris=benar, kolom=prediksi):")
off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j]
for n, a, b in sorted(off, reverse=True)[:5]:
    print(f"  kelas {a} diprediksi {b}: {n} sampel")
