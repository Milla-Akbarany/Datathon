# Datathon Deep Learning – Tim T017

Klasifikasi data tabular pada **Dataset A, B, dan C** menggunakan **Multilayer Perceptron (MLP)** dengan TensorFlow/Keras.

- **ID Tim:** T017
- **Anggota:** Milla AKbarany B. P; Indira Zein Rizqin; Selly RIzkiyah
- **NPM:** 23083010021; 23083010015; 23083010010
- **Mata kuliah / dosen:** Deep Learning A / Amri Muhaimin

Seluruh prediksi untuk leaderboard dihasilkan oleh model Neural Network. Random Forest dan XGBoost hanya dipakai sebagai pembanding di laporan.

## Ringkasan Hasil

Metrik dihitung pada data validasi (split stratified 80:20 dari `train.csv`, seed 42). Label data testing tidak diberikan, sehingga tidak ada evaluasi pada data testing.

| Dataset | Tugas | Model terpilih | Val. accuracy | Val. error |
|---|---|---|---|---|
| A | Biner (8 fitur) | MLP 128-128-128, ReLU, dropout 0,0, Adam (lr 1e-3, batch 64) | 0,8503 | 0,1497 |
| B | 6 kelas (33 fitur) | MLP 256-128-64, GELU, dropout 0,2, Adam (lr 1e-3, batch 64) | 0,5975 | 0,4025 |
| C | Biner (3 fitur) | MLP 128-128-128, ReLU, dropout 0,3, Adam (lr 3e-4, batch 64) | 0,8609 | 0,1391 |

Catatan: hasil run ulang dapat bergeser sekitar 0,05–0,3 poin akibat sifat tidak deterministik TensorFlow antar platform.

## Struktur Folder

```
.
├── code/                  # Notebook (.ipynb) yang dipakai untuk seluruh eksperimen
├── models/                # Model terbaik (.keras), preprocessor dan label encoder (.pkl)
├── predictions/           # File submission: T017_<DATASET>_<ATTEMPT>.csv
├── figures/               # Gambar EDA, kurva loss, dan confusion matrix (untuk laporan)
├── requirements.txt       # Versi library yang dipakai
├── experiments.csv        # Rekap semua eksperimen (dataset, config_name, seed, val_accuracy, val_error, training_time, ...)
└── README.md
```

Berkas pendukung lain (jika ada): `tuning_C_log.csv` (log tuning), `baseline_comparison_*.csv` (pembanding Random Forest/XGBoost), `eda_summary*.csv` (ringkasan EDA).

## Lingkungan

- Python 3.13 (Google Colab); notebook juga dijalankan di Kaggle
- TensorFlow 2.21.0, scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.1.3, joblib 1.6.0, matplotlib 3.10.0
- XGBoost 3.4.1 (opsional, hanya untuk model pembanding)
- Seluruh pelatihan berjalan di CPU

Instalasi:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan Ulang

1. Siapkan dataset dengan nama berkas sebagai berikut:
   - Dataset A: `train.csv`, `test.csv`, `sample_submission.csv`
   - Dataset B: `train_B.csv`, `test_B.csv`, `sample_submission_B.csv`
   - Dataset C: `train_C.csv`, `test_C.csv`, `sample_submission_C.csv`
2. Buka notebook di folder `code/` dan sesuaikan lokasi data pada sel **Fase 0** (variabel `DATA_ROOT` / `DATA_DIR`).
3. Jalankan sel secara berurutan. Notebook disusun mengikuti alur tugas:
   - **Fase 1** – Exploratory Data Analysis
   - **Fase 2** – Preprocessing dan pembagian data
   - **Fase 3** – Eksperimen model (3 konfigurasi per dataset), pembanding non-DL, dan tuning
   - **Fase 4** – Pembuatan file submission dan validasi format
   - **Fase 5–6** – Ringkasan, `requirements.txt`, dan ZIP final
4. Sebelum membuat file submission baru, ubah variabel `ATTEMPT` (misalnya `'02'`) agar file lama tidak tertimpa.

## Ringkasan Metode

- **Seed:** 42 (random Python, NumPy, TensorFlow, dan `train_test_split`).
- **Split:** 80:20 stratified; kolom `sample_id` dibuang dan tidak pernah menjadi fitur.
- **Preprocessing:** `StandardScaler` di-fit **hanya** pada data train internal (mencegah data leakage), lalu diterapkan pada data validasi dan test. Label di-encode dengan `LabelEncoder`.
- **Arsitektur:** MLP (Dense → aktivasi → Dropout). Output sigmoid dengan binary cross-entropy (A, C) dan softmax dengan sparse categorical cross-entropy (B).
- **Pelatihan:** optimizer Adam, Early Stopping (monitor `val_loss`, `restore_best_weights=True`), dan ReduceLROnPlateau.
- **Konfigurasi:** Config 1 (Baseline), Config 2 (Deeper), dan Config 3 (hasil hyperparameter tuning dengan random search) untuk tiap dataset.

Rincian lengkap ada di laporan.

## Menggunakan Model yang Tersimpan

Contoh memuat model dan membuat prediksi untuk satu dataset (ubah `a` menjadi `b` atau `c` sesuai dataset):

```python
import joblib, numpy as np, pandas as pd, tensorflow as tf

pre   = joblib.load('models/preprocessor_a.pkl')   # scaler / preprocessor
le    = joblib.load('models/labelencoder_a.pkl')   # label encoder
model = tf.keras.models.load_model('models/model_a_best.keras')

test = pd.read_csv('test.csv')
X = pre.transform(test.drop(columns=['sample_id'])).astype('float32')

prob = model.predict(X, verbose=0)
idx = np.argmax(prob, axis=1) if prob.shape[1] > 1 else (prob.ravel() > 0.5).astype(int)
sub = pd.DataFrame({'sample_id': test['sample_id'], 'target': le.inverse_transform(idx)})
sub.to_csv('T017_A_01.csv', index=False)
```

Jika model akhir sebuah dataset memakai transformasi fitur tambahan (berkas `fe_*.pkl` ada di `models/`), terapkan transformasi itu pada `X` sebelum `model.predict`.

## File Submission

Format: `T<ID_TIM>_<DATASET>_<ATTEMPT>.csv`, dua kolom (`sample_id,target`), berisi label kelas (bukan probabilitas), setiap `sample_id` muncul tepat satu kali, ukuran < 5 MB.

## Catatan

- Penggunaan AI assistant selama pengerjaan dijelaskan pada Bab 7 laporan.
- Eksekusi TensorFlow tidak sepenuhnya deterministik; angka pada laporan berasal dari run yang tercatat di `experiments.csv`.