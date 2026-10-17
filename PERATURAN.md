# Deep Learning Mystery Data Challenge

## Tujuan dan pelaksanaan
Bangun model deep learning untuk memprediksi kelas pada tiga dataset tabular anonim A, B, dan C. Setiap tim wajib mengerjakan ketiganya. Rekomendasi: tim 2–3 mahasiswa, waktu 7 hari, perangkat komputasi bebas dengan sumber daya dilaporkan. Dosen menetapkan tanggal mulai, tenggat (termasuk zona waktu), kanal pengumpulan, dan daftar tim sebelum peluncuran.

Setiap folder memuat train.csv (fitur + target), test.csv (tanpa target), dan sample_submission.csv. sample_id hanya untuk pencocokan prediksi dan dilarang menjadi fitur model. Nama fitur disamarkan; jangan menafsirkan nama tersebut sebagai domain nyata. Template submission berisi nilai nol sebagai placeholder, bukan contoh prediksi yang benar.

## Model dan validasi
1. Gunakan minimal satu model neural network yang dilatih untuk setiap dataset; MLP dianjurkan sebagai baseline. Ensemble diperbolehkan apabila seluruh anggotanya neural network. Model klasik hanya boleh menjadi pembanding di laporan dan bukan sumber prediksi final.
2. Pisahkan validasi stratified dari training, misalnya 20% training. Fit scaler, imputasi, seleksi fitur dan resampling hanya pada bagian training internal. Testing tidak boleh digunakan untuk fitting, pemilihan hiperparameter atau pelatihan semi-supervised.
3. Laporkan arsitektur, parameter, optimizer, learning rate, batch size, epoch, seed, early stopping, perangkat keras dan waktu pelatihan. Berikan kurva loss, confusion matrix validasi serta analisis kesalahan.
4. Lakukan minimal dua eksperimen terkontrol per dataset, misalnya jumlah hidden layer atau dropout. Setelah pemilihan konfigurasi melalui validasi internal, boleh melatih ulang pada seluruh train.csv dengan konfigurasi yang telah dipilih.
5. AI assistant boleh membantu kode/debugging dan harus diungkapkan; mahasiswa tetap bertanggung jawab menjelaskan kode dan keputusan eksperimen. Dilarang bertukar prediksi/kode antartim, mencari label data sumber, mengambil jawaban dari dataset eksternal, atau menggunakan akun/tim tambahan untuk menambah kuota.

## Submission dan kuota
Maksimal 3 submission valid PER DATASET PER TIM, total maksimum 9. Kuota berlaku sepanjang kompetisi, bukan per hari. Berkas format salah ditolak tanpa mengurangi kuota. Berkas valid yang skor prediksinya buruk tetap mengurangi kuota. ID tim ditetapkan dosen dan tidak boleh berubah.

Satu submission adalah satu CSV untuk satu dataset, dua kolom tepat: sample_id,target. Semua ID testing harus muncul tepat satu kali, tanpa ID tambahan atau nilai kosong. target berupa label kelas integer yang tersedia di training, bukan probabilitas. Urutan baris boleh berbeda. Gunakan nama TEAMID_A_01.csv, dan seterusnya. Batas ukuran 5 MB per CSV. Unggah melalui kanal resmi yang diumumkan dosen; waktu diterima kanal tersebut menjadi patokan tenggat.

## Leaderboard
Masing-masing dataset memiliki leaderboard sendiri. Metrik utama accuracy = jumlah prediksi benar / jumlah sampel; classification error = 1 − accuracy, bukan MAE/RMSE. Selama lomba hanya sebagian testing tersembunyi yang menghasilkan skor public. Sisanya digunakan untuk skor private setelah penutupan. Peserta tidak diberi informasi keanggotaan kedua bagian ini.

Sistem memilih submission dengan public accuracy tertinggi untuk setiap tim/dataset; jika sama, dipilih yang diterima lebih awal. Setelah penutupan, submission terpilih yang SAMA dinilai pada private testing untuk menentukan peringkat akhir. Skor seri ditentukan waktu penerimaan submission terpilih lebih awal. Jangan menyesuaikan model berulang kali hanya berdasarkan skor leaderboard; gunakan validasi internal.

Ada tiga peringkat akhir, masing-masing untuk A, B, C. Jika diperlukan juara umum, gunakan rata-rata private accuracy ketiga dataset dengan bobot sama. Tim tanpa submission pada salah satu dataset tidak memenuhi syarat juara umum.

## Pengumpulan akhir dan penilaian akademik
Serahkan satu ZIP maksimal 50 MB berisi notebook/script runnable, requirements.txt, README cara menjalankan, laporan PDF maksimal 8 halaman, log eksperimen, dan tiga CSV prediksi terpilih. Model weights boleh melalui tautan terpisah jika melampaui batas. Jangan sertakan ulang dataset. Semua eksperimen harus dapat direproduksi dengan seed yang dilaporkan.

Rekomendasi rubrik: performa private tiga dataset 30%; kebenaran pipeline dan pencegahan leakage 25%; eksperimen dan analisis 25%; reproduktibilitas 15%; kejelasan laporan dan pengungkapan penggunaan AI 5%. Komponen performa = 30 × rata-rata private accuracy (skala 0–1). Peringkat kompetisi tidak otomatis sama dengan nilai akademik. Pelanggaran integritas ditinjau dosen berdasarkan bukti dan aturan akademik institusi.
