# Draft PPT Tugas 3

Struktur mengikuti alur pada `task-2-ppt.pdf`, tetapi isi disesuaikan dengan notebook `Tugas_3_Data_Mining_Notebook.ipynb`.

## Slide 1 - Judul

**Judul**
Perbaikan Tahapan Pemodelan untuk Prediksi Penyakit Jantung

**Subjudul**
Data Mining - Tugas 3

**Identitas**
- Nama anggota kelompok
- Paper acuan: *A Comparative Study of Machine Learning Models for Heart Disease Prediction Using Grid Search and Random Search for Hyperparameter Tuning*
- Jurnal dan tahun: Journal of Computing and Biomedical Informatics, 2024

**Catatan presenter**
- Tekankan bahwa fokus Tugas 3 bukan replikasi ulang penuh, tetapi perbaikan kualitas metodologi pada tahapan eksplorasi, preprocessing, dan evaluasi model.

## Slide 2 - Data Collection

**Judul**
Data Collection

**Isi utama**
- Sumber dataset: Kaggle `heart-statlog-cleveland-hungary-final`
- Dataset: *Heart Disease Dataset (Comprehensive)*
- Dataset sama dengan Tugas 2 agar hasil tetap dapat dibandingkan

**Visual yang disarankan**
- Screenshot halaman dataset Kaggle atau potongan tabel dataset

## Slide 3 - Ringkasan Dataset

**Judul**
Data Collection

**Isi utama**
- Jumlah data awal: **1190 observasi**
- Jumlah fitur: **11 fitur** dan **1 target**
- Fitur numerik: `age`, `resting bp s`, `cholesterol`, `max heart rate`, `oldpeak`
- Fitur kategorikal/biner: `sex`, `chest pain type`, `fasting blood sugar`, `resting ecg`, `exercise angina`, `ST slope`
- Target:
  - `0` = tidak memiliki penyakit jantung
  - `1` = memiliki penyakit jantung

**Catatan presenter**
- Tambahkan bahwa seluruh kolom sudah numerik secara format, tetapi sebagian merepresentasikan kategori sehingga perlu perlakuan berbeda saat preprocessing.

## Slide 4 - Temuan EDA Awal

**Judul**
EDA

**Isi utama**
- Secara eksplisit dataset tidak memiliki `NaN`
- Ditemukan nilai tidak valid yang secara domain medis tidak realistis:
  - `cholesterol = 0` sebanyak **172 baris** (**14.45%**)
  - `resting bp s = 0` sebanyak **1 baris**
  - `ST slope = 0` sebanyak **1 baris**
- Nilai tersebut diperlakukan sebagai missing pada tahap preprocessing

**Pesan utama**
- Walaupun data tampak bersih secara teknis, secara substantif masih ada kualitas data yang harus diperbaiki.

## Slide 5 - EDA: Distribusi dan Outlier

**Judul**
EDA

**Isi utama**
- Outlier paling menonjol terdapat pada:
  - `cholesterol` sekitar **3.08%**
  - `resting bp s` sekitar **2.94%**
  - `oldpeak` sekitar **1.74%**
- `cholesterol` dan `oldpeak` cenderung **right-skewed**
- Nilai ekstrem tidak langsung dihapus karena pada data medis bisa merepresentasikan kondisi klinis nyata

**Visual yang disarankan**
- Boxplot fitur numerik
- Histogram distribusi fitur numerik

**Pesan utama**
- Penanganan outlier dipilih secara robust, bukan dengan membuang data secara agresif.

## Slide 6 - EDA: Relasi Fitur dengan Target

**Judul**
EDA

**Isi utama**
- Fitur kategorikal yang paling membedakan target:
  - `exercise angina`
  - `ST slope`
  - `chest pain type`
- Korelasi Spearman terhadap target paling kuat:
  - `ST slope` = **0.5917**
  - `chest pain type` = **0.5116**
  - `exercise angina` = **0.4943**
  - `oldpeak` = **0.4190**
  - `max heart rate` = **-0.4048**
- Tidak ada multikolinearitas ekstrem antarf fitur

**Visual yang disarankan**
- Countplot kategorikal vs target
- Heatmap korelasi Spearman

## Slide 7 - Preprocessing

**Judul**
Preprocessing

**Isi utama**
- Ditemukan **272 baris duplikat** (**22.86%**) lalu dihapus
- Nilai tidak valid direcode menjadi missing:
  - `cholesterol=0`
  - `resting bp s=0`
  - `ST slope=0`
- Split data menggunakan **train-test stratify**
- Imputasi missing:
  - numerik -> **median**
  - kategorikal -> **most frequent**
- Encoding kategorikal -> **One-Hot Encoding**
- Scaling numerik -> **RobustScaler**

**Ringkasan hasil**
- Data sebelum hapus duplikat: **1190**
- Data sesudah hapus duplikat: **918**
- Data train: **734**
- Data test: **184**

## Slide 8 - Model Klasifikasi

**Judul**
Model Klasifikasi

**Isi utama**
- Logistic Regression
- KNN
- SVM (RBF)
- Random Forest
- Gradient Boosting
- Stacking Classifier

**Catatan presenter**
- Jelaskan bahwa daftar model tetap dibuat konsisten dengan studi sebelumnya, tetapi evaluasinya diperkuat.

## Slide 9 - Skenario Eksperimen

**Judul**
Model Klasifikasi

**Subjudul**
Skenario

**Isi utama**
- Seluruh model dibangun dalam **pipeline** yang menggabungkan preprocessing dan model
- Validasi utama menggunakan **StratifiedKFold 5-fold**
- Metrik dievaluasi pada dua level:
  1. rata-rata performa cross-validation
  2. performa final pada data test
- Tujuan utama:
  - mengukur kestabilan model
  - membandingkan hasil CV dengan hasil test
  - mengurangi risiko evaluasi yang terlalu optimis

## Slide 10 - Konfigurasi Pemodelan

**Judul**
Model Klasifikasi

**Subjudul**
Konfigurasi Model

**Isi utama**
- Logistic Regression: `max_iter=3000`, `class_weight=balanced`
- KNN: `n_neighbors=11`, `weights=distance`
- SVM (RBF): `C=10`, `class_weight=balanced`, `probability=True`
- Random Forest: `n_estimators=300`, `class_weight=balanced`
- Gradient Boosting: parameter default dengan `random_state=42`
- Stacking:
  - base learner: Logistic Regression, Random Forest, Gradient Boosting
  - final estimator: Logistic Regression
  - internal CV = 5

**Catatan presenter**
- Slide ini menggantikan slide search space pada Tugas 2, karena fokus Tugas 3 adalah perbaikan validasi dan bukan tuning hyperparameter.

## Slide 11 - Metrics

**Judul**
Metrics

**Isi utama**
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cohen's Kappa
- Matthews Correlation Coefficient (MCC)

**Pesan utama**
- ROC-AUC dan F1 diberi perhatian khusus karena lebih informatif untuk kasus klasifikasi medis.

## Slide 12 - Hasil Cross-Validation

**Judul**
Comparison Cross-Validation

**Isi utama**
- Peringkat berdasarkan **CV ROC-AUC**:
  - Stacking: **0.9296**
  - Random Forest: **0.9275**
  - Logistic Regression: **0.9221**
  - Gradient Boosting: **0.9186**
  - KNN: **0.9141**
  - SVM: **0.9003**
- CV F1 terbaik: **KNN = 0.8775**
- CV ROC-AUC terbaik: **Stacking = 0.9296**

**Visual yang disarankan**
- Tabel ringkasan hasil CV

## Slide 13 - Hasil Test Set

**Judul**
Comparison Test Set

**Isi utama**
- Peringkat berdasarkan **Test ROC-AUC**:
  - KNN: **0.9448**
  - Stacking: **0.9341**
  - Logistic Regression: **0.9321**
  - Random Forest: **0.9306**
  - Gradient Boosting: **0.9142**
  - SVM: **0.9137**
- Test F1 terbaik: **KNN = 0.9216**
- Model terbaik pada data test: **KNN**

**Visual yang disarankan**
- Tabel hasil test
- ROC curve atau confusion matrix model terbaik

## Slide 14 - Perbandingan CV vs Test

**Judul**
Comparison CV vs Test

**Isi utama**
- Gap kecil menunjukkan model lebih konsisten
- Beberapa contoh gap:
  - Stacking: gap ROC-AUC **+0.0045**
  - Random Forest: gap ROC-AUC **+0.0031**
  - Gradient Boosting: gap ROC-AUC **-0.0045**
  - KNN: gap ROC-AUC **+0.0307**
- Interpretasi:
  - Stacking dan Random Forest relatif paling stabil
  - KNN menghasilkan skor test tertinggi, tetapi gap terhadap CV lebih besar

**Visual yang disarankan**
- Bar chart CV F1 vs Test F1
- Bar chart CV ROC-AUC vs Test ROC-AUC

## Slide 15 - Analisis Perbedaan dari Tugas 2

**Judul**
Analisis Perbedaan

**Isi utama**
- Tugas 2 belum menyajikan EDA poin 3-10 secara lengkap
- Tugas 2 belum menangani:
  - duplikat
  - nilai tidak valid sebagai missing
  - validasi stratified cross-validation secara eksplisit
- Tugas 3 memperbaiki evaluasi agar:
  - lebih adil
  - lebih tahan terhadap bias split data
  - lebih mudah menjelaskan alasan preprocessing

**Pesan utama**
- Perbedaan utama Tugas 3 bukan pada banyaknya model, tetapi pada kekuatan metodologinya.

## Slide 16 - Evaluasi Hasil

**Judul**
Evaluasi Hasil

**Isi utama**
1. Seluruh tahapan eksplorasi, preprocessing, dan pemodelan berhasil dijalankan dengan alur yang lebih kuat secara metodologis.
2. Penghapusan duplikat dan recoding nilai tidak valid meningkatkan kualitas data sebelum model dilatih.
3. Stratified cross-validation memberi gambaran performa yang lebih reliabel dibanding hanya melihat skor test.
4. Model terbaik pada CV adalah **Stacking**, sedangkan model terbaik pada test adalah **KNN**.
5. Secara konsistensi, **Stacking** dan **Random Forest** tampak lebih stabil karena gap CV-test relatif kecil.

## Slide 17 - Penutup

**Judul**
Thank You

**Isi utama**
- Thank you for your attention
- Q&A

## Saran Penyusunan Visual

- Slide 4: tabel invalid values
- Slide 5: boxplot + histogram
- Slide 6: countplot kategorikal + heatmap korelasi
- Slide 12: tabel hasil cross-validation
- Slide 13: tabel hasil test + confusion matrix atau ROC curve
- Slide 14: grafik perbandingan CV vs test
