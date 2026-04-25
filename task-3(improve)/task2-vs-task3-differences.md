# Perbedaan Presentasi Task 2 vs Task 3

Dokumen ini menjelaskan perbedaan antara presentasi Tugas 2 dan draft presentasi Tugas 3. Struktur umumnya sengaja dibuat mirip, tetapi isi dan pesan tiap slide berbeda karena tujuan kedua tugas juga berbeda.

## Perbedaan Umum

**Task 2**
- Fokus utama pada replikasi paper.
- Cerita utama presentasi adalah membandingkan baseline, grid search, dan random search.
- Tekanan terbesar ada pada tuning hyperparameter dan kesesuaian dengan hasil paper.

**Task 3**
- Fokus utama pada perbaikan tahapan pemodelan.
- Cerita utama presentasi adalah memperkuat kualitas evaluasi melalui EDA yang lebih lengkap, preprocessing berbasis temuan data, dan stratified cross-validation.
- Tekanan terbesar ada pada reliabilitas proses, bukan hanya skor akhir.

## Slide-by-Slide

### Slide 1 - Judul

**Task 2**
- Menekankan judul paper asli dan konteks replikasi metodologi.

**Task 3**
- Menekankan bahwa ini adalah perbaikan tahapan pemodelan untuk prediksi penyakit jantung.
- Tetap menyebut paper acuan, tetapi posisinya sebagai referensi, bukan sebagai pusat isi presentasi.

### Slide 2 - Data Collection

**Task 2**
- Fokus pada sumber dataset yang dipakai untuk replikasi.

**Task 3**
- Tetap memakai sumber dataset yang sama, tetapi penekanan ada pada alasan mempertahankan dataset yang sama supaya perbedaan hasil berasal dari perbaikan metodologi, bukan dari perubahan data.

### Slide 3 - Ringkasan Dataset

**Task 2**
- Menjelaskan ukuran dataset, jumlah fitur, dan target secara umum.

**Task 3**
- Menjelaskan hal yang sama, tetapi ditambah penekanan bahwa sebagian fitur yang berupa angka sebenarnya adalah fitur kategorikal sehingga perlu perlakuan khusus saat preprocessing.

### Slide 4 - EDA Awal

**Task 2**
- EDA cenderung lebih umum dan belum terlalu menyorot kualitas data secara mendalam.

**Task 3**
- Menyorot masalah validitas data:
  - tidak ada `NaN` eksplisit,
  - tetapi ada nilai tidak valid seperti `cholesterol = 0`, `resting bp s = 0`, dan `ST slope = 0`.
- Slide ini menjadi dasar keputusan recoding missing pada preprocessing.

### Slide 5 - EDA Lanjutan

**Task 2**
- EDA lebih ringkas atau lebih bersifat deskriptif.

**Task 3**
- Membahas distribusi data, outlier, dan skewness.
- Memberi alasan mengapa outlier tidak langsung dihapus dan mengapa dipilih pendekatan robust.

### Slide 6 - EDA Relasi Fitur

**Task 2**
- Biasanya hanya menampilkan EDA secara umum.

**Task 3**
- Menjelaskan fitur yang paling berkaitan dengan target, baik dari countplot kategorikal maupun korelasi Spearman.
- Slide ini dipakai untuk membenarkan keputusan One-Hot Encoding dan menjaga fitur tetap dipakai.

### Slide 7 - Preprocessing

**Task 2**
- Preprocessing ada, tetapi justifikasinya belum terlalu kuat atau belum secara eksplisit diturunkan dari hasil EDA.

**Task 3**
- Preprocessing menjadi salah satu inti presentasi.
- Ada langkah-langkah yang sebelumnya belum ditekankan:
  - hapus duplikat,
  - recode nilai tidak valid menjadi missing,
  - imputasi berbasis jenis fitur,
  - robust scaling,
  - split stratified.

### Slide 8 - Model Klasifikasi

**Task 2**
- Memperkenalkan model-model yang akan diuji untuk replikasi dan tuning.

**Task 3**
- Daftar model tetap mirip, tetapi pesannya berubah:
  bukan menambah model baru, melainkan mengevaluasi model lama dengan prosedur yang lebih baik.

### Slide 9 - Skenario

**Task 2**
- Skenario terdiri dari baseline, grid search, dan random search.

**Task 3**
- Skenario berubah total.
- Fokus pada:
  - pipeline preprocessing + model,
  - stratified 5-fold cross-validation,
  - evaluasi tambahan pada data test,
  - analisis kestabilan model.

### Slide 10 - Search Space / Konfigurasi

**Task 2**
- Berisi hyperparameter search space untuk grid search dan random search.

**Task 3**
- Tidak lagi menampilkan search space tuning.
- Diganti dengan konfigurasi model yang digunakan dalam eksperimen akhir.
- Ini mencerminkan pergeseran fokus dari tuning ke validasi yang reliabel.

### Slide 11 - Metrics

**Task 2**
- Menampilkan daftar metrik evaluasi.

**Task 3**
- Daftar metrik tetap mirip, tetapi interpretasinya lebih ditekankan, khususnya untuk konteks klasifikasi medis:
  - recall,
  - F1,
  - ROC-AUC,
  - Kappa,
  - MCC.

### Slide 12 - Comparison 1

**Task 2**
- Slide ini berisi perbandingan hasil **baseline**.

**Task 3**
- Slide ini diganti menjadi perbandingan hasil **cross-validation**.
- Fokus pada performa rata-rata di 5 fold, bukan pada baseline model default.

### Slide 13 - Comparison 2

**Task 2**
- Slide ini berisi perbandingan hasil **grid search**.

**Task 3**
- Slide ini diganti menjadi perbandingan hasil **data test**.
- Fokusnya adalah performa akhir pada hold-out set.

### Slide 14 - Comparison 3

**Task 2**
- Slide ini berisi perbandingan hasil **random search**.

**Task 3**
- Slide ini diganti menjadi **perbandingan CV vs test**.
- Ini adalah salah satu inti baru yang tidak ada pada Task 2.

### Slide 15 - Analisis Perbedaan

**Task 2**
- Analisis perbedaan lebih diarahkan ke alasan mengapa hasil replikasi bisa berbeda dari paper asli.

**Task 3**
- Analisis perbedaan diarahkan ke apa yang diperbaiki dari Tugas 2:
  - EDA lebih lengkap,
  - preprocessing lebih justified,
  - evaluasi lebih reliabel,
  - interpretasi model lebih hati-hati.

### Slide 16 - Evaluasi Hasil

**Task 2**
- Menyimpulkan hasil eksperimen tuning dan dominasi model tertentu.

**Task 3**
- Menyimpulkan bahwa proses modeling sekarang lebih kuat secara metodologis.
- Menunjukkan bahwa:
  - Stacking unggul pada CV,
  - KNN unggul pada test,
  - konsistensi model juga menjadi bahan evaluasi.

### Slide 17 - Penutup

**Task 2**
- Slide penutup sederhana.

**Task 3**
- Tetap sederhana, tetapi konteks keseluruhannya sudah berbeda karena presentasi sebelumnya berisi cerita metodologis yang lebih kuat.

## Ringkasan Singkat

Kalau disederhanakan:

- **Task 2** menjawab:
  “Bagaimana hasil replikasi paper dan bagaimana pengaruh baseline, grid search, dan random search?”

- **Task 3** menjawab:
  “Bagaimana memperbaiki alur modeling supaya evaluasi model lebih valid dan lebih reliabel?”

Jadi, **struktur presentasinya memang mirip**, tetapi **isi, fokus, dan pesan analisisnya cukup berbeda**.
