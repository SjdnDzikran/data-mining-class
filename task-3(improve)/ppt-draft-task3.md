# Draft PPT Tugas 3

Struktur mengikuti alur presentasi pada `task-2-ppt.pdf`, tetapi isi diperluas sesuai fokus Tugas 3, yaitu perbaikan tahapan eksplorasi data, preprocessing, dan evaluasi model.

Setiap bagian di bawah ini ditulis agar lebih mudah langsung dipindahkan ke slide. Jika nanti ingin, isi ini masih bisa dipadatkan lagi menjadi versi yang lebih presentable.

## Slide 1 - Judul

**Judul utama**
Perbaikan Tahapan Pemodelan untuk Prediksi Penyakit Jantung

**Subjudul**
Tugas 3 Data Mining

**Isi slide**
- Tugas ini merupakan lanjutan dari Tugas 2 yang sebelumnya berfokus pada replikasi metodologi dari paper.
- Pada Tugas 3, fokus utama bukan lagi membandingkan baseline, grid search, dan random search, tetapi memperbaiki kualitas alur analisis agar lebih kuat secara metodologis.
- Perbaikan dilakukan pada tiga bagian utama:
  - data exploration yang lebih lengkap,
  - preprocessing yang berbasis hasil exploration,
  - evaluasi model menggunakan stratified cross-validation dan perbandingan terhadap data test.

**Identitas**
- Nama anggota kelompok
- Paper acuan: *A Comparative Study of Machine Learning Models for Heart Disease Prediction Using Grid Search and Random Search for Hyperparameter Tuning*
- Jurnal: *Journal of Computing and Biomedical Informatics*, 2024

**Catatan presenter**
- Saat membuka presentasi, langsung jelaskan bahwa Tugas 3 adalah “versi metodologis yang diperbaiki” dari Tugas 2.

## Slide 2 - Data Collection

**Judul**
Data Collection

**Isi slide**
- Dataset yang digunakan tetap sama dengan Tugas 2 agar hasil tetap dapat dibandingkan secara adil.
- Sumber dataset berasal dari Kaggle:
  `https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final`
- Nama dataset: **Heart Disease Dataset (Comprehensive)**.
- Dataset ini dipakai karena sesuai dengan paper acuan dan memuat fitur-fitur klinis yang relevan untuk prediksi penyakit jantung.
- Dengan memakai dataset yang sama, perbedaan hasil pada Tugas 3 lebih mencerminkan perbaikan proses analisis, bukan karena perubahan data.

**Visual yang disarankan**
- Screenshot halaman dataset Kaggle
- Atau tabel ringkas sumber dataset

## Slide 3 - Ringkasan Dataset

**Judul**
Data Collection

**Isi slide**
- Dataset awal terdiri dari **1190 observasi** dan **12 kolom**.
- Komposisi kolom:
  - **11 fitur**
  - **1 variabel target**
- Target bersifat biner:
  - `0` = tidak memiliki penyakit jantung
  - `1` = memiliki penyakit jantung
- Secara umum fitur dapat dibagi menjadi dua kelompok:
  - **fitur numerik**: `age`, `resting bp s`, `cholesterol`, `max heart rate`, `oldpeak`
  - **fitur kategorikal/biner**: `sex`, `chest pain type`, `fasting blood sugar`, `resting ecg`, `exercise angina`, `ST slope`
- Seluruh kolom memang tersimpan dalam format numerik, tetapi tidak semuanya boleh diperlakukan sebagai numerik saat modeling.

**Pesan utama**
- Karakteristik data ini membuat preprocessing menjadi penting, karena ada campuran fitur numerik dan fitur encoded categorical.

## Slide 4 - EDA Awal dan Validitas Data

**Judul**
EDA

**Isi slide**
- Hasil pemeriksaan awal menunjukkan bahwa dataset tidak memiliki missing value eksplisit (`NaN`).
- Namun setelah diperiksa lebih dalam, ditemukan beberapa nilai yang tidak realistis secara domain medis:
  - `cholesterol = 0` sebanyak **172 baris** atau sekitar **14.45%**
  - `resting bp s = 0` sebanyak **1 baris**
  - `ST slope = 0` sebanyak **1 baris**
- Secara teknis nilai tersebut bukan missing, tetapi secara substantif nilainya sangat mungkin merepresentasikan data yang tidak valid.
- Temuan ini penting karena jika dibiarkan, model akan menganggap nilai tersebut valid dan bisa menghasilkan pola yang menyesatkan.
- Oleh sebab itu, pada Tugas 3 nilai-nilai tersebut direcode menjadi missing terlebih dahulu sebelum proses imputasi.

**Pesan utama**
- Data “tanpa NaN” belum tentu benar-benar bersih. Pemeriksaan validitas domain tetap diperlukan.

## Slide 5 - EDA: Distribusi Data dan Outlier

**Judul**
EDA

**Isi slide**
- Ringkasan statistik menunjukkan beberapa fitur memiliki sebaran yang cukup lebar, terutama `cholesterol` dan `oldpeak`.
- Hasil analisis outlier berbasis IQR menunjukkan:
  - `cholesterol` memiliki outlier sekitar **3.08%**
  - `resting bp s` sekitar **2.94%**
  - `oldpeak` sekitar **1.74%**
- Selain itu, analisis skewness menunjukkan bahwa:
  - `cholesterol` bersifat **right-skewed**
  - `oldpeak` juga **right-skewed**
  - `age` dan `max heart rate` relatif lebih stabil
- Pada data medis, nilai ekstrem tidak selalu salah karena bisa merepresentasikan pasien dengan kondisi klinis yang berat.
- Karena itu, outlier **tidak dihapus langsung**, melainkan ditangani dengan preprocessing yang lebih robust.

**Implikasi**
- Median dipilih untuk imputasi numerik
- `RobustScaler` dipilih agar model yang sensitif terhadap skala tidak terlalu dipengaruhi outlier

## Slide 6 - EDA: Hubungan Fitur dengan Target

**Judul**
EDA

**Isi slide**
- Eksplorasi fitur kategorikal terhadap target menunjukkan beberapa variabel sangat informatif untuk klasifikasi.
- Fitur yang paling membedakan target secara visual adalah:
  - `exercise angina`
  - `ST slope`
  - `chest pain type`
- Hasil korelasi Spearman terhadap target juga mendukung temuan tersebut:
  - `ST slope` = **0.5917**
  - `chest pain type` = **0.5116**
  - `exercise angina` = **0.4943**
  - `oldpeak` = **0.4190**
  - `max heart rate` = **-0.4048**
- Tidak ditemukan korelasi antarf fitur yang sangat ekstrem, sehingga tidak ada kebutuhan kuat untuk membuang fitur karena multikolinearitas berat.

**Kesimpulan slide**
- Meskipun sebagian fitur direpresentasikan dengan angka, maknanya tetap kategorikal. Karena itu, One-Hot Encoding menjadi pilihan yang lebih tepat untuk modeling.

## Slide 7 - Preprocessing Berdasarkan EDA

**Judul**
Preprocessing

**Isi slide**
- Seluruh keputusan preprocessing pada Tugas 3 dibuat berdasarkan temuan dari EDA, bukan dipilih secara acak.
- Langkah preprocessing yang diterapkan adalah:
  - menghapus **272 baris duplikat** atau sekitar **22.86%** data,
  - mengubah nilai tidak valid menjadi missing,
  - membagi data train dan test dengan **stratify** agar proporsi kelas tetap seimbang,
  - melakukan imputasi missing:
    - numerik menggunakan **median**,
    - kategorikal menggunakan **most frequent**,
  - melakukan **One-Hot Encoding** untuk fitur kategorikal,
  - melakukan **RobustScaler** untuk fitur numerik.
- Hasil akhir setelah preprocessing awal:
  - jumlah data sebelum penghapusan duplikat = **1190**
  - jumlah data sesudah penghapusan duplikat = **918**
  - data train = **734**
  - data test = **184**

**Pesan utama**
- Pada Tugas 3, preprocessing bukan sekadar tahap teknis, tetapi bagian penting untuk memperkuat validitas evaluasi model.

## Slide 8 - Model Klasifikasi

**Judul**
Model Klasifikasi

**Isi slide**
- Model yang digunakan tetap dibuat konsisten dengan Tugas 2 dan paper acuan, sehingga perbandingan tetap relevan.
- Model yang dibandingkan:
  - Logistic Regression
  - KNN
  - SVM (RBF)
  - Random Forest
  - Gradient Boosting
  - Stacking Classifier
- Dengan menjaga kumpulan model yang sama, fokus analisis di Tugas 3 bisa diarahkan pada kualitas evaluasi, bukan pada variasi algoritma.

**Pesan utama**
- Perubahan terbesar pada Tugas 3 bukan menambah model baru, tetapi memperbaiki cara model dievaluasi.

## Slide 9 - Skenario Eksperimen

**Judul**
Model Klasifikasi

**Subjudul**
Skenario

**Isi slide**
- Setiap model dibangun di dalam **pipeline** yang menggabungkan preprocessing dan model klasifikasi.
- Pendekatan pipeline penting agar seluruh proses imputasi, encoding, dan scaling terjadi secara konsisten pada setiap fold maupun pada data test.
- Validasi utama dilakukan menggunakan **StratifiedKFold 5-fold**.
- Alasan memilih stratified CV:
  - menjaga proporsi kelas di setiap fold,
  - lebih sesuai untuk klasifikasi biner,
  - mengurangi bias evaluasi akibat pembagian data tertentu.
- Selain cross-validation, seluruh model juga dievaluasi pada **data test** agar dapat dibandingkan:
  - mana model yang performanya tinggi secara rata-rata,
  - mana model yang stabil saat diuji pada hold-out data.

**Pesan utama**
- Slide ini menjelaskan bahwa Tugas 3 menekankan reliabilitas evaluasi, bukan sekadar skor akhir.

## Slide 10 - Konfigurasi Model

**Judul**
Model Klasifikasi

**Subjudul**
Konfigurasi Model

**Isi slide**
- **Logistic Regression**
  - `max_iter = 3000`
  - `class_weight = balanced`
- **KNN**
  - `n_neighbors = 11`
  - `weights = distance`
- **SVM (RBF)**
  - `kernel = rbf`
  - `C = 10`
  - `probability = True`
  - `class_weight = balanced`
- **Random Forest**
  - `n_estimators = 300`
  - `class_weight = balanced`
- **Gradient Boosting**
  - menggunakan konfigurasi standar dengan `random_state = 42`
- **Stacking Classifier**
  - base estimator: Logistic Regression, Random Forest, Gradient Boosting
  - final estimator: Logistic Regression
  - internal CV = 5

**Catatan penting**
- Slide ini menggantikan slide hyperparameter search space pada Tugas 2, karena di Tugas 3 fokusnya bukan tuning, melainkan evaluasi yang lebih ketat.

## Slide 11 - Metrics

**Judul**
Metrics

**Isi slide**
- Evaluasi model dilakukan dengan multi-metrik agar performa model tidak dinilai hanya dari satu sudut pandang.
- Metrik yang digunakan:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **ROC-AUC**
  - **Cohen’s Kappa**
  - **Matthews Correlation Coefficient (MCC)**
- Dalam konteks klasifikasi medis:
  - **Recall** penting untuk menangkap sebanyak mungkin kasus positif,
  - **Precision** penting agar false alarm tidak terlalu tinggi,
  - **F1** membantu melihat keseimbangan precision dan recall,
  - **ROC-AUC** membantu melihat kualitas pemisahan kelas secara umum.

**Pesan utama**
- Karena itu, Tugas 3 lebih menekankan interpretasi gabungan beberapa metrik, terutama F1 dan ROC-AUC.

## Slide 12 - Hasil Cross-Validation

**Judul**
Comparison Cross-Validation

**Isi slide**
- Hasil cross-validation menunjukkan performa rata-rata model pada 5 fold stratified CV.
- Jika diurutkan berdasarkan **CV ROC-AUC**, hasilnya adalah:
  - **Stacking** = **0.9296**
  - **Random Forest** = **0.9275**
  - **Logistic Regression** = **0.9221**
  - **Gradient Boosting** = **0.9186**
  - **KNN** = **0.9141**
  - **SVM (RBF)** = **0.9003**
- Untuk metrik **CV F1**, skor tertinggi diperoleh oleh:
  - **KNN** = **0.8775**
  - **Stacking** = **0.8737**
  - **Random Forest** = **0.8736**
- Temuan penting:
  - model ensemble tetap kuat,
  - Stacking menjadi model terbaik jika prioritasnya adalah ROC-AUC rata-rata pada CV,
  - KNN tetap kompetitif terutama pada F1.

**Interpretasi**
- Cross-validation memberi gambaran yang lebih stabil tentang performa umum model dibanding hanya sekali split data.

## Slide 13 - Hasil pada Data Test

**Judul**
Comparison Test Set

**Isi slide**
- Setelah model dilatih pada data train, performa akhir juga diuji pada data test.
- Jika diurutkan berdasarkan **Test ROC-AUC**, hasilnya adalah:
  - **KNN** = **0.9448**
  - **Stacking** = **0.9341**
  - **Logistic Regression** = **0.9321**
  - **Random Forest** = **0.9306**
  - **Gradient Boosting** = **0.9142**
  - **SVM (RBF)** = **0.9137**
- Untuk metrik **Test F1**, model terbaik juga adalah:
  - **KNN** = **0.9216**
- Artinya, pada hold-out data, KNN menghasilkan performa paling tinggi di antara model lain.

**Interpretasi**
- Hasil ini menarik karena model terbaik pada CV ROC-AUC adalah Stacking, tetapi model terbaik pada test ROC-AUC justru KNN.
- Karena itu, evaluasi tidak cukup hanya melihat model terbaik di satu tabel saja.

## Slide 14 - Perbandingan CV vs Test

**Judul**
Comparison CV vs Test

**Isi slide**
- Perbandingan antara skor CV dan skor test penting untuk menilai kestabilan model.
- Jika gap test jauh lebih tinggi dari CV, bisa jadi model sedang diuntungkan oleh pembagian test tertentu.
- Jika gap test jauh lebih rendah dari CV, ada risiko overfitting.
- Beberapa contoh gap ROC-AUC:
  - **Stacking**: `+0.0045`
  - **Random Forest**: `+0.0031`
  - **Gradient Boosting**: `-0.0045`
  - **Logistic Regression**: `+0.0099`
  - **SVM**: `+0.0134`
  - **KNN**: `+0.0307`
- Interpretasi hasil:
  - **Stacking** dan **Random Forest** terlihat paling stabil karena selisih CV dan test kecil,
  - **KNN** memang paling tinggi pada test, tetapi gap terhadap CV lebih besar sehingga perlu dibaca secara hati-hati.

**Pesan utama**
- Model terbaik tidak selalu hanya yang skornya paling tinggi, tetapi juga yang paling konsisten.

## Slide 15 - Analisis Perbedaan dengan Tugas 2

**Judul**
Analisis Perbedaan

**Isi slide**
- Tugas 2 lebih berfokus pada **replikasi paper** dan **perbandingan tuning hyperparameter**.
- Tugas 3 berfokus pada **perbaikan kualitas proses modeling**.
- Beberapa perbaikan utama pada Tugas 3 dibanding Tugas 2:
  - EDA poin 3-10 disajikan lebih lengkap,
  - duplikat diidentifikasi dan dihapus,
  - nilai tidak valid ditangani sebagai missing,
  - preprocessing disusun berdasarkan hasil eksplorasi,
  - evaluasi menggunakan **Stratified Cross-Validation**,
  - hasil CV dibandingkan langsung dengan hasil pada data test.
- Dengan demikian, Tugas 3 tidak hanya menunjukkan angka performa, tetapi juga memberi alasan metodologis mengapa angka tersebut lebih dapat dipercaya.

**Kesimpulan slide**
- Jika Tugas 2 menjawab “model mana yang bagus”, maka Tugas 3 lebih menjawab “apakah proses evaluasi modelnya sudah reliabel”.

## Slide 16 - Evaluasi Hasil

**Judul**
Evaluasi Hasil

**Isi slide**
1. Seluruh tahapan eksplorasi data, preprocessing, dan pemodelan berhasil dijalankan dengan pendekatan yang lebih kuat secara metodologis.
2. Penghapusan duplikat dan recoding nilai tidak valid membantu meningkatkan kualitas data sebelum model dilatih.
3. Stratified cross-validation memberikan gambaran performa yang lebih adil untuk klasifikasi biner dibanding hanya mengandalkan sekali split train-test.
4. Berdasarkan rata-rata **CV ROC-AUC**, model terbaik adalah **Stacking**.
5. Berdasarkan **Test ROC-AUC**, model terbaik adalah **KNN**.
6. Dari sisi konsistensi performa antara CV dan test, **Stacking** dan **Random Forest** tampak lebih stabil dibanding model lain.

**Pesan penutup**
- Secara keseluruhan, Tugas 3 menghasilkan evaluasi model yang lebih reliabel dibanding Tugas 2.

## Slide 17 - Penutup

**Judul**
Thank You

**Isi slide**
- Thank you for your attention
- Questions and discussion

**Saran tambahan**
- Jika ingin, slide terakhir bisa diberi satu kalimat penutup seperti:
  “Perbaikan metodologi membantu menghasilkan evaluasi model yang lebih kuat dan dapat dipertanggungjawabkan.”

## Saran Visual Per Slide

- Slide 2: screenshot sumber dataset
- Slide 3: tabel ringkasan fitur dan target
- Slide 4: tabel invalid values / missing recoding
- Slide 5: boxplot dan histogram fitur numerik
- Slide 6: countplot kategorikal vs target dan heatmap Spearman
- Slide 7: diagram alur preprocessing
- Slide 8: ikon atau blok daftar model
- Slide 9: diagram pipeline + stratified CV
- Slide 10: tabel konfigurasi tiap model
- Slide 11: ikon atau tabel penjelasan metrik
- Slide 12: tabel hasil cross-validation
- Slide 13: tabel hasil test + ROC curve / confusion matrix
- Slide 14: grafik perbandingan CV vs test
- Slide 15: tabel singkat “Task 2 vs Task 3”
- Slide 16: summary box
