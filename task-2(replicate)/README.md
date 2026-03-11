# Task 2 - Replikasi Metodologi Artikel

Folder ini disiapkan untuk workflow **Google Colab** (bukan local runtime).
Scope di folder ini: **replikasi dan hasil eksperimen**. Analisis perbandingan/report dikerjakan terpisah.

## Isi Folder

- `notebook/replikasi_metodologi.ipynb`: notebook utama untuk menjalankan replikasi end-to-end.
- `config/experiment.yaml`: konfigurasi data, model, dan search space.
- `requirements.txt`: dependency Python (untuk `%pip install` di Colab).
- `data/raw/`: lokasi dataset `statlog_cleveland_hungary_final.csv`.
- `results/tables/`: output CSV hasil evaluasi.
- `results/figures/`: output confusion matrix, ROC, dan perbandingan Grid vs Random.
- `submission_bundle.zip`: paket submit siap upload.

## Cara Menjalankan di Google Colab

1. Upload folder `task-2(replicate)` ke Google Drive.
2. Buka `notebook/replikasi_metodologi.ipynb` di Colab.
3. Jalankan semua sel dari atas ke bawah.
4. Jika download dataset otomatis gagal, upload manual CSV ke:
   - `task-2(replicate)/data/raw/statlog_cleveland_hungary_final.csv`
5. Setelah selesai, ambil file:
   - `task-2(replicate)/submission_bundle.zip`

## Catatan Metodologi

- Protokol anti-leakage:
  - Split train/test dilakukan dulu.
  - Preprocessing fit hanya di data train melalui `Pipeline` + `CV`.
- Model:
  - Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting.
  - Stacking dievaluasi pada Grid Search (sesuai scope tabel paper).
- Metrik:
  - Accuracy, Precision, Recall, F1, AUC, Kappa, MCC.

## Struktur Output Utama

- `results/tables/hasil_grid_search.csv`
- `results/tables/hasil_random_search.csv`
- `results/tables/data_profile.csv`
- `results/tables/run_summary.json`
- `submission_bundle.zip`
