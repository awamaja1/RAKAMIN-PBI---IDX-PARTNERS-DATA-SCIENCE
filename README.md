## Project Credit Risk Prediction

### Overview

Proyek ini bertujuan membangun model machine learning untuk memprediksi risiko kredit (credit risk) pada dataset pinjaman (2007-2014). Model akan membantu perusahaan multifinance dalam menilai dan mengelola risiko kredit agar keputusan bisnis lebih optimal dan kerugian dapat diminimalisir.

### Dataset

* File: `data/loan_data_2007_2014.csv` (letakkan di folder `data/` setelah diunduh manual dari sumber)
* Data Dictionary: [Link Data Dictionary](https://docs.google.com/spreadsheets/d/1iT1JNOBwU4l616_rnJpo0iny7blZvNBs/edit?gid=1666154857)

### Struktur Project

```
project-root/
├── notebooks/
│   └── notebook.ipynb         # Notebook eksplorasi dan eksperimentasi
├── data/
│   └── loan_data_2007_2014.csv # Dataset pinjaman
├── models/
│   └── credit_risk_model.pkl  # Hasil penyimpanan model terlatih
├── utils/
│   └── preprocess.py          # Fungsi pra-pemrosesan data
├── app/
│   └── app.py                 # (Opsional) Aplikasi sederhana untuk inferensi
├── infografis/
│   └── infographic.png        # Hasil infografis presentasi end-to-end solution
├── README.md                  # Dokumen ini
├── requirements.txt           # Daftar dependensi
```

### Langkah Pengerjaan

1. **Data Understanding**

   * Memuat dataset (`pandas.read_csv`), ringkasan struktur (jumlah baris, kolom).
   * Identifikasi atribut berdasarkan data dictionary: tipe data, arti, nilai unik.
   * Statistik deskriptif awal: mean, median, distribusi, missing values, duplikat.
   * Visualisasi sederhana: histogram untuk variabel numerik, bar plot untuk kategorikal.

2. **Exploratory Data Analysis (EDA)**

   * Visualisasi hubungan antar fitur: scatter plot, box plot, heatmap korelasi.
   * Analisis univariat: distribusi fitur, outlier.
   * Analisis bivariat: fitur vs target (jika label sudah disiapkan), melihat perbedaan distribusi.
   * Analisis korelasi multivariat: korelasi antar fitur, identifikasi multikolinearitas.
   * Eksplorasi kreatif (opsional): clustering awal untuk mencari segmen peminjam.

3. **Data Preparation**

   * Penentuan label: identifikasi kolom performa kredit, menetapkan GOOD/BAD berdasarkan definisi (mis. keterlambatan pembayaran).
   * Tangani missing values: hapus kolom/row atau imputasi (mean/median/mode atau metode lain).
   * Tangani duplikat: hapus baris duplikat jika ada.
   * Tangani outlier: identifikasi (IQR atau z-score), dan putuskan aksi (hapus atau transformasi).
   * Encoding variabel kategorikal: one-hot encoding atau target encoding jika banyak kategori.
   * Feature engineering: buat variabel baru jika perlu (mis. rasio utang, durasi pinjaman, dsb.).
   * Scaling: standarisasi atau normalisasi pada fitur numerik jika algoritma membutuhkan.
   * Split data: train/test (mis. 80/20), dapat gunakan stratified split jika label tidak seimbang.

4. **Modeling**

   * Pilih algoritma: wajib Logistic Regression, dan minimal satu algoritma lain (mis. Random Forest, XGBoost, Gradient Boosting, SVM).
   * Pipeline: integrasi preprocessing + model dengan `sklearn.pipeline.Pipeline`.
   * Hyperparameter tuning: `GridSearchCV` atau `RandomizedSearchCV` dengan cross-validation.
   * Menangani imbalance label: teknik oversampling (SMOTE) atau class weight.
   * Latih model pada train set.

5. **Evaluation**

   * Evaluasi pada test set: metrik relevan seperti ROC-AUC, precision, recall, F1-score, confusion matrix.
   * Visualisasi ROC curve, precision-recall curve.
   * Bandingkan performa dua model (Logistic Regression vs alternatif).
   * Analisis overfitting/underfitting: perbandingan performa train vs validation.
   * Feature importance atau koefisien model: interpretasi kontribusi fitur.

6. **Dokumentasi**

   * Notebook (`notebooks/notebook.ipynb`): dokumentasi langkah demi langkah disertai visualisasi dan komentar.
   * Script modul: `utils/preprocess.py` berisi fungsi-fungsi pra-pemrosesan (mis. fungsi load\_data, clean\_data, feature\_engineering).
   * Script main training: `train_model.py` (opsional) untuk melatih dan menyimpan model.
   * Simpan model: `models/credit_risk_model.pkl`.
   * Infografis: ringkas proses end-to-end (diagram alur, hasil kunci, metrik utama, rekomendasi bisnis).
   * README: panduan setup dan cara menjalankan notebook serta interpretasi hasil.

### Panduan Setup

1. Buat virtual environment (venv atau conda).
2. Install dependencies: `pip install -r requirements.txt`.
3. Pastikan file dataset (`loan_data_2007_2014.csv`) tersedia di folder `data/`.
4. Jalankan notebook di `notebooks/notebook.ipynb`.

### Contoh `requirements.txt`

```
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn
jupyterlab
joblib
```

### Contoh `utils/preprocess.py`

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_data(path):
    df = pd.read_csv(path)
    return df

# Tambahkan fungsi clean_data, feature_engineering, encode_features, scale_features, dsb.
```

### Panduan Notebook

1. **Section 1: Setup**

   * Import library, load dataset, tampilkan head dan info.
2. **Section 2: Data Understanding**

   * Ringkasan statistik, missing values, tipe data.
3. **Section 3: EDA**

   * Visualisasi distribusi, korelasi, analisis label.
4. **Section 4: Data Preparation**

   * Tentukan label, bersihkan data, encoding, scaling, split.
5. **Section 5: Modeling**

   * Definisikan pipeline Logistic Regression dan pipeline alternatif.
   * Hyperparameter tuning.
6. **Section 6: Evaluation**

   * Evaluasi model, visualisasi ROC, confusion matrix, interpretasi.
7. **Section 7: Kesimpulan dan Rekomendasi Bisnis**

   * Ringkas temuan penting, rekomendasi untuk implementasi.

### Infografis

* Gunakan alat desain pilihan (PowerPoint, Canva, dsb.).
* Sertakan alur proses (Data Understanding → EDA → Preprocessing → Modeling → Evaluation).
* Tampilkan metrik utama (mis. ROC-AUC Logistic vs alternatif).
* Insight kunci (fitur paling berpengaruh, saran mitigasi risiko).

---

Silakan tinjau README ini sebagai panduan. Langkah selanjutnya: membuat notebook dengan template kode untuk memulai EDA dan preprocessing.

### Catatan
dalam konteks penilaian risiko kredit, false negative (meminjamkan ke orang yang akhirnya gagal bayar) jauh lebih berbahaya dibandingkan false positive (tidak meminjamkan ke orang yang sebenarnya bisa membayar). Ini adalah pendekatan konservatif yang umum di industri finansial.

🔍 Implikasi Strategi: Conservative Risk Labeling
Berdasarkan logika:

Lebih baik kehilangan peluang pinjaman yang aman daripada memberikan pinjaman ke calon gagal bayar.

Maka kita perlu memperluas definisi risiko BAD (target = 1) untuk mengurangi risiko false negative.

✅ Revisi Label: BAD vs GOOD
* loan_status	Risiko	target
* Charged Off	BAD	1
* Default	BAD	1
* Late (31-120 days)	BAD	1
* Late (16-30 days)	BAD	1
* Does not meet the credit policy. Status:Charged Off	BAD	1
* In Grace Period	BAD	1
* Current	BAD	1
* Fully Paid	GOOD	0
* Does not meet the credit policy. Status:Fully Paid	GOOD	0

🔴 Alasan: Current & In Grace Period artinya belum telat bayar tapi belum lunas juga. Dalam konservatif view, itu tetap berisiko.

The image displays a Receiver Operating Characteristic (ROC) curve for a Logistic Regression model, showing a perfect classification performance with an Area Under the Curve (AUC) of 1.00. 
ROC Curve:
This graph illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. 
Logistic Regression:
A statistical model used for binary classification tasks, predicting the probability of a specific class. 
True Positive Rate (TPR):
Also known as sensitivity, it measures the proportion of actual positive cases that are correctly identified. 
False Positive Rate (FPR):
It measures the proportion of actual negative cases that are incorrectly identified as positive. 
AUC (Area Under the Curve):
A single metric summarizing the overall performance of a binary classifier, representing the area under the ROC curve. An AUC of 1.0 indicates a perfect classifier. 

!!Top 5 Feature Importances (XGBoost)
initial_list_status_f
term_36 months -
tot_cur_bal
sub_grade_C2-
verification_status_Source Verified