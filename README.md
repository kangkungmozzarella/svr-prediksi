# Prediksi Harga Bahan Pangan (Beras) - Kabupaten Langkat

## Deskripsi Proyek
Penerapan metode **Support Vector Regression (SVR)** dengan kernel RBF untuk memprediksi harga bahan pangan (beras) di Kabupaten Langkat, Sumatera Utara, menggunakan data historis harga periode 2020-2024 yang dikombinasikan dengan **faktor cuaca** (Temperatur & Curah Hujan) sebagai variabel eksogen tambahan.

## Dataset
- **Sumber:** harga harian 6 kategori beras (2020-2024, 1.258 hari) + data cuaca harian (Temperatur, Curah Hujan)
- **Kategori beras:** Bawah I, Bawah II, Medium I, Medium II, Super I, Super II
- Data digabung berdasarkan tanggal, lalu di-preprocess menjadi 1.228 baris (setelah drop NaN akibat fitur lag/rolling)

## Fitur (Feature Engineering)
- **Time features:** Year, Month, Day, DayOfWeek, DayOfYear, Quarter
- **Weather features:** Temperatur, Curah_Hujan, Temperatur_rolling_7, Curah_Hujan_rolling_7, Curah_Hujan_rolling_30
- **Lag features (per kategori):** lag1, lag7, lag30
- **Rolling features (per kategori):** rolling_mean_7, rolling_mean_30

Untuk prediksi 2 tahun ke depan, cuaca masa depan tidak diketahui sehingga diestimasi menggunakan **klimatologi** (rata-rata historis per hari-dalam-tahun, 2020-2024).

## Performa Model (Testing Set)
- Algoritma: SVR kernel RBF (C=100, epsilon=0.1, gamma='scale')
- Split: Chronological 80% training (982 hari) / 20% testing (246 hari)
- **Average MAPE: ±5.66%** (interpretasi: Sangat Baik)
- **Average MAE: ±Rp 779**
- Kategori dengan prediksi terbaik: Beras Kualitas Medium II

> Korelasi cuaca terhadap harga relatif lemah (|r| < 0.2) — cuaca bukan faktor dominan dibanding pola harga historis, namun tetap relevan dimasukkan sebagai variabel eksogen sesuai arahan pembimbing.

## Struktur File
```
svr-prediksi/
├── app.py                      # Aplikasi Flask
├── train.ipynb                 # Notebook training & feature engineering SVR
├── dataset.csv                 # Dataset harga beras (wide format)
├── cuaca_harian.xlsx           # Data cuaca harian (Temperatur, Curah Hujan)
├── data_final.xlsx             # Data gabungan harga + cuaca (long format)
├── templates/                  # Template HTML (Jinja2)
│   ├── index.html              # Homepage publik
│   ├── admin/login.html
│   ├── dashboard/
│   ├── analisis/                # Hasil analisis & visualisasi
│   ├── prediksi/                # Prediksi interaktif
│   └── tentang/
├── static/                     # CSS, JS, images, visualisasi (auto-generated)
└── flask_components/            # Output dari train.ipynb, dikonsumsi app.py
    ├── models/                  # svr_models.pkl, svr_scalers.pkl, dll
    ├── data/                    # dataset_info.json, processed_dataset.csv, dll
    ├── results/                 # Metrik evaluasi & prediksi detail
    └── visualizations/          # Chart interaktif (Plotly HTML)
```

## Cara Menjalankan Aplikasi
1. Install dependensi
```bash
pip3 install -r requirements.txt
```

2. Jalankan aplikasi Flask
```bash
python3 app.py
```

3. Buka browser dan akses `http://localhost:5000`

> Jika port 5000 sudah dipakai (umumnya AirPlay Receiver di macOS), matikan AirPlay Receiver di System Settings → General → AirDrop & Handoff, atau jalankan dengan port lain.

### Login Admin
- Username: `admin`
- Password: `admin123`

### Halaman
- `/` — Homepage publik
- `/admin/login` — Login admin
- `/dashboard` — Dashboard ringkasan performa model
- `/analisis` — Analisis lengkap, detail prediksi, dan visualisasi (termasuk faktor cuaca)
- `/prediksi` — Prediksi interaktif (pilih kategori & jumlah hari ke depan)
- `/tentang` — Informasi penelitian & metodologi

## Melatih Ulang Model
Jalankan seluruh sel di `train.ipynb` secara berurutan untuk regenerasi model, metrik, dan visualisasi di folder `flask_components/`.

## Teknologi
Python, Flask, Pandas, NumPy, Scikit-learn, Plotly, Bootstrap 5

## Lokasi Penelitian
Kabupaten Langkat, Sumatera Utara
