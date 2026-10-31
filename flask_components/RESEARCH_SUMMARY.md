
# PENERAPAN METODE SUPPORT VECTOR REGRESSION
## DALAM MEMPREDIKSI HARGA BAHAN PANGAN DI KABUPATEN LANGKAT

**Peneliti:** Haris  
**Periode Data:** 2020-01-02 s/d 2024-12-31  
**Completed:** 2026-10-31

### ABSTRACT

Penelitian ini mengembangkan model prediksi harga bahan pangan (beras) di Kabupaten Langkat menggunakan algoritma Support Vector Regression (SVR) dengan kernel Radial Basis Function (RBF). Dataset terdiri dari 1258 hari data time series dengan 6 kategori beras berbeda.

### METODOLOGI

- **Algoritma:** Support Vector Regression (SVR)
- **Kernel:** RBF (Radial Basis Function)
- **Parameter:** C=100, epsilon=0.1, gamma='scale'
- **Features:** Time-based features, lag features (1, 7, 30 hari), rolling mean (7, 30 hari)
- **Data Split:** Chronological split - 80% training, 20% testing
- **Preprocessing:** StandardScaler untuk normalisasi features

### HASIL UTAMA

**Performa Rata-rata:**
- **MAE:** Rp 68
- **RMSE:** Rp 101
- **MAPE:** 0.49%

**Kategori Terbaik:** Beras Kualitas Super I
- MAPE: 0.31%

**Kategori Terburuk:** Beras Kualitas Super II
- MAPE: 0.61%

### KESIMPULAN

Model SVR menunjukkan performa sangat baik (≤10%) untuk prediksi harga beras dengan tingkat error rata-rata 0.49%. Sistem telah diimplementasikan dalam bentuk web application dengan Flask untuk prediksi real-time.

### IMPLEMENTASI

- ✅ 6 SVR Models (satu untuk setiap kategori)
- ✅ Web Application dengan Flask
- ✅ Real-time Prediction API
- ✅ Analytics Dashboard dengan Plotly
- ✅ Time Series Visualization

**Keywords:** Support Vector Regression, SVR, RBF Kernel, Price Prediction, Food Commodities, Time Series, Machine Learning

**Metrik Evaluasi:** MAE (Mean Absolute Error), RMSE (Root Mean Square Error), MAPE (Mean Absolute Percentage Error)
