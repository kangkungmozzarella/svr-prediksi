
# SETUP INSTRUCTIONS - PREDIKSI HARGA BAHAN PANGAN KABUPATEN LANGKAT

## 1. INSTALASI DEPENDENCIES
```bash
pip install -r flask_components/requirements.txt
```

## 2. STRUKTUR PROJECT FLASK
```
your_flask_project/
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   ├── index.html                 # Home page
│   ├── predict.html               # Prediction form
│   └── dashboard.html             # Analytics dashboard
├── static/                        # CSS, JS, images
│   ├── css/
│   └── js/
└── flask_components/              # ML components (copy dari Jupyter)
    ├── models/                    # Trained SVR models
    ├── data/                      # Datasets
    ├── results/                   # Analysis results
    └── visualizations/            # Interactive charts
```

## 3. SAMPLE FLASK APP CODE

### app.py
```python
from flask import Flask, render_template, request, jsonify
import pickle
import json
import pandas as pd

app = Flask(__name__)

# Load ML pipeline
with open('flask_components/models/complete_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

with open('flask_components/models/prediction_functions.pkl', 'rb') as f:
    prediction_functions = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        kategori = request.form['kategori']

        input_data = {
            'Year': int(request.form['year']),
            'Month': int(request.form['month']),
            'Day': int(request.form['day']),
            'DayOfWeek': int(request.form['dayofweek']),
            'DayOfYear': int(request.form['dayofyear']),
            'Quarter': int(request.form['quarter']),
            f'{kategori}_lag1': float(request.form['lag1']),
            f'{kategori}_lag7': float(request.form['lag7']),
            f'{kategori}_lag30': float(request.form['lag30']),
            f'{kategori}_rolling_mean_7': float(request.form['rolling7']),
            f'{kategori}_rolling_mean_30': float(request.form['rolling30'])
        }

        # Predict
        result = prediction_functions['predict_price'](input_data, kategori, pipeline)

        return jsonify(result)

    # GET request - show form
    categories = pipeline['kategori_beras']
    return render_template('predict.html', categories=categories)

@app.route('/dashboard')
def dashboard():
    # Load analytics data
    with open('flask_components/results/final_performance_summary.json', 'r') as f:
        performance = json.load(f)

    return render_template('dashboard.html', performance=performance)

@app.route('/api/historical_data')
def historical_data():
    # Return historical data untuk charts
    df = pd.read_csv('flask_components/data/sorted_dataset.csv')

    # Convert ke format JSON
    data = {
        'dates': df['Tanggal'].tolist(),
        'prices': {
            kategori: df[kategori].tolist()
            for kategori in pipeline['kategori_beras']
        }
    }

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
```

## 4. DEPLOYMENT

Untuk deployment ke production, gunakan:
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## 5. FITUR YANG TERSEDIA

- ✅ Prediksi harga beras real-time untuk 6 kategori
- ✅ Support Vector Regression dengan kernel RBF
- ✅ Analytics dashboard dengan Plotly
- ✅ Time series visualization
- ✅ Model performance metrics (MAE, RMSE, MAPE)
- ✅ Error analysis per tahun
- ✅ RESTful API endpoints

## 6. CATATAN PENTING

### Feature Engineering
Model membutuhkan lag features dan rolling mean features:
- lag1: Harga 1 hari sebelumnya
- lag7: Harga 7 hari sebelumnya
- lag30: Harga 30 hari sebelumnya
- rolling_mean_7: Rata-rata 7 hari
- rolling_mean_30: Rata-rata 30 hari

### Kategori Beras
1. Beras Kualitas Bawah I
2. Beras Kualitas Bawah II
3. Beras Kualitas Medium I
4. Beras Kualitas Medium II
5. Beras Kualitas Super I
6. Beras Kualitas Super II

### Model Performance
- Metode: Support Vector Regression (SVR) dengan kernel RBF
- Metrik Evaluasi: MAE, RMSE, MAPE
- Data: Time series harian 2020-2024
