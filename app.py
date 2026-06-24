from flask import Flask, redirect, request, render_template, url_for, session, flash, jsonify, send_file
import numpy as np
import pandas as pd
import json
import pickle
import os
from functools import wraps
from datetime import datetime
import pymysql
import csv
from werkzeug.utils import secure_filename
import io

# Warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

app.secret_key = "svr-rice-price-prediction-secret-key-2025"

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Silakan login untuk mengakses halaman ini', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper function untuk generate sample statistics
def generate_sample_stats():
    """Generate sample statistics untuk development"""
    return {
        'total_records': 1228,  # Setelah preprocessing (dari 1258 - 30 untuk lag/rolling)
        'original_records': 1258,
        'data_period': '2020-2024',
        'data_start': '2020-01-02',
        'data_end': '2024-12-31',
        'categories_count': 6,
        'categories': [
            'Beras Kualitas Bawah I',
            'Beras Kualitas Bawah II',
            'Beras Kualitas Medium I',
            'Beras Kualitas Medium II',
            'Beras Kualitas Super I',
            'Beras Kualitas Super II'
        ],
        'price_range': {
            'min': 9700,
            'max': 14800,
            'avg': 12275
        },
        'model_performance': {
            'avg_mape': 5.66,
            'avg_mae': 779,
            'avg_rmse': 834,
            'best_category': 'Beras Kualitas Medium I',
            'worst_category': 'Beras Kualitas Super II'
        },
        'features': {
            'time_features': ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Quarter'],
            'lag_features': ['lag1', 'lag7', 'lag30'],
            'rolling_features': ['rolling_mean_7', 'rolling_mean_30']
        }
    }

@app.route('/')
def homepage():
    """Homepage publik untuk prediksi harga bahan pangan"""
    
    # Default values
    homepage_data = {
        'research_title': 'Prediksi Harga Bahan Pangan',
        'research_location': 'Kabupaten Langkat, Sumatera Utara',
        'total_records': 1228,
        'categories_count': 6,
        'categories_list': [],
        'data_period': '2020-2024',
        'avg_mape': 5.66,
        'avg_mae': 779,
        'avg_rmse': 834,
        'interpretation': 'Sangat Baik',
        'algorithm': 'SVR',
        'kernel': 'RBF',
        'total_models': 6,
        'model_status': 'Active',
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        base_path = os.path.join(os.getcwd(), 'flask_components')
        print("\n" + "="*60)
        print("🏠 LOADING HOMEPAGE DATA")
        print("="*60)
        
        # ===== LOAD DATASET INFO =====
        dataset_path = os.path.join(base_path, 'data', 'dataset_info.json')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
            
            kategori_beras = dataset_info.get('kategori_beras', [])
            print(f"✅ Loaded dataset_info.json - {len(kategori_beras)} categories")
            
            # Get data period
            periode = dataset_info.get('periode', {})
            if periode:
                start_year = periode.get('start', '2020-01-02')[:4]
                end_year = periode.get('end', '2024-12-31')[:4]
                data_period = f"{start_year}-{end_year}"
            else:
                data_period = '2020-2024'
        else:
            print("⚠️ dataset_info.json not found, using defaults")
            kategori_beras = [
                'Beras Kualitas Bawah I',
                'Beras Kualitas Bawah II',
                'Beras Kualitas Medium I',
                'Beras Kualitas Medium II',
                'Beras Kualitas Super I',
                'Beras Kualitas Super II'
            ]
            data_period = '2020-2024'
        
        # ===== LOAD SVR RESULTS =====
        svr_results_path = os.path.join(base_path, 'results', 'svr_results.json')
        if os.path.exists(svr_results_path):
            with open(svr_results_path, 'r', encoding='utf-8') as f:
                svr_results = json.load(f)
            print("✅ Loaded svr_results.json")
            
            # Calculate average metrics
            all_mape = []
            all_mae = []
            all_rmse = []
            
            for k in kategori_beras:
                if k in svr_results:
                    all_mape.append(svr_results[k]['testing_metrics']['mape'])
                    all_mae.append(svr_results[k]['testing_metrics']['mae'])
                    all_rmse.append(svr_results[k]['testing_metrics']['rmse'])
            
            if all_mape:
                avg_mape = sum(all_mape) / len(all_mape)
                avg_mae = sum(all_mae) / len(all_mae)
                avg_rmse = sum(all_rmse) / len(all_rmse)
                print(f"📈 Avg MAPE: {avg_mape:.2f}%")
                print(f"📈 Avg MAE: Rp {avg_mae:.0f}")
                print(f"📈 Avg RMSE: Rp {avg_rmse:.0f}")
            else:
                avg_mape = homepage_data['avg_mape']
                avg_mae = homepage_data['avg_mae']
                avg_rmse = homepage_data['avg_rmse']
                print("⚠️ Using default metrics")
        else:
            print("⚠️ svr_results.json not found, using defaults")
            avg_mape = homepage_data['avg_mape']
            avg_mae = homepage_data['avg_mae']
            avg_rmse = homepage_data['avg_rmse']
        
        # ===== LOAD SPLITTING INFO =====
        splitting_path = os.path.join(base_path, 'data', 'splitting_info.json')
        total_records = 1228  # Default
        
        if os.path.exists(splitting_path):
            with open(splitting_path, 'r', encoding='utf-8') as f:
                splitting_info = json.load(f)
            
            # Try different keys
            if 'total_samples' in splitting_info:
                total_records = splitting_info['total_samples']
            elif 'train_size' in splitting_info and 'test_size' in splitting_info:
                total_records = splitting_info['train_size'] + splitting_info['test_size']
            
            print(f"✅ Loaded splitting_info.json - {total_records} records")
        else:
            print(f"⚠️ splitting_info.json not found, using default: {total_records}")
        
        # ===== INTERPRETATION =====
        if avg_mape <= 10:
            interpretation = "Sangat Baik"
        elif avg_mape <= 20:
            interpretation = "Baik"
        elif avg_mape <= 30:
            interpretation = "Cukup"
        else:
            interpretation = "Perlu Perbaikan"
        
        # ===== UPDATE HOMEPAGE DATA =====
        homepage_data.update({
            'research_title': 'Prediksi Harga Bahan Pangan',
            'research_location': 'Kabupaten Langkat, Sumatera Utara',
            'total_records': total_records,
            'categories_count': len(kategori_beras),
            'categories_list': kategori_beras,
            'data_period': data_period,
            'avg_mape': avg_mape,
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse,
            'interpretation': interpretation,
            'algorithm': 'SVR',
            'kernel': 'RBF',
            'total_models': len(kategori_beras),
            'model_status': 'Active',
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        print(f"\n✅ Homepage loaded successfully")
        print(f"   Total Records: {total_records}")
        print(f"   Categories: {len(kategori_beras)}")
        print(f"   Avg MAPE: {avg_mape:.2f}%")
        print(f"   Interpretation: {interpretation}")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        error_msg = f'File tidak ditemukan: {str(e)}'
        print(f"❌ File Error: {error_msg}")
        
    except Exception as e:
        error_msg = f'Error loading homepage: {str(e)}'
        print(f"❌ Error: {error_msg}")
        import traceback
        traceback.print_exc()
    
    return render_template('index.html', **homepage_data)

#-------Login---------#
@app.route('/admin/login', methods=['GET', 'POST'])
def login():
    """Login admin untuk dashboard"""
    if request.method == 'POST':
        username = request.form['username'] 
        password = request.form['password']

        # Simple authentication
        if username == 'admin' and password == 'admin123':
            session['username'] = username
            session['status'] = "Login"
            flash(f"Selamat datang di Sistem Prediksi Harga Bahan Pangan, {username}!", 'success')
            return redirect(url_for('dashboard'))
        else:
            flash("Username atau Password Salah", 'danger')

    return render_template('admin/login.html')

@app.route('/logout')
def logout():
    """Logout admin"""
    session.clear()
    flash('Anda telah logout', 'info')
    return redirect(url_for('homepage'))

#-------Dashboard---------#
@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard utama untuk prediksi harga bahan pangan"""
    
    # Default values jika terjadi error
    dashboard_data = {
        'research_title': 'Penerapan Metode Support Vector Regression dalam Memprediksi Harga Bahan Pangan di Kabupaten Langkat',
        'research_location': 'Kabupaten Langkat, Sumatera Utara',
        'data_period': '2020-2024',
        'total_records': 0,
        'original_records': 0,
        'categories_count': 0,
        'categories_list': [],
        'algorithm': 'Support Vector Regression (SVR)',
        'kernel': 'RBF (Radial Basis Function)',
        'total_models': 0,
        'avg_mape': 0.0,
        'avg_mae': 0.0,
        'avg_rmse': 0.0,
        'best_category': 'N/A',
        'best_category_short': 'N/A',
        'worst_category': 'N/A',
        'worst_category_short': 'N/A',
        'interpretation': 'N/A',
        'category_performance': {},
        'model_status': 'Error',
        'data_status': 'Error Loading Data',
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'weather_statistics': {},
        'weather_insight': None,
        'visualizations': {}
    }
    
    try:
        # Path ke flask_components
        base_path = os.path.join(os.getcwd(), 'flask_components')
        print(f"📂 Base path: {base_path}")
        
        # 1. Load SVR Results (metrik performa)
        svr_results_path = os.path.join(base_path, 'results', 'svr_results.json')
        print(f"📄 Loading: {svr_results_path}")
        with open(svr_results_path, 'r', encoding='utf-8') as f:
            svr_results = json.load(f)
        print("✅ Loaded svr_results.json")
        
        # 2. Load Dataset Info
        dataset_info_path = os.path.join(base_path, 'data', 'dataset_info.json')
        print(f"📄 Loading: {dataset_info_path}")
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        print("✅ Loaded dataset_info.json")
        
        # 3. Load Preprocessing Info
        preprocessing_info_path = os.path.join(base_path, 'data', 'preprocessing_info.json')
        print(f"📄 Loading: {preprocessing_info_path}")
        with open(preprocessing_info_path, 'r', encoding='utf-8') as f:
            preprocessing_info = json.load(f)
        print("✅ Loaded preprocessing_info.json")
        
        # 4. Load Splitting Info
        splitting_info_path = os.path.join(base_path, 'data', 'splitting_info.json')
        print(f"📄 Loading: {splitting_info_path}")
        with open(splitting_info_path, 'r', encoding='utf-8') as f:
            splitting_info = json.load(f)
        print("✅ Loaded splitting_info.json")
        
        # Kategori beras
        kategori_beras = dataset_info.get('kategori_beras', [])
        if not kategori_beras:
            raise ValueError("kategori_beras is empty in dataset_info.json")
        
        print(f"📊 Categories found: {len(kategori_beras)}")
        
        # Hitung average metrics dari semua kategori (OTOMATIS)
        all_mape = []
        all_mae = []
        all_rmse = []
        
        for k in kategori_beras:
            if k in svr_results:
                all_mape.append(svr_results[k]['testing_metrics']['mape'])
                all_mae.append(svr_results[k]['testing_metrics']['mae'])
                all_rmse.append(svr_results[k]['testing_metrics']['rmse'])
        
        if not all_mape:
            raise ValueError("No metrics found in svr_results.json")
        
        avg_mape = sum(all_mape) / len(all_mape)
        avg_mae = sum(all_mae) / len(all_mae)
        avg_rmse = sum(all_rmse) / len(all_rmse)
        
        print(f"📈 Avg MAPE: {avg_mape:.2f}%")
        print(f"📈 Avg MAE: Rp {avg_mae:.0f}")
        print(f"📈 Avg RMSE: Rp {avg_rmse:.0f}")
        
        # Best category (MAPE terkecil)
        best_category = min(kategori_beras, key=lambda k: svr_results[k]['testing_metrics']['mape'])
        best_category_short = best_category.replace('Beras Kualitas ', '')
        
        # Worst category (MAPE terbesar)
        worst_category = max(kategori_beras, key=lambda k: svr_results[k]['testing_metrics']['mape'])
        worst_category_short = worst_category.replace('Beras Kualitas ', '')
        
        # Category performance (OTOMATIS dari svr_results.json)
        category_performance = {}
        for kategori in kategori_beras:
            if kategori in svr_results:
                category_performance[kategori] = {
                    'mape': svr_results[kategori]['testing_metrics']['mape'],
                    'mae': svr_results[kategori]['testing_metrics']['mae'],
                    'rmse': svr_results[kategori]['testing_metrics']['rmse']
                }
        
        # Total records (OTOMATIS)
        total_records = splitting_info.get('total_samples', 0)
        if total_records == 0:
            total_records = preprocessing_info.get('total_records', dataset_info.get('total_rows', 0))
        
        original_records = dataset_info.get('total_rows', 0)
        
        # Data period
        start_year = dataset_info.get('periode', {}).get('start', '2020')[:4]
        end_year = dataset_info.get('periode', {}).get('end', '2024')[:4]
        data_period = f"{start_year}-{end_year}"
        
        # Interpretasi MAPE
        if avg_mape <= 10:
            interpretation = "Sangat Baik"
        elif avg_mape <= 20:
            interpretation = "Baik"
        elif avg_mape <= 30:
            interpretation = "Cukup"
        else:
            interpretation = "Perlu Perbaikan"
        
        # ===== FAKTOR CUACA (info terbaru, sesuai revisi penelitian) =====
        weather_statistics = dataset_info.get('weather_statistics', {})
        weather_price_correlation = dataset_info.get('weather_price_correlation', {})

        weather_insight = None
        if weather_price_correlation:
            # Cari korelasi cuaca-harga terkuat (absolut) untuk insight singkat
            strongest = None
            for kategori, corr in weather_price_correlation.items():
                for factor, value in corr.items():
                    if strongest is None or abs(value) > abs(strongest[2]):
                        strongest = (kategori.replace('Beras Kualitas ', ''), factor.replace('_', ' '), value)
            if strongest:
                weather_insight = {
                    'category': strongest[0],
                    'factor': strongest[1],
                    'correlation': strongest[2]
                }

        # ===== COPY VISUALISASI UNTUK DASHBOARD (chart, bukan cuma tabel) =====
        visualizations = {}
        viz_source_path = os.path.join(base_path, 'visualizations')
        if os.path.exists(viz_source_path):
            static_viz_path = os.path.join('static', 'visualizations')
            os.makedirs(static_viz_path, exist_ok=True)

            dashboard_viz_files = {
                'weather_trend': 'trend_cuaca_harian.html',
                'mape_comparison': 'mape_comparison_all_categories.html'
            }
            for key, filename in dashboard_viz_files.items():
                source = os.path.join(viz_source_path, filename)
                if os.path.exists(source):
                    dest = os.path.join(static_viz_path, filename)
                    if not os.path.exists(dest) or os.path.getmtime(source) > os.path.getmtime(dest):
                        import shutil
                        shutil.copy2(source, dest)
                    visualizations[key] = filename

        # Update dashboard_data dengan data real
        dashboard_data.update({
            'research_title': 'Penerapan Metode Support Vector Regression dalam Memprediksi Harga Bahan Pangan di Kabupaten Langkat',
            'research_location': 'Kabupaten Langkat, Sumatera Utara',
            'data_period': data_period,
            'total_records': total_records,
            'original_records': original_records,
            'categories_count': len(kategori_beras),
            'categories_list': kategori_beras,
            'algorithm': 'Support Vector Regression (SVR)',
            'kernel': 'RBF (Radial Basis Function)',
            'total_models': len(kategori_beras),
            'avg_mape': avg_mape,
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse,
            'best_category': best_category,
            'best_category_short': best_category_short,
            'worst_category': worst_category,
            'worst_category_short': worst_category_short,
            'interpretation': interpretation,
            'category_performance': category_performance,
            'model_status': 'Active',
            'data_status': 'Real Data from Jupyter Notebook',
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'weather_statistics': weather_statistics,
            'weather_insight': weather_insight,
            'visualizations': visualizations
        })
        
        print(f"✅ Dashboard loaded with 100% REAL DATA")
        print(f"   Total Records: {total_records}")
        print(f"   Categories: {len(kategori_beras)}")
        print(f"   Best Category: {best_category_short}")
        print(f"   Interpretation: {interpretation}")
        
    except FileNotFoundError as e:
        error_msg = f'File tidak ditemukan: {str(e)}'
        flash(error_msg, 'danger')
        print(f"❌ File Error: {error_msg}")
        dashboard_data['data_status'] = f'Error: File not found'
        
    except KeyError as e:
        error_msg = f'Key tidak ditemukan: {str(e)}'
        flash(error_msg, 'danger')
        print(f"❌ Key Error: {error_msg}")
        dashboard_data['data_status'] = f'Error: Missing key in JSON'
        
    except ValueError as e:
        error_msg = f'Value error: {str(e)}'
        flash(error_msg, 'danger')
        print(f"❌ Value Error: {error_msg}")
        dashboard_data['data_status'] = f'Error: Invalid data'
        
    except Exception as e:
        error_msg = f'Error: {str(e)}'
        flash(error_msg, 'danger')
        print(f"❌ Error: {error_msg}")
        import traceback
        traceback.print_exc()
        dashboard_data['data_status'] = f'Error: {str(e)[:50]}'
    
    return render_template('dashboard/index.html', **dashboard_data)
    
#-------Tentang Route---------#
@app.route('/tentang')
@login_required
def tentang():
    """Halaman tentang penelitian SVR"""
    
    # Default values
    tentang_data = {
        'research_title': 'Penerapan Metode Support Vector Regression dalam Memprediksi Harga Bahan Pangan di Kabupaten Langkat',
        'research_location': 'Kabupaten Langkat, Sumatera Utara',
        'research_year': '2025',
        'algorithm': 'Support Vector Regression (SVR)',
        'kernel': 'RBF (Radial Basis Function)',
        'total_records': 1228,
        'original_records': 1258,
        'categories_count': 6,
        'categories_list': [],
        'data_period': '2020-2024',
        'split_method': 'Chronological Split (80:20)',
        'train_percentage': 80,
        'test_percentage': 20,
        'avg_mape': 5.66,
        'avg_mae': 779,
        'avg_rmse': 834,
        'best_category_short': 'Medium II',
        'interpretation': 'Sangat Baik',
        'total_models': 6,
        'model_parameters': {
            'C': 100,
            'epsilon': 0.1,
            'gamma': 'scale'
        },
        'time_features': ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Quarter'],
        'weather_features': ['Temperatur', 'Curah_Hujan', 'Temperatur_rolling_7', 'Curah_Hujan_rolling_7', 'Curah_Hujan_rolling_30'],
        'lag_features': ['lag_1', 'lag_7', 'lag_30'],
        'rolling_features': ['rolling_mean_7', 'rolling_mean_30'],
        'weather_price_correlation': {},
        'price_min': 9700,
        'price_max': 14800,
        'price_avg': 12275,
        'model_status': 'Active'
    }
    
    try:
        # Path ke flask_components
        base_path = os.path.join(os.getcwd(), 'flask_components')
        print(f"📂 Loading data from: {base_path}")
        
        # 1. Load SVR Results
        svr_results_path = os.path.join(base_path, 'results', 'svr_results.json')
        with open(svr_results_path, 'r', encoding='utf-8') as f:
            svr_results = json.load(f)
        print("✅ Loaded svr_results.json")
        
        # 2. Load Dataset Info
        dataset_info_path = os.path.join(base_path, 'data', 'dataset_info.json')
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        print("✅ Loaded dataset_info.json")
        
        # 3. Load Preprocessing Info
        preprocessing_info_path = os.path.join(base_path, 'data', 'preprocessing_info.json')
        with open(preprocessing_info_path, 'r', encoding='utf-8') as f:
            preprocessing_info = json.load(f)
        print("✅ Loaded preprocessing_info.json")
        
        # 4. Load Splitting Info
        splitting_info_path = os.path.join(base_path, 'data', 'splitting_info.json')
        with open(splitting_info_path, 'r', encoding='utf-8') as f:
            splitting_info = json.load(f)
        print("✅ Loaded splitting_info.json")
        
        # Kategori beras
        kategori_beras = dataset_info.get('kategori_beras', [])
        print(f"📊 Categories: {len(kategori_beras)}")
        
        # Hitung average metrics
        all_mape = []
        all_mae = []
        all_rmse = []
        
        for k in kategori_beras:
            if k in svr_results:
                all_mape.append(svr_results[k]['testing_metrics']['mape'])
                all_mae.append(svr_results[k]['testing_metrics']['mae'])
                all_rmse.append(svr_results[k]['testing_metrics']['rmse'])
        
        if all_mape:
            avg_mape = sum(all_mape) / len(all_mape)
            avg_mae = sum(all_mae) / len(all_mae)
            avg_rmse = sum(all_rmse) / len(all_rmse)
            print(f"📈 Avg MAPE: {avg_mape:.2f}%")
        else:
            avg_mape = tentang_data['avg_mape']
            avg_mae = tentang_data['avg_mae']
            avg_rmse = tentang_data['avg_rmse']
            print("⚠️ Using default metrics")
        
        # Best category
        if kategori_beras and all(k in svr_results for k in kategori_beras):
            best_category = min(kategori_beras, key=lambda k: svr_results[k]['testing_metrics']['mape'])
            best_category_short = best_category.replace('Beras Kualitas ', '')
            print(f"🏆 Best category: {best_category_short}")
        else:
            best_category_short = tentang_data['best_category_short']
        
        # Total records - DENGAN FALLBACK YANG LEBIH BAIK
        total_records = 0
        
        # Coba berbagai key yang mungkin ada
        if 'total_samples' in splitting_info:
            total_records = splitting_info['total_samples']
            print(f"✅ Total records from splitting_info['total_samples']: {total_records}")
        elif 'train_size' in splitting_info and 'test_size' in splitting_info:
            total_records = splitting_info['train_size'] + splitting_info['test_size']
            print(f"✅ Total records from train+test: {total_records}")
        elif 'total_records' in preprocessing_info:
            total_records = preprocessing_info['total_records']
            print(f"✅ Total records from preprocessing_info: {total_records}")
        elif 'records_after_preprocessing' in preprocessing_info:
            total_records = preprocessing_info['records_after_preprocessing']
            print(f"✅ Total records from preprocessing_info: {total_records}")
        elif 'total_rows' in dataset_info:
            # Kurangi 30 karena lag/rolling features drop rows
            total_records = dataset_info['total_rows'] - 30
            print(f"✅ Total records from dataset_info (adjusted): {total_records}")
        else:
            # Fallback ke nilai real dari notebook
            total_records = 1228
            print(f"⚠️ Using fallback total_records: {total_records}")
        
        # Original records
        original_records = dataset_info.get('total_rows', 1258)
        print(f"📊 Original records: {original_records}")
        
        # Data period
        periode = dataset_info.get('periode', {})
        if periode:
            start_year = periode.get('start', '2020-01-02')[:4]
            end_year = periode.get('end', '2024-12-31')[:4]
            data_period = f"{start_year}-{end_year}"
        else:
            data_period = '2020-2024'
        print(f"📅 Data period: {data_period}")
        
        # Interpretasi MAPE
        if avg_mape <= 10:
            interpretation = "Sangat Baik"
        elif avg_mape <= 20:
            interpretation = "Baik"
        elif avg_mape <= 30:
            interpretation = "Cukup"
        else:
            interpretation = "Perlu Perbaikan"
        
        # Features dari preprocessing_info
        time_features = preprocessing_info.get('time_features', tentang_data['time_features'])
        weather_features = preprocessing_info.get('weather_features', tentang_data['weather_features'])
        lag_features = preprocessing_info.get('lag_features', tentang_data['lag_features'])
        rolling_features = preprocessing_info.get('rolling_features', tentang_data['rolling_features'])
        weather_price_correlation = dataset_info.get('weather_price_correlation', {})

        print(f"🔧 Features: {len(time_features)} time, {len(weather_features)} weather, {len(lag_features)} lag, {len(rolling_features)} rolling")
        
        # Price range dari dataset_info
        price_min = tentang_data['price_min']
        price_max = tentang_data['price_max']
        price_avg = tentang_data['price_avg']
        
        descriptive_stats = dataset_info.get('descriptive_statistics', {})
        if descriptive_stats and kategori_beras:
            try:
                all_mins = []
                all_maxs = []
                all_means = []
                
                for k in kategori_beras:
                    if k in descriptive_stats:
                        all_mins.append(descriptive_stats[k].get('min', 0))
                        all_maxs.append(descriptive_stats[k].get('max', 0))
                        all_means.append(descriptive_stats[k].get('mean', 0))
                
                if all_mins:
                    price_min = min(all_mins)
                    price_max = max(all_maxs)
                    price_avg = sum(all_means) / len(all_means)
                    print(f"💰 Price range: Rp {price_min:,.0f} - Rp {price_max:,.0f} (avg: Rp {price_avg:,.0f})")
            except Exception as e:
                print(f"⚠️ Error calculating price range: {e}")
        
        # Update dengan data real
        tentang_data.update({
            'total_records': total_records,
            'original_records': original_records,
            'categories_count': len(kategori_beras),
            'categories_list': kategori_beras,
            'data_period': data_period,
            'avg_mape': avg_mape,
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse,
            'best_category_short': best_category_short,
            'interpretation': interpretation,
            'total_models': len(kategori_beras),
            'time_features': time_features,
            'weather_features': weather_features,
            'lag_features': lag_features,
            'rolling_features': rolling_features,
            'weather_price_correlation': weather_price_correlation,
            'price_min': price_min,
            'price_max': price_max,
            'price_avg': price_avg
        })
        
        print(f"✅ Tentang page loaded successfully")
        print(f"   Total Records: {total_records}")
        print(f"   Categories: {len(kategori_beras)}")
        print(f"   Avg MAPE: {avg_mape:.2f}%")
        print(f"   Best: {best_category_short}")
        
    except FileNotFoundError as e:
        error_msg = f'File tidak ditemukan: {str(e)}'
        flash(error_msg, 'warning')
        print(f"❌ File Error: {error_msg}")
        
    except KeyError as e:
        error_msg = f'Key tidak ditemukan: {str(e)}'
        flash(error_msg, 'warning')
        print(f"❌ Key Error: {error_msg}")
        
    except Exception as e:
        error_msg = f'Error loading tentang: {str(e)}'
        flash(error_msg, 'warning')
        print(f"❌ Error: {error_msg}")
        import traceback
        traceback.print_exc()
    
    return render_template('tentang/index.html', **tentang_data)

#-------Analisis Route (LOAD EXISTING VISUALIZATIONS + DETAILED PREDICTIONS)---------#
@app.route('/analisis')
@login_required
def analisis():
    """Halaman analisis lengkap hasil penelitian SVR dengan detailed predictions"""
    
    # Default values
    analisis_data = {
        'research_title': 'Analisis Prediksi Harga Bahan Pangan - Kabupaten Langkat',
        'research_location': 'Kabupaten Langkat, Sumatera Utara',
        'algorithm': 'Support Vector Regression (SVR)',
        'kernel': 'RBF (Radial Basis Function)',
        'total_records': 1228,
        'categories_count': 6,
        'categories_list': [],
        'data_period': '2020-2024',
        'avg_mape': 5.66,
        'avg_mae': 779,
        'avg_rmse': 834,
        'best_category': 'Beras Kualitas Medium II',
        'best_category_short': 'Medium II',
        'best_mape': 5.07,
        'interpretation': 'Sangat Baik',
        'total_models': 6,
        'model_parameters': {'C': 100, 'epsilon': 0.1, 'gamma': 'scale'},
        'train_samples': 982,
        'test_samples': 246,
        'train_percentage': 80,
        'test_percentage': 20,
        'time_features': ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Quarter'],
        'weather_features': ['Temperatur', 'Curah_Hujan', 'Temperatur_rolling_7', 'Curah_Hujan_rolling_7', 'Curah_Hujan_rolling_30'],
        'lag_features': ['lag_1 (1 hari sebelumnya)', 'lag_7 (7 hari sebelumnya)', 'lag_30 (30 hari sebelumnya)'],
        'rolling_features': ['rolling_mean_7 (rata-rata 7 hari)', 'rolling_mean_30 (rata-rata 30 hari)'],
        'weather_price_correlation': {},
        'category_performance': {},
        'visualizations': {},
        'detailed_predictions': {},
        'prediction_stats': {}
    }
    
    try:
        base_path = os.path.join(os.getcwd(), 'flask_components')
        
        # Load JSON files
        print("\n" + "="*60)
        print("📂 LOADING ANALISIS DATA")
        print("="*60)
        
        with open(os.path.join(base_path, 'results', 'svr_results.json'), 'r', encoding='utf-8') as f:
            svr_results = json.load(f)
        print("✅ Loaded svr_results.json")
        
        with open(os.path.join(base_path, 'data', 'dataset_info.json'), 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        print("✅ Loaded dataset_info.json")
        
        with open(os.path.join(base_path, 'data', 'preprocessing_info.json'), 'r', encoding='utf-8') as f:
            preprocessing_info = json.load(f)
        print("✅ Loaded preprocessing_info.json")
        
        with open(os.path.join(base_path, 'data', 'splitting_info.json'), 'r', encoding='utf-8') as f:
            splitting_info = json.load(f)
        print("✅ Loaded splitting_info.json")
        
        # Process data
        kategori_beras = dataset_info.get('kategori_beras', [])
        print(f"📊 Categories: {len(kategori_beras)}")
        
        # Calculate metrics
        all_mape = [svr_results[k]['testing_metrics']['mape'] for k in kategori_beras if k in svr_results]
        all_mae = [svr_results[k]['testing_metrics']['mae'] for k in kategori_beras if k in svr_results]
        all_rmse = [svr_results[k]['testing_metrics']['rmse'] for k in kategori_beras if k in svr_results]
        
        avg_mape = sum(all_mape) / len(all_mape) if all_mape else 5.66
        avg_mae = sum(all_mae) / len(all_mae) if all_mae else 509
        avg_rmse = sum(all_rmse) / len(all_rmse) if all_rmse else 537
        
        print(f"📈 Avg MAPE: {avg_mape:.2f}%")
        print(f"📈 Avg MAE: Rp {avg_mae:.0f}")
        print(f"📈 Avg RMSE: Rp {avg_rmse:.0f}")
        
        # Best category
        if kategori_beras:
            best_category = min(kategori_beras, key=lambda k: svr_results[k]['testing_metrics']['mape'])
            best_category_short = best_category.replace('Beras Kualitas ', '')
            best_mape = svr_results[best_category]['testing_metrics']['mape']
            print(f"🏆 Best: {best_category_short} (MAPE: {best_mape:.2f}%)")
        else:
            best_category = 'Beras Kualitas Medium II'
            best_category_short = 'Medium II'
            best_mape = 5.07
        
        # Category performance
        category_performance = {}
        for k in kategori_beras:
            if k in svr_results:
                category_performance[k] = {
                    'mape': svr_results[k]['testing_metrics']['mape'],
                    'mae': svr_results[k]['testing_metrics']['mae'],
                    'rmse': svr_results[k]['testing_metrics']['rmse']
                }
        
        # Get other info
        total_records = splitting_info.get('total_records', 1228)
        if total_records == 0:
            total_records = splitting_info.get('train_size', 0) + splitting_info.get('test_size', 0)
            if total_records == 0:
                total_records = 1228
        
        train_samples = splitting_info.get('train_size', 982)
        test_samples = splitting_info.get('test_size', 246)
        train_percentage = round((train_samples / total_records) * 100) if total_records > 0 else 80
        test_percentage = round((test_samples / total_records) * 100) if total_records > 0 else 20
        
        # Data period
        periode = dataset_info.get('periode', {})
        if periode:
            start_year = periode.get('start', '2020-01-02')[:4]
            end_year = periode.get('end', '2024-12-31')[:4]
            data_period = f"{start_year}-{end_year}"
        else:
            data_period = '2020-2024'
        
        # Interpretation
        if avg_mape <= 10:
            interpretation = "Sangat Baik"
        elif avg_mape <= 20:
            interpretation = "Baik"
        else:
            interpretation = "Cukup"
        
        # Features
        base_features = preprocessing_info.get('base_features', ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Quarter'])
        time_features = base_features
        weather_features = preprocessing_info.get('weather_features', ['Temperatur', 'Curah_Hujan', 'Temperatur_rolling_7', 'Curah_Hujan_rolling_7', 'Curah_Hujan_rolling_30'])
        lag_features = ['lag_1 (1 hari sebelumnya)', 'lag_7 (7 hari sebelumnya)', 'lag_30 (30 hari sebelumnya)']
        rolling_features = ['rolling_mean_7 (rata-rata 7 hari)', 'rolling_mean_30 (rata-rata 30 hari)']
        weather_price_correlation = dataset_info.get('weather_price_correlation', {})
        
        # ===== LOAD DETAILED PREDICTIONS (CELL 8) =====
        print("\n" + "="*60)
        print("📋 LOADING DETAILED PREDICTIONS")
        print("="*60)
        
        detailed_predictions = {}
        prediction_stats = {}
        
        results_path = os.path.join(base_path, 'results')
        
        for kategori in kategori_beras:
            # Create safe filename
            safe_kategori = kategori.replace(' ', '_').replace('/', '_')
            csv_filename = f'display_predictions_{safe_kategori}.csv'
            csv_path = os.path.join(results_path, csv_filename)
            
            if os.path.exists(csv_path):
                try:
                    import pandas as pd
                    df_pred = pd.read_csv(csv_path)
                    
                    # Convert to list of dicts
                    predictions_list = []
                    for idx, row in df_pred.iterrows():
                        # Parse values (remove 'Rp' and formatting)
                        actual_str = str(row.get('Harga Aktual', '0')).replace('Rp ', '').replace(',', '')
                        predicted_str = str(row.get('Harga Prediksi', '0')).replace('Rp ', '').replace(',', '')
                        error_str = str(row.get('Error', '0')).replace('Rp ', '').replace(',', '')
                        abs_error_str = str(row.get('Absolute Error', '0')).replace('Rp ', '').replace(',', '')
                        pct_error_str = str(row.get('Percentage Error (%)', '0')).replace('%', '')
                        ape_str = str(row.get('APE (%)', '0')).replace('%', '')
                        temperatur_str = str(row.get('Temperatur', '')).replace('°C', '')
                        curah_hujan_str = str(row.get('Curah Hujan', '')).replace('mm', '')

                        predictions_list.append({
                            'date': row.get('Tanggal', ''),
                            'actual': float(actual_str) if actual_str else 0,
                            'predicted': float(predicted_str) if predicted_str else 0,
                            'error': float(error_str) if error_str else 0,
                            'absolute_error': float(abs_error_str) if abs_error_str else 0,
                            'percentage_error': float(pct_error_str) if pct_error_str else 0,
                            'ape': float(ape_str) if ape_str else 0,
                            'temperatur': float(temperatur_str) if temperatur_str else None,
                            'curah_hujan': float(curah_hujan_str) if curah_hujan_str else None
                        })
                    
                    detailed_predictions[kategori] = predictions_list
                    
                    # Calculate stats from svr_results
                    if kategori in svr_results:
                        prediction_stats[kategori] = {
                            'mae': svr_results[kategori]['testing_metrics']['mae'],
                            'rmse': svr_results[kategori]['testing_metrics']['rmse'],
                            'mape': svr_results[kategori]['testing_metrics']['mape']
                        }
                    
                    print(f"  ✅ Loaded predictions for {kategori}: {len(predictions_list)} records")
                    
                except Exception as e:
                    print(f"  ⚠️ Error loading predictions for {kategori}: {e}")
                    detailed_predictions[kategori] = []
                    prediction_stats[kategori] = {'mae': 0, 'rmse': 0, 'mape': 0}
            else:
                print(f"  ⚠️ File not found: {csv_filename}")
                detailed_predictions[kategori] = []
                prediction_stats[kategori] = {'mae': 0, 'rmse': 0, 'mape': 0}
        
        print(f"\n✅ Loaded detailed predictions for {len(detailed_predictions)} categories")
        
        # ===== LOAD EXISTING VISUALIZATIONS =====
        print("\n" + "="*60)
        print("📊 LOADING VISUALIZATIONS")
        print("="*60)
        
        viz_source_path = os.path.join(base_path, 'visualizations')
        visualizations = {}
        
        # List semua file HTML di folder visualizations
        if os.path.exists(viz_source_path):
            viz_files = [f for f in os.listdir(viz_source_path) if f.endswith('.html')]
            print(f"Found {len(viz_files)} visualization files")
            
            # Copy ke static/visualizations jika belum ada
            static_viz_path = os.path.join('static', 'visualizations')
            os.makedirs(static_viz_path, exist_ok=True)
            
            for viz_file in viz_files:
                source = os.path.join(viz_source_path, viz_file)
                dest = os.path.join(static_viz_path, viz_file)
                
                # Copy file jika belum ada atau outdated
                if not os.path.exists(dest) or os.path.getmtime(source) > os.path.getmtime(dest):
                    import shutil
                    shutil.copy2(source, dest)
                    print(f"  ✅ Copied: {viz_file}")
            
            # Map visualizations dengan key yang mudah diingat
            viz_mapping = {
                # Performance metrics comparison
                'mape_comparison': 'mape_comparison_all_categories.html',
                'mae_comparison': 'mae_comparison_all_categories.html',
                'rmse_comparison': 'rmse_comparison_all_categories.html',
                
                # Train vs Test comparison
                'mae_train_test': 'mae_comparison_train_test.html',
                'mape_train_test': 'mape_comparison_train_test.html',
                
                # Correlation & heatmap
                'correlation_heatmap': 'korelasi_harga_antar_kategori.html',
                'heatmap_mape': 'heatmap_mape_kategori_tahun.html',
                
                # Time series - Trend harga
                'trend_harga': 'trend_harga_semua_kategori.html',
                'yearly_comparison': 'perbandingan_harga_per_tahun.html',
                'box_distribution': 'distribusi_harga_per_kategori.html',

                # Faktor cuaca (revisi: pengaruh cuaca terhadap harga)
                'weather_trend': 'trend_cuaca_harian.html',
                'weather_correlation': 'korelasi_cuaca_harga.html',

                # Time series comparison all categories
                'time_series_all': 'time_series_all_categories_comparison.html',
                
                # Error metrics over time
                'mape_per_tahun': 'mape_per_tahun_all_categories.html',
                'mae_per_tahun': 'mae_per_tahun_all_categories.html',
            }
            
            # Check mana yang ada
            for key, filename in viz_mapping.items():
                if filename in viz_files:
                    visualizations[key] = filename
                    print(f"  ✓ Mapped: {key} -> {filename}")
            
            # Dynamic mapping untuk per-category visualizations
            for kategori in kategori_beras:
                safe_kategori = kategori.replace(' ', '_').replace('/', '_')
                
                # Time series per category
                ts_patterns = [
                    f'time_series_prediksi_{safe_kategori}.html',
                    f'time_series_{safe_kategori}.html'
                ]
                for pattern in ts_patterns:
                    if pattern in viz_files:
                        visualizations[f'time_series_{safe_kategori}'] = pattern
                        print(f"  ✓ Mapped: time_series_{safe_kategori} -> {pattern}")
                        break
                
                # Scatter plot per category
                scatter_patterns = [
                    f'prediksi_vs_aktual_{safe_kategori}.html',
                    f'scatter_{safe_kategori}.html'
                ]
                for pattern in scatter_patterns:
                    if pattern in viz_files:
                        visualizations[f'scatter_{safe_kategori}'] = pattern
                        print(f"  ✓ Mapped: scatter_{safe_kategori} -> {pattern}")
                        break
                
                # Residual plot per category
                residual_patterns = [
                    f'residual_plot_{kategori}.html',
                    f'residual_plot_{safe_kategori}.html',
                    f'residual_{safe_kategori}.html'
                ]
                for pattern in residual_patterns:
                    if pattern in viz_files:
                        visualizations[f'residual_{safe_kategori}'] = pattern
                        print(f"  ✓ Mapped: residual_{safe_kategori} -> {pattern}")
                        break
            
            print(f"\n✅ Loaded {len(visualizations)} visualizations")
        else:
            print(f"⚠️ Visualizations folder not found: {viz_source_path}")
        
        # Update data
        analisis_data.update({
            'total_records': total_records,
            'categories_count': len(kategori_beras),
            'categories_list': kategori_beras,
            'data_period': data_period,
            'avg_mape': avg_mape,
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse,
            'best_category': best_category,
            'best_category_short': best_category_short,
            'best_mape': best_mape,
            'interpretation': interpretation,
            'total_models': len(kategori_beras),
            'train_samples': train_samples,
            'test_samples': test_samples,
            'train_percentage': train_percentage,
            'test_percentage': test_percentage,
            'time_features': time_features,
            'weather_features': weather_features,
            'lag_features': lag_features,
            'rolling_features': rolling_features,
            'weather_price_correlation': weather_price_correlation,
            'category_performance': category_performance,
            'visualizations': visualizations,
            'detailed_predictions': detailed_predictions,
            'prediction_stats': prediction_stats
        })
        
        print(f"\n✅ Analisis page ready with detailed predictions")
        print(f"   - Categories: {len(kategori_beras)}")
        print(f"   - Visualizations: {len(visualizations)}")
        print(f"   - Detailed predictions: {len(detailed_predictions)}")
        print("="*60 + "\n")
        
    except Exception as e:
        error_msg = f'Error loading analisis: {str(e)}'
        flash(error_msg, 'warning')
        print(f"\n❌ Error: {error_msg}")
        import traceback
        traceback.print_exc()
    
    return render_template('analisis/index.html', **analisis_data)

#-------Prediksi Route (INTERACTIVE PREDICTION WITH FORM)---------#
@app.route('/prediksi', methods=['GET', 'POST'])
@login_required
def prediksi():
    """Halaman prediksi harga beras interaktif dengan form input"""
    
    # Default values
    prediksi_data = {
        'page_title': 'Prediksi Harga Beras - Interaktif',
        'research_location': 'Kabupaten Langkat, Sumatera Utara',
        'categories_count': 6,
        'categories_list': [],
        'selected_days': 30,  # Default 30 hari
        'selected_category': 'all',  # Default semua kategori
        'available_days': 730,  # Total hari tersedia
        'start_date': '',
        'end_date': '',
        'last_historical_date': '',
        'filtered_predictions': {},
        'prediction_stats': {},
        'comparison_data': {},
        'visualizations': {},
        'prediction_available': False,
        'show_results': False
    }
    
    try:
        base_path = os.path.join(os.getcwd(), 'flask_components')
        
        print("\n" + "="*60)
        print("🔮 LOADING PREDIKSI DATA")
        print("="*60)
        
        # ===== LOAD DATASET INFO =====
        dataset_path = os.path.join(base_path, 'data', 'dataset_info.json')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
            kategori_beras = dataset_info.get('kategori_beras', [])
            print(f"✅ Loaded dataset_info.json - {len(kategori_beras)} categories")
        else:
            kategori_beras = [
                'Beras Kualitas Bawah I',
                'Beras Kualitas Bawah II',
                'Beras Kualitas Medium I',
                'Beras Kualitas Medium II',
                'Beras Kualitas Super I',
                'Beras Kualitas Super II'
            ]
            print(f"⚠️ Using default categories")
        
        # ===== LOAD FUTURE PREDICTIONS SUMMARY =====
        summary_path = os.path.join(base_path, 'results', 'future_predictions_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                future_summary = json.load(f)
            
            prediction_info = future_summary.get('prediction_info', {})
            available_days = prediction_info.get('total_days', 730)
            last_historical_date = prediction_info.get('last_historical_date', '')
            
            print(f"✅ Loaded future_predictions_summary.json")
            print(f"   Available days: {available_days}")
            
            prediksi_data['prediction_available'] = True
            prediksi_data['available_days'] = available_days
            prediksi_data['last_historical_date'] = last_historical_date
        else:
            print(f"⚠️ future_predictions_summary.json not found")
            available_days = 730
            last_historical_date = ''
        
        # ===== HANDLE FORM SUBMISSION =====
        if request.method == 'POST':
            print("\n📝 Processing form submission...")
            
            # Get form data
            selected_days = int(request.form.get('prediction_days', 30))
            selected_category = request.form.get('category', 'all')
            
            print(f"   Selected days: {selected_days}")
            print(f"   Selected category: {selected_category}")
            
            # Validate days
            if selected_days < 1:
                selected_days = 1
            elif selected_days > available_days:
                selected_days = available_days
            
            prediksi_data['selected_days'] = selected_days
            prediksi_data['selected_category'] = selected_category
            prediksi_data['show_results'] = True
            
            # ===== LOAD AND FILTER PREDICTIONS =====
            print(f"\n📊 Loading predictions for {selected_days} days...")
            filtered_predictions = {}
            data_path = os.path.join(base_path, 'data')
            
            # Determine which categories to load
            if selected_category == 'all':
                categories_to_load = kategori_beras
            else:
                categories_to_load = [selected_category]
            
            for kategori in categories_to_load:
                safe_kategori = kategori.replace(' ', '_').replace('/', '_')
                csv_filename = f'future_predictions_{safe_kategori}.csv'
                csv_path = os.path.join(data_path, csv_filename)
                
                if os.path.exists(csv_path):
                    try:
                        import pandas as pd
                        df_pred = pd.read_csv(csv_path)
                        
                        # Convert date column
                        df_pred['date'] = pd.to_datetime(df_pred['date'])
                        
                        # FILTER: Ambil N hari pertama sesuai input user
                        df_filtered = df_pred.head(selected_days)
                        
                        # Get date range
                        start_date = df_filtered['date'].min()
                        end_date = df_filtered['date'].max()
                        
                        # Convert to list of dicts
                        predictions_list = []
                        for idx, row in df_filtered.iterrows():
                            predictions_list.append({
                                'date': row['date'].strftime('%Y-%m-%d'),
                                'date_display': row['date'].strftime('%d/%m/%Y'),
                                'predicted_price': float(row['predicted_price']),
                                'year': int(row['year']),
                                'month': int(row['month']),
                                'quarter': int(row['quarter']),
                                'temperatur': float(row['temperatur']) if 'temperatur' in row else None,
                                'curah_hujan': float(row['curah_hujan']) if 'curah_hujan' in row else None
                            })
                        
                        filtered_predictions[kategori] = predictions_list
                        
                        # Update date range in prediksi_data
                        if not prediksi_data['start_date']:
                            prediksi_data['start_date'] = start_date.strftime('%Y-%m-%d')
                            prediksi_data['end_date'] = end_date.strftime('%Y-%m-%d')
                        
                        print(f"  ✅ {kategori}: {len(predictions_list)} predictions loaded")
                        
                    except Exception as e:
                        print(f"  ⚠️ Error loading {kategori}: {e}")
                        filtered_predictions[kategori] = []
                else:
                    print(f"  ⚠️ File not found: {csv_filename}")
                    filtered_predictions[kategori] = []
            
            # ===== CALCULATE STATISTICS FOR FILTERED DATA =====
            print(f"\n📈 Calculating statistics for filtered data...")
            prediction_stats = {}
            
            for kategori, predictions in filtered_predictions.items():
                if predictions:
                    prices = [p['predicted_price'] for p in predictions]
                    
                    prediction_stats[kategori] = {
                        'count': len(prices),
                        'min': min(prices),
                        'max': max(prices),
                        'mean': sum(prices) / len(prices),
                        'first_price': prices[0],
                        'last_price': prices[-1],
                        'change': prices[-1] - prices[0],
                        'change_pct': ((prices[-1] - prices[0]) / prices[0]) * 100 if prices[0] != 0 else 0
                    }
                    
                    print(f"  ✅ {kategori}: mean=Rp{prediction_stats[kategori]['mean']:,.0f}")
            
            prediksi_data['filtered_predictions'] = filtered_predictions
            prediksi_data['prediction_stats'] = prediction_stats
            
            print(f"\n✅ Filtered predictions ready: {len(filtered_predictions)} categories")
        
        # ===== LOAD COMPARISON DATA (for reference) =====
        comparison_path = os.path.join(base_path, 'results', 'historical_vs_future_comparison.csv')
        comparison_data = {}
        
        if os.path.exists(comparison_path):
            try:
                import pandas as pd
                comp_df = pd.read_csv(comparison_path)
                
                for idx, row in comp_df.iterrows():
                    kategori_short = row['Kategori']
                    
                    # Find full category name
                    full_kategori = None
                    for k in kategori_beras:
                        if kategori_short in k:
                            full_kategori = k
                            break
                    
                    if full_kategori:
                        comparison_data[full_kategori] = {
                            'historical_mean': row['Historical Mean'],
                            'future_mean': row['Future Mean'],
                            'change': row['Perubahan'],
                            'change_pct': row['Perubahan (%)'],
                            'trend': row['Trend']
                        }
                
                print(f"✅ Loaded comparison data for reference")
            except Exception as e:
                print(f"⚠️ Error loading comparison data: {e}")
        
        # ===== LOAD VISUALIZATIONS (full 2 years for reference) =====
        print("\n📊 Loading reference visualizations...")
        viz_source_path = os.path.join(base_path, 'visualizations')
        visualizations = {}
        
        if os.path.exists(viz_source_path):
            viz_files = [f for f in os.listdir(viz_source_path) if f.endswith('.html') and 'future' in f.lower()]
            
            # Copy to static
            static_viz_path = os.path.join('static', 'visualizations')
            os.makedirs(static_viz_path, exist_ok=True)
            
            for viz_file in viz_files:
                source = os.path.join(viz_source_path, viz_file)
                dest = os.path.join(static_viz_path, viz_file)
                
                if not os.path.exists(dest) or os.path.getmtime(source) > os.path.getmtime(dest):
                    import shutil
                    shutil.copy2(source, dest)
            
            # Map visualizations
            viz_mapping = {
                'future_all': 'future_predictions_all_categories.html',
                'future_yearly': 'future_predictions_yearly_avg.html',
                'future_distribution': 'future_predictions_distribution.html'
            }
            
            for key, filename in viz_mapping.items():
                if filename in viz_files:
                    visualizations[key] = filename
            
            # Per category predictions
            for kategori in kategori_beras:
                safe_kategori = kategori.replace(' ', '_').replace('/', '_')
                pattern = f'future_prediction_{safe_kategori}.html'
                
                if pattern in viz_files:
                    visualizations[f'future_{safe_kategori}'] = pattern
        
        print(f"✅ Loaded {len(visualizations)} reference visualizations")
        
        # ===== UPDATE PREDIKSI DATA =====
        prediksi_data.update({
            'categories_list': kategori_beras,
            'categories_count': len(kategori_beras),
            'comparison_data': comparison_data,
            'visualizations': visualizations
        })
        
        print(f"\n✅ Prediksi page ready")
        print(f"   - Categories: {len(kategori_beras)}")
        print(f"   - Available days: {available_days}")
        print(f"   - Show results: {prediksi_data['show_results']}")
        print("="*60 + "\n")
        
    except Exception as e:
        error_msg = f'Error loading prediksi: {str(e)}'
        flash(error_msg, 'warning')
        print(f"\n❌ Error: {error_msg}")
        import traceback
        traceback.print_exc()
    
    return render_template('prediksi/index.html', **prediksi_data)

#-------API Routes (untuk future development)---------#
@app.route('/api/categories')
def api_categories():
    """API endpoint untuk mendapatkan daftar kategori beras"""
    try:
        stats = generate_sample_stats()
        return jsonify({
            'success': True,
            'categories': stats['categories']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/price_history/<category>')
def api_price_history(category):
    """API endpoint untuk mendapatkan historical price data"""
    try:
        # Nanti akan load dari file atau database
        # Untuk sekarang return sample data
        return jsonify({
            'success': True,
            'category': category,
            'data': [],
            'message': 'Feature coming soon'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

#-------Error Handlers---------#
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('layouts/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('layouts/500.html'), 500

#-------Context Processors---------#
@app.context_processor
def inject_global_vars():
    """Inject global variables ke semua template"""
    return {
        'app_name': 'Prediksi Harga Bahan Pangan',
        'app_description': 'Sistem Prediksi Harga Bahan Pangan di Kabupaten Langkat menggunakan Support Vector Regression (SVR) dengan kernel RBF',
        'current_year': datetime.now().year,
        'is_logged_in': 'username' in session,
        'current_user': session.get('username', 'Guest'),
        'research_location': 'Kabupaten Langkat, Sumatera Utara',
        'research_algorithm': 'SVR + kernel RBF',
        'research_method': 'Support Vector Regression'
    }

if __name__ == '__main__':
    print("=" * 80)
    print("Starting Rice Price Prediction System...")
    print("=" * 80)
    print("Research: Penerapan Metode Support Vector Regression")
    print("          dalam Memprediksi Harga Bahan Pangan di Kabupaten Langkat")
    print("-" * 80)
    print("Algorithm: Support Vector Regression (SVR) dengan kernel RBF")
    print("Location:  Kabupaten Langkat, Sumatera Utara")
    print("Period:    2020-2024 (5 tahun, 1258 hari)")
    print("Categories: 6 kategori beras")
    print("-" * 80)
    print("Metrics:   MAE, RMSE, MAPE")
    print("Avg MAPE:  5.66% (Sangat Baik)")
    print("=" * 80)
    
    # Check flask_components folder
    flask_components_path = os.path.join(os.getcwd(), 'flask_components')
    if os.path.exists(flask_components_path):
        print("✅ flask_components folder: FOUND")
        
        # Check subfolders
        subfolders = ['models', 'data', 'results', 'visualizations']
        for folder in subfolders:
            folder_path = os.path.join(flask_components_path, folder)
            if os.path.exists(folder_path):
                files_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                print(f"✅ {folder}/: {files_count} files")
            else:
                print(f"❌ {folder}/: NOT FOUND")
    else:
        print("❌ flask_components folder: NOT FOUND")
    
    print("=" * 80)
    print("Flask application starting...")
    print("-" * 80)
    print("Login Credentials:")
    print("  Username: admin")
    print("  Password: admin123")
    print("-" * 80)
    print("URLs:")
    print("  Homepage:  http://localhost:5000")
    print("  Login:     http://localhost:5000/admin/login")
    print("  Dashboard: http://localhost:5000/dashboard")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)