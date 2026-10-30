"""Perbarui data penelitian dari sumber terbuka.

Penggunaan:
    python update_data.py cuaca                      # tambah data cuaca terbaru dari Open-Meteo ke cuaca_harian.xlsx
    python update_data.py harga-inspect              # simpan contoh respons PIHPS ke pihps_raw.json untuk diperiksa
"""
import argparse
import json
import shutil
import urllib.parse
import urllib.request
from datetime import date, timedelta

import pandas as pd

CUACA_FILE = 'cuaca_harian.xlsx'
OPEN_METEO_URL = 'https://archive-api.open-meteo.com/v1/archive'
PIHPS_URL = 'https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataKomoditas'

# Titik pusat Stabat (ibu kota Kabupaten Langkat); ubah lewat --lat / --lon bila memakai titik lain
DEFAULT_LAT, DEFAULT_LON = 3.76, 98.45
# Data arsip Open-Meteo tertunda beberapa hari
ARCHIVE_LAG_DAYS = 5


def http_get(url, params):
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode('utf-8')


def fetch_cuaca(start, end, lat, lon):
    raw = http_get(OPEN_METEO_URL, {
        'latitude': lat,
        'longitude': lon,
        'start_date': start.isoformat(),
        'end_date': end.isoformat(),
        'daily': 'temperature_2m_mean,precipitation_sum',
        'timezone': 'Asia/Jakarta',
    })
    daily = json.loads(raw)['daily']
    return pd.DataFrame({
        'Tanggal': pd.to_datetime(daily['time']),
        'Temperatur': daily['temperature_2m_mean'],
        'Curah_Hujan': daily['precipitation_sum'],
    })


def update_cuaca(lat, lon):
    cuaca = pd.read_excel(CUACA_FILE)
    cuaca['Tanggal'] = pd.to_datetime(cuaca['Tanggal'])
    start = cuaca['Tanggal'].max().date() + timedelta(days=1)
    end = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    if start > end:
        print(f"Data cuaca sudah terbaru (sampai {cuaca['Tanggal'].max().date()}).")
        return
    print(f"Mengambil cuaca {start} s/d {end} (lat={lat}, lon={lon})...")
    baru = fetch_cuaca(start, end, lat, lon).dropna()
    shutil.copy(CUACA_FILE, CUACA_FILE.replace('.xlsx', '.backup.xlsx'))
    gabung = pd.concat([cuaca, baru]).drop_duplicates('Tanggal').sort_values('Tanggal')
    gabung.to_excel(CUACA_FILE, index=False)
    print(f"Ditambahkan {len(baru)} hari. Total {len(gabung)} hari, sampai {gabung['Tanggal'].max().date()}.")
    print(f"Cadangan file lama: {CUACA_FILE.replace('.xlsx', '.backup.xlsx')}")


def inspect_harga(args):
    """Simpan respons mentah PIHPS agar strukturnya bisa diperiksa sebelum parser ditulis."""
    raw = http_get(PIHPS_URL, {
        'price_type_id': args.price_type_id,
        'comcat_id': args.comcat_id,
        'province_id': args.province_id,
        'regency_id': args.regency_id,
        'market_id': '',
        'tipe_laporan': args.tipe_laporan,
        'start_date': args.start,
        'end_date': args.end,
    })
    with open('pihps_raw.json', 'w', encoding='utf-8') as f:
        f.write(raw)
    print(f"Respons disimpan ke pihps_raw.json ({len(raw)} karakter). Awal respons:")
    print(raw[:1500])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='perintah', required=True)

    cuaca = sub.add_parser('cuaca')
    cuaca.add_argument('--lat', type=float, default=DEFAULT_LAT)
    cuaca.add_argument('--lon', type=float, default=DEFAULT_LON)

    harga = sub.add_parser('harga-inspect')
    harga.add_argument('--start', default=(date.today() - timedelta(days=7)).strftime('%Y-%m-%d'))
    harga.add_argument('--end', default=date.today().strftime('%Y-%m-%d'))
    harga.add_argument('--price-type-id', default='1')
    harga.add_argument('--comcat-id', default='cat_1')
    harga.add_argument('--province-id', default='0')
    harga.add_argument('--regency-id', default='')
    harga.add_argument('--tipe-laporan', default='1')

    args = parser.parse_args()
    if args.perintah == 'cuaca':
        update_cuaca(args.lat, args.lon)
    else:
        inspect_harga(args)


if __name__ == '__main__':
    main()
