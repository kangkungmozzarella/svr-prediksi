"""Ekspor detail model SVR (gamma hasil hitung 'scale', jumlah support vector) ke JSON untuk ditampilkan di web."""
import json
import pickle

with open('flask_components/models/svr_models.pkl', 'rb') as f:
    models = pickle.load(f)

details = {
    kategori: {
        'gamma_value': float(model._gamma),
        'n_features': int(model.n_features_in_),
        'n_support_vectors': int(len(model.support_)),
    }
    for kategori, model in models.items()
}

with open('flask_components/data/model_details.json', 'w', encoding='utf-8') as f:
    json.dump(details, f, indent=2, ensure_ascii=False)
print(json.dumps(details, indent=2))
