# Credit Risk MVP — prosty projekt do rozmowy rekrutacyjnej

Minimalny, działający out-of-the-box przykład ML w bankowości (credit scoring):

- **Bez MLflow** — model zapisany jako `joblib` na dysku
- **Bez zewnętrznych serwisów** — wszystko lokalne
- **3 kroki:** train → API → Streamlit

Dobre praktyki: pipeline (preprocessing + model), train/val split, threshold z validation, schema i expected_columns przy serwisie.

---

## Szybki start

```bash
cd mvp
pip install -r requirements.txt

# 1) Wytrenuj model (zapis do mvp/artifacts/)
python train.py

# 2) Uruchom API (port 8000)
uvicorn api:app --host 0.0.0.0 --port 8000

# 3) W drugim terminalu — UI
streamlit run streamlit_app.py
```

Otwórz przeglądarkę: `http://localhost:8501`. Podaj `http://localhost:8000` jako API_URL i użyj przycisków Health / Schema / Predict.

---

## Struktura

```
mvp/
  train.py         # Uczenie + zapis model.pkl, expected_columns, threshold
  api.py           # FastAPI: /health, /schema, /predict
  streamlit_app.py # Minimalny UI do testów
  artifacts/       # Tworzone przez train.py — model i metadane
  requirements.txt
  README.md
```

---

## Endpointy API

| Endpoint    | Opis |
|------------|------|
| `GET /health` | Status API, nazwa modelu, threshold |
| `GET /schema` | Lista kolumn + przykładowy payload |
| `POST /predict` | Body: `{"features": {...}}` → score, decision, threshold |

---

## Dla rozmowy rekrutacyjnej (VELO / Santander)

- **Dane:** German Credit (OpenML 31) — standard w credit scoringu.
- **Pipeline:** `ColumnTransformer` (OneHotEncoder dla kategorii, StandardScaler opcjonalnie) + `RandomForestClassifier`.
- **Decyzja:** `decline` gdy `score >= threshold`; threshold wybrany na zbiorze walidacyjnym (np. min koszt FP/FN).
- **Serwis:** FastAPI, ładowanie modelu przy starcie, schema i expected_columns — gotowe do opowiedzenia o wersjonowaniu i SLA.
