# Credit Risk Decisioning — demo

This repo is a demo of an end-to-end credit risk decisioning service:
- training script that logs models + metrics + artifacts to MLflow,
- FastAPI inference service that loads a selected run (best_model.json / explicit run id / registry),
- Streamlit UI for demo scenarios and monitoring,
- lightweight monitoring (request logs + PSI/KS drift),
- minimal RAG module that answers questions using docs + MLflow artifacts from the loaded run.

The model outputs a probability score:
- `score = P(default = bad)`

Decision rule (fixed in this demo):
- **decline if score >= threshold**
- otherwise approve

Threshold is selected on validation data using a cost function.

---

## Quick Start (Offline/Demo Mode)

**For interview demos - everything runs locally, no external dependencies after initial setup.**

### Ścieżka A: MVP (zero MLflow, 3 komendy)
Jeśli chcesz najprostsze demo bez MLflow i sieci:
```bash
cd mvp && pip install -r requirements.txt && python train.py
uvicorn api:app --host 0.0.0.0 --port 8000   # terminal 1
streamlit run streamlit_app.py               # terminal 2
```
Szczegóły: [mvp/README.md](mvp/README.md).

### Ścieżka B: Pełny projekt (MLflow + drift + RAG)
Wymaga uruchomienia MLflow, treningów i `select_best_model` — szczegóły poniżej.

---

### 1. Initial Setup (Run Once)

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Manual setup:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

This will:
- Create virtual environment
- Install all dependencies
- Download the German Credit dataset
- Create necessary directories

### 2. Prepare for Demo (Run Before Interview)

**Step 1: Start MLflow server (Terminal 1)**
```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5050 \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts
```

**Step 2: Train models (Terminal 2)**
```bash
# Activate venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Train a few models
python -m src.train --experiment bank-credit-risk-http --run-name rf-baseline --model rf
python -m src.train --experiment bank-credit-risk-http --run-name xgb-baseline --model xgb
python -m src.train --experiment bank-credit-risk-http --run-name logreg-baseline --model logreg

# Select best model
python src/scripts/select_best_model.py
```

**Step 3: Verify everything works**
```bash
# Start API (Terminal 3)
uvicorn src.api.app:app --reload --port 8000

# In another terminal, test:
curl http://localhost:8000/health
```

**Step 4: Pre-warm cache (Important for demo!)**
```bash
# Make a few predictions to ensure model is cached
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"duration": 12, "credit_amount": 1000, "age": 35}}'
```

**Step 5: Start Streamlit (Terminal 4)**
```bash
streamlit run streamlit_app.py
```

**Now everything is ready!** The model is cached locally, so the API will start quickly even if MLflow is slow.

---

## Requirements

Python 3.12 recommended.

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run MLflow

In one terminal:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5050 \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts
```

Open MLflow UI:

* [http://localhost:5050](http://localhost:5050)

---

## Train a model (logs metrics + artifacts)

Example:

```bash
python -m src.train --experiment bank-credit-risk-http --run-name rf-baseline --model rf
```

After training you should see a new run in MLflow with artifacts like:

* expected_columns.json
* threshold_cost_table_val.csv
* feature_importance_top.csv (depends on model)
* drift_reference_train.csv (baseline for drift)

---

## Select the best run (writes best_model.json)

```bash
python src/scripts/select_best_model.py
```

This produces `best_model.json` which the API uses by default.

---

## Run the API

In another terminal:

```bash
uvicorn src.api.app:app --reload --port 8000
```

Test:

```bash
curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/schema | jq
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"duration": 12, "credit_amount": 1000, "age": 35}}' | jq
```

Reload model without restarting the server:

```bash
curl -s -X POST http://localhost:8000/reload | jq
```

---

## Run Streamlit UI

In another terminal:

```bash
streamlit run streamlit_app.py
```

Open:

* [http://localhost:8501](http://localhost:8501)

The UI includes:

* demo scenarios (SAFE / RISKY / BORDER)
* request logging
* drift endpoint (PSI/KS) based on logged requests
* RAG tab (answers from docs + MLflow artifacts)

---

## RAG (docs + MLflow artifacts)

RAG indexes:

* `rag_docs/*.md`
* and, if run_id is known, selected MLflow artifacts from that run:

  * threshold tables
  * feature importance exports
  * CV summary
  * schema / example payload
  * drift reference summary

This is intentionally extractive for demos:

* answers are grounded in retrieved chunks,
* citations show the exact source fragment.

Example questions:

* "How does threshold work in this demo?"
* "What does best_model.json control?"
* "What artifacts are logged and why?"
* "Which features are most important in the current run?"

---

## Demo script (suggested flow)

1. Open MLflow UI and show the latest run:

   * metrics: ROC AUC / PR AUC, test_cost_at_val_threshold
   * artifacts: threshold tables, feature importance, drift_reference_train.csv

2. Open Streamlit:

   * run SAFE / RISKY / BORDER scenario and show:

     * score, threshold, decision
     * run_id and classifier name (from /health)

3. Trigger /reload:

   * run select_best_model.py (if you want to switch best_model.json)
   * press Reload in Streamlit
   * verify /health run_id changed

4. Monitoring:

   * make 10–20 predictions to create request logs
   * open "Monitoring" tab and run Drift (PSI/KS)

5. RAG:

   * ask about threshold and show citations
   * ask about feature importance and show it cites MLflow artifacts

---

## Troubleshooting

### API returns 503 "Model is loading/reloading"

**Solution:** Wait 30-60 seconds after starting the API. The model is being downloaded from MLflow and cached locally. After the first load, subsequent starts are much faster.

**For demo:** Pre-warm the cache before the interview by making a prediction request.

### API hangs on startup

**Solution:** 
1. Check MLflow server is running: `curl http://localhost:5050`
2. Check if model exists in MLflow UI
3. Try explicit run_id: `export MODEL_RUN_ID=<your_run_id>`
4. Check cache: `ls -la data/mlflow_cache/runs/`

### Streamlit shows timeout errors

**Solution:**
- API might still be loading - wait and refresh
- Check API health: `curl http://localhost:8000/health`
- Increase timeout in `streamlit_app.py` if needed

### RAG not working

**Solution:**
- RAG initializes lazily on first request - this is normal
- Check `/rag/health` endpoint
- MLflow artifacts are optional - RAG works with just `rag_docs/*.md`

### Offline mode (no MLflow server)

**Solution:**
- Models are cached in `data/mlflow_cache/` after first download
- API can work with cached models even if MLflow server is down
- For true offline, use `MODEL_FALLBACK_PATH` environment variable

---

## Notes

* Drift needs `drift_reference_train.csv` logged in the training run.
  Older runs may not have it; switch run or re-train.
* "Reasons" depend on classifier type.
  LogisticRegression and XGBoost support stronger local explanations.
  For RandomForest/HGB we can show global importances as a simple demo.
* **Model caching:** After first download, models are cached locally in `data/mlflow_cache/`.
  This allows the API to start quickly even if MLflow server is slow or unavailable.
* **RAG lazy loading:** RAG index is built on first request, not during API startup.
  This prevents blocking during initialization.

---

## Before pushing to GitHub

- Copy `.env.example` to `.env` if you use env vars locally; **do not commit `.env`** (it is in `.gitignore`).
- After clone, others run **setup** (e.g. `./setup.sh` or `.\setup.ps1`), then either **MVP** (see path A above) or **full flow** (MLflow + train + select_best_model).
- Dataset: `data/raw/credit-g.csv` is not in the repo; setup downloads it via `src.data.load_credit_g` (OpenML 31). MVP `train.py` can also fetch it if `mvp/data/raw/credit-g.csv` is missing.
- For **demo flow and interview tips**, see [DEMO_AND_INTERVIEW.md](DEMO_AND_INTERVIEW.md).

