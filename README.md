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

## Requirements

Python 3.12 recommended.

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

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

## Notes

* Drift needs `drift_reference_train.csv` logged in the training run.
  Older runs may not have it; switch run or re-train.
* "Reasons" depend on classifier type.
  LogisticRegression and XGBoost support stronger local explanations.
  For RandomForest/HGB we can show global importances as a simple demo.

