# Credit Risk Decisioning — Overview

This repo is a demo of an end-to-end credit risk decisioning service:
- a training script that logs models and evaluation to MLflow,
- an inference API (FastAPI) that loads a selected model (by best_model.json / run id / registry),
- a simple UI (Streamlit) for demo scenarios and monitoring,
- lightweight monitoring signals (request logging + drift metrics),
- a minimal RAG module that answers questions using project docs and artifacts.

The core output of the model is a probability score:
- `score = P(default = "bad")` for a given customer feature vector.
Decisioning applies a single rule:
- **decline if score >= threshold**, otherwise approve.

This demo is intentionally pragmatic:
- it avoids heavy dependencies,
- focuses on reproducible MLflow logging,
- keeps the API stable even when optional pieces are missing (old runs, missing artifacts),
- provides clear demo flows and “what you see is what you logged”.

## Components

### Training (`src/train.py`)
- Loads the German Credit dataset (OpenML `credit-g`).
- Splits data into train/val/test.
- Trains a model pipeline (preprocessing + classifier).
- Selects the best decision threshold on validation data.
- Logs:
  - model artifact (`model/`),
  - threshold metrics,
  - plots,
  - schema artifacts (`expected_columns.json`, example payload),
  - drift reference file (`drift_reference_train.csv`) when enabled.

### API (`src/api/app.py`)
- Loads the model bundle on startup and serves:
  - `/health` for operational info,
  - `/schema` with expected columns and example payload,
  - `/predict` returning score/decision and (when available) reasons,
  - `/reload` to re-load the currently configured “best” model without restarting the process,
  - `/metrics/drift` to compute drift vs the reference file.

### UI (`streamlit_app.py`)
- Preset scenarios (SAFE / BORDER / RISKY).
- Calls the API endpoints and shows results.
- Drift view uses request logs as “current” population.
- RAG tab answers questions using local docs (and later MLflow artifacts).

## What is “RAG” here?
RAG = Retrieval Augmented Generation:
- Retrieve relevant chunks from local documents / artifacts.
- Compose an answer based on retrieved context.
In MVP mode we can be “extractive” (no LLM), or we can generate with an LLM later.
The key point: answers come with citations (source + snippet), so it’s explainable.
