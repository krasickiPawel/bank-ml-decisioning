# src/scripts/select_best_model.py
from __future__ import annotations

import json
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "bank-credit-risk-http")
OUT_PATH = os.getenv("BEST_MODEL_JSON", "best_model.json")

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


def main() -> None:
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {EXPERIMENT_NAME}")

    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        output_format="pandas",
    )

    if df.empty:
        raise RuntimeError("No FINISHED runs found")

    required = [
        "metrics.test_cost_at_val_threshold",
        "metrics.test_roc_auc",
        "metrics.best_threshold_val",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.dropna(subset=["metrics.test_cost_at_val_threshold", "metrics.test_roc_auc"])
    if df.empty:
        raise RuntimeError("No runs with required metrics. Check train.py logging.")

    df = df.sort_values(
        by=["metrics.test_cost_at_val_threshold", "metrics.test_roc_auc"],
        ascending=[True, False],
    )

    best = df.iloc[0]
    run_id = str(best["run_id"])

    payload = {
        "run_id": run_id,
        "experiment": EXPERIMENT_NAME,
        "criteria": {
            "primary": "min(test_cost_at_val_threshold)",
            "secondary": "max(test_roc_auc)",
        },
        "metrics": {
            "test_cost_at_val_threshold": float(best["metrics.test_cost_at_val_threshold"]),
            "test_roc_auc": float(best["metrics.test_roc_auc"]),
            "best_threshold_val": float(best.get("metrics.best_threshold_val") or 0.5),
        },
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {OUT_PATH} -> run_id={run_id}")


if __name__ == "__main__":
    main()
