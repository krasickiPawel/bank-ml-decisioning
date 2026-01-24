import os
import json
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
mlflow.set_tracking_uri(TRACKING_URI)

EXPERIMENT = "bank-credit-risk-http"
OUT = "best_model.json"

# progi “bankowe” – ustaw pod siebie
MAX_GAP = 0.08
MIN_APPROVAL = 0.20
MAX_APPROVAL = 0.90

SORT_KEYS = [
    ("metrics.test_cost_at_val_threshold", True),   # True = rosnąco
    ("metrics.cv_test_pr_auc_mean", False),         # malejąco
    ("metrics.cv_test_roc_auc_mean", False),
]

def main():
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {EXPERIMENT}")

    # Pobierz sporo runów (możesz zwiększyć max_results)
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=200,
        order_by=["attributes.start_time DESC"],
    )

    if runs.empty:
        raise RuntimeError("No runs found")

    # Ujednolicenie braków
    for col in [
        "metrics.cv_train_test_roc_gap",
        "metrics.val_approval_rate_at_best_threshold",
        "metrics.test_cost_at_val_threshold",
    ]:
        if col not in runs.columns:
            runs[col] = pd.NA

    # Filtry jakości
    filtered = runs.copy()

    # 1) musi mieć koszt (inaczej nie porównasz)
    filtered = filtered[filtered["metrics.test_cost_at_val_threshold"].notna()]

    # 2) overfitting: jeśli jest gap – filtruj
    if "metrics.cv_train_test_roc_gap" in filtered.columns:
        filtered = filtered[
            filtered["metrics.cv_train_test_roc_gap"].isna()
            | (filtered["metrics.cv_train_test_roc_gap"] <= MAX_GAP)
        ]

    # 3) approval rate constraint (jeśli jest)
    if "metrics.val_approval_rate_at_best_threshold" in filtered.columns:
        ar = filtered["metrics.val_approval_rate_at_best_threshold"]
        filtered = filtered[ar.isna() | ((ar >= MIN_APPROVAL) & (ar <= MAX_APPROVAL))]

    if filtered.empty:
        raise RuntimeError("All runs filtered out. Relax constraints or check logging.")

    # Sortowanie wg reguły
    sort_cols = []
    asc = []
    for key, ascending in SORT_KEYS:
        if key in filtered.columns:
            sort_cols.append(key)
            asc.append(ascending)

    filtered = filtered.sort_values(by=sort_cols, ascending=asc)

    best = filtered.iloc[0]
    best_run_id = best["run_id"]

    payload = {
        "experiment": EXPERIMENT,
        "best_run_id": best_run_id,
        "selection_rules": {
            "max_gap": MAX_GAP,
            "min_approval": MIN_APPROVAL,
            "max_approval": MAX_APPROVAL,
            "sort_keys": SORT_KEYS,
        },
        "best_metrics": {
            k.replace("metrics.", ""): float(best[k])
            for k in filtered.columns
            if k.startswith("metrics.") and pd.notna(best[k])
        },
    }

    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"BEST RUN: {best_run_id}")
    print(f"Wrote: {OUT}")

if __name__ == "__main__":
    main()
