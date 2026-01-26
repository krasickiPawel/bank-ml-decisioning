# src/train.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, callback as xgb_cb

from src.data import load_credit_g

FIG_DIR = Path("reports/figures")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
mlflow.set_tracking_uri(TRACKING_URI)


@dataclass
class TrainConfig:
    experiment: str
    run_name: str
    model: str
    seed: int
    test_size: float
    val_size: float
    cv_folds: int
    date_col: Optional[str]
    fp_cost: float
    fn_cost: float
    register_name: Optional[str]


def build_pipeline(model_name: str, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_name == "hgb":
        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        )
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "xgb":
        es = xgb_cb.EarlyStopping(rounds=50, save_best=True)
        clf = XGBClassifier(
            n_estimators=5000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=5,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="auc",
            n_jobs=-1,
            random_state=42,
            callbacks=[es],
        )
    else:
        raise ValueError("Unknown model. Use: logreg | hgb | rf | xgb")

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def fit_pipeline(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    model_name: str,
    seed: int,
    es_val_fraction: float = 0.1,
) -> Pipeline:
    if model_name != "xgb":
        pipe.fit(X_train, y_train)
        return pipe

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    if X_val is not None and y_val is not None:
        Xtr = pre.fit_transform(X_train, y_train)
        Xva = pre.transform(X_val)
        clf.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
        return pipe

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=es_val_fraction, random_state=seed)
    tr_idx, es_idx = next(splitter.split(X_train, y_train))

    X_tr = X_train.iloc[tr_idx]
    y_tr = y_train.iloc[tr_idx]
    X_es = X_train.iloc[es_idx]
    y_es = y_train.iloc[es_idx]

    Xtr = pre.fit_transform(X_tr, y_tr)
    Xes = pre.transform(X_es)

    clf.fit(Xtr, y_tr, eval_set=[(Xes, y_es)], verbose=False)
    return pipe


def make_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    seed: int,
    date_col: Optional[str] = None,
):
    # For this dataset: we keep only stratified split.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval
    )
    split_type = "stratified"
    return X_train, X_val, X_test, y_train, y_val, y_test, split_type


def threshold_table(y_true: np.ndarray, y_prob: np.ndarray, fp_cost: float = 1.0, fn_cost: float = 5.0) -> pd.DataFrame:
    rows = []
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        cost = fp_cost * fp + fn_cost * fn
        approval_rate = float(np.mean(y_pred == 0))  # approve if predicted good (0)
        rows.append(
            {
                "threshold": float(t),
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "tn": tn,
                "cost": float(cost),
                "approval_rate": approval_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("cost")


def find_best_threshold_from_table(tbl: pd.DataFrame) -> Dict[str, Any]:
    best = tbl.iloc[0].to_dict()
    return {
        "threshold": float(best["threshold"]),
        "fp": int(best["fp"]),
        "fn": int(best["fn"]),
        "tp": int(best["tp"]),
        "tn": int(best["tn"]),
        "cost": float(best["cost"]),
        "approval_rate": float(best["approval_rate"]),
    }


def _save_and_log_plot(fig, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(path))


def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    _save_and_log_plot(fig, "roc_curve.png")


def _plot_pr(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    _save_and_log_plot(fig, "pr_curve.png")


def _plot_confmat(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    _save_and_log_plot(fig, "confusion_matrix.png")


def _plot_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig = plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot")
    _save_and_log_plot(fig, "calibration.png")


def _run_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_name: str,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_train, y_train)):
        Xtr = X_train.iloc[tr_idx]
        ytr = y_train.iloc[tr_idx]
        Xte = X_train.iloc[te_idx]
        yte = y_train.iloc[te_idx]

        pipe = build_pipeline(model_name, numeric_cols, categorical_cols)
        pipe = fit_pipeline(pipe, Xtr, ytr, None, None, model_name=model_name, seed=seed)

        p_tr = pipe.predict_proba(Xtr)[:, 1]
        p_te = pipe.predict_proba(Xte)[:, 1]

        fold_metrics = {
            "train_roc_auc": float(roc_auc_score(ytr, p_tr)),
            "train_pr_auc": float(average_precision_score(ytr, p_tr)),
            "test_roc_auc": float(roc_auc_score(yte, p_te)),
            "test_pr_auc": float(average_precision_score(yte, p_te)),
        }

        with mlflow.start_run(run_name=f"cv-fold-{fold}", nested=True):
            mlflow.log_param("fold", fold)
            mlflow.log_param("cv_folds", folds)
            if model_name == "xgb":
                mlflow.log_param("xgb_es_val_fraction", 0.1)
            mlflow.log_metrics(fold_metrics)

        rows.append({"fold": fold, **fold_metrics})

    return pd.DataFrame(rows)


def main() -> None:
    if mlflow.active_run() is not None:
        mlflow.end_run(status="KILLED")

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="bank-credit-risk-http")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--model", choices=["logreg", "hgb", "rf", "xgb"], default="logreg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--fp-cost", type=float, default=1.0)
    parser.add_argument("--fn-cost", type=float, default=5.0)
    parser.add_argument("--register-name", default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        experiment=args.experiment,
        run_name=args.run_name,
        model=args.model,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        cv_folds=args.cv_folds,
        date_col=args.date_col,
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
        register_name=args.register_name,
    )

    X, y = load_credit_g()

    X_train, X_val, X_test, y_train, y_val, y_test, split_type = make_split(
        X, y, cfg.test_size, cfg.val_size, cfg.seed, date_col=cfg.date_col
    )

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    expected_columns = list(X.columns)

    example_features = {c: None for c in expected_columns}
    example_features.update({"duration": 12, "credit_amount": 1000, "age": 35, "checking_status": "<0"})

    mlflow.set_experiment(cfg.experiment)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("split_type", split_type)
        mlflow.log_param("n_rows", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_numeric", len(numeric_cols))
        mlflow.log_param("n_categorical", len(categorical_cols))

        mlflow.set_tag("positive_class", "1=default/bad")
        mlflow.set_tag("decision_rule", "decline if score >= threshold")
        mlflow.set_tag("threshold_selected_on", "validation")

        expected_columns = list(X.columns)

        with open("expected_columns.json", "w", encoding="utf-8") as f:
            json.dump({"expected_columns": expected_columns}, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact("expected_columns.json")

        ref_path = "drift_reference_train.csv"
        X_train[expected_columns].to_csv(ref_path, index=False)
        mlflow.log_artifact(ref_path)

        # --- Schema + baseline for drift (single source of truth)
        schema_payload = {
            "expected_columns": expected_columns,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "example_features": example_features,
        }
        schema_path = "schema.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_payload, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(schema_path)

        drift_ref_path = "drift_reference_train.csv"
        X_train[expected_columns].to_csv(drift_ref_path, index=False)
        mlflow.log_artifact(drift_ref_path)

        # --- CV (optional)
        if cfg.cv_folds and cfg.cv_folds >= 2:
            cv_df = _run_cv(X_train, y_train, numeric_cols, categorical_cols, cfg.model, cfg.cv_folds, cfg.seed)
            cv_df.to_csv("cv_results.csv", index=False)
            mlflow.log_artifact("cv_results.csv")

            mlflow.log_metric("cv_test_roc_auc_mean", float(cv_df["test_roc_auc"].mean()))
            mlflow.log_metric("cv_test_pr_auc_mean", float(cv_df["test_pr_auc"].mean()))
            gap = float(cv_df["train_roc_auc"].mean() - cv_df["test_roc_auc"].mean())
            mlflow.log_metric("cv_train_test_roc_gap", gap)

        # --- Train final model
        pipe = build_pipeline(cfg.model, numeric_cols, categorical_cols)
        pipe = fit_pipeline(pipe, X_train, y_train, X_val, y_val, model_name=cfg.model, seed=cfg.seed)

        # --- Predict probabilities
        y_val_prob = pipe.predict_proba(X_val)[:, 1]
        y_test_prob = pipe.predict_proba(X_test)[:, 1]

        # --- Choose threshold on VAL
        val_tbl = threshold_table(y_val.to_numpy(), y_val_prob, fp_cost=cfg.fp_cost, fn_cost=cfg.fn_cost)
        val_tbl.to_csv("threshold_cost_table_val.csv", index=False)
        mlflow.log_artifact("threshold_cost_table_val.csv")

        best = find_best_threshold_from_table(val_tbl)
        thr = best["threshold"]

        mlflow.log_metric("best_threshold_val", thr)
        mlflow.log_metric("val_cost_at_best_threshold", best["cost"])
        mlflow.log_metric("val_fp_at_best_threshold", best["fp"])
        mlflow.log_metric("val_fn_at_best_threshold", best["fn"])
        mlflow.log_metric("val_approval_rate_at_best_threshold", best["approval_rate"])

        # --- Evaluate on TEST using val-threshold
        y_test_pred = (y_test_prob >= thr).astype(int)
        fp = int(np.sum((y_test_pred == 1) & (y_test.to_numpy() == 0)))
        fn = int(np.sum((y_test_pred == 0) & (y_test.to_numpy() == 1)))
        test_cost = cfg.fp_cost * fp + cfg.fn_cost * fn

        mlflow.log_metric("threshold_used_for_test_pred", thr)
        mlflow.log_metric("test_cost_at_val_threshold", float(test_cost))
        mlflow.log_metric("test_fp_at_val_threshold", float(fp))
        mlflow.log_metric("test_fn_at_val_threshold", float(fn))

        mlflow.log_metric("val_roc_auc", float(roc_auc_score(y_val, y_val_prob)))
        mlflow.log_metric("val_pr_auc", float(average_precision_score(y_val, y_val_prob)))
        mlflow.log_metric("test_roc_auc", float(roc_auc_score(y_test, y_test_prob)))
        mlflow.log_metric("test_pr_auc", float(average_precision_score(y_test, y_test_prob)))

        # --- Plots (nice for demo)
        _plot_roc(y_test.to_numpy(), y_test_prob)
        _plot_pr(y_test.to_numpy(), y_test_prob)
        _plot_confmat(y_test.to_numpy(), y_test_pred)
        _plot_calibration(y_test.to_numpy(), y_test_prob)

        # --- Log model
        mlflow.sklearn.log_model(sk_model=pipe, artifact_path="model")

        # --- Optional: register model version (NO auto Production)
        if cfg.register_name:
            client = MlflowClient()
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri=model_uri, name=cfg.register_name)

            mlflow.log_param("registered_model_name", cfg.register_name)
            mlflow.log_param("registered_model_version", mv.version)

            # Helpful for demo: set a non-production alias (safe default)
            # You can manually set alias "Production" / stage in UI later.
            try:
                client.set_registered_model_alias(cfg.register_name, "candidate", mv.version)
                mlflow.set_tag("registered_alias", "candidate")
            except Exception:
                pass

        print(f"RUN_ID={run.info.run_id}")
        print(
            f"test_roc_auc={roc_auc_score(y_test, y_test_prob):.4f}, "
            f"test_pr_auc={average_precision_score(y_test, y_test_prob):.4f}, "
            f"best_threshold(val)={thr:.2f}, test_cost_at_val_threshold={test_cost:.1f}"
        )


if __name__ == "__main__":
    main()
