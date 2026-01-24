import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.data import load_credit_g
from xgboost import XGBClassifier, callback as xgb_cb


FIG_DIR = Path("reports/figures")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
mlflow.set_tracking_uri(TRACKING_URI)


# -------------------------
# Config
# -------------------------
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


# -------------------------
# Modeling
# -------------------------
def build_pipeline(model_name: str, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
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
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "xgb":
        # if XGBClassifier is None:
        #     raise RuntimeError("XGBoost not installed. Run: pip install xgboost")

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


# def fit_pipeline(pipe: Pipeline, X_train, y_train, X_val=None, y_val=None, model_name: str = "logreg"):
#     """
#     For xgb: we must early-stop on preprocessed eval_set.
#     For others: normal pipe.fit.
#     """
#     if model_name == "xgb" and X_val is not None:
#         pre = pipe.named_steps["pre"]
#         clf = pipe.named_steps["clf"]
#
#         Xtr = pre.fit_transform(X_train, y_train)
#         Xva = pre.transform(X_val)
#
#         clf.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
#         return pipe
#
#     pipe.fit(X_train, y_train)
#     return pipe
#     # if model_name != "xgb":
#     #     pipe.fit(X_train, y_train)
#     #     return
#     #
#     # pre = pipe.named_steps["pre"]
#     # clf = pipe.named_steps["clf"]
#     #
#     # Xtr = pre.fit_transform(X_train)
#     # if X_val is None or y_val is None:
#     #     # fallback: fit without early stopping
#     #     clf.fit(Xtr, y_train)
#     #     return
#     #
#     # Xv = pre.transform(X_val)
#     # clf.fit(
#     #     Xtr, y_train,
#     #     eval_set=[(Xv, y_val)],
#     #     early_stopping_rounds=50,
#     #     verbose=False,
#     # )

def fit_pipeline(
    pipe: Pipeline,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    model_name: str = "logreg",
    seed: int = 42,
    es_val_fraction: float = 0.1,
):
    """
    Book version:
    - For xgb: early stopping uses an INTERNAL split from X_train (not the outer test fold).
      If X_val is provided (final training), we use it as the early-stopping set.
    - For others: normal pipe.fit on X_train.
    """

    if model_name != "xgb":
        pipe.fit(X_train, y_train)
        return pipe

    # --- XGB
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # If explicit validation was provided (final model training), use it
    if X_val is not None and y_val is not None:
        Xtr = pre.fit_transform(X_train, y_train)
        Xva = pre.transform(X_val)
        clf.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
        return pipe

    # Otherwise (CV): create an internal early-stopping split from X_train
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

# -------------------------
# Explainability / artifacts
# -------------------------
def save_and_log_plot(fig, filename: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(path))


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    save_and_log_plot(fig, "roc_curve.png")


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    save_and_log_plot(fig, "pr_curve.png")


def plot_confmat(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    save_and_log_plot(fig, "confusion_matrix.png")


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig = plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot")
    save_and_log_plot(fig, "calibration.png")


def log_feature_importance(pipe: Pipeline, top_k: int = 30):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(getattr(clf, "n_features_in_", 0))])

    if hasattr(clf, "coef_"):
        importance = clf.coef_.ravel()
        kind = "coef"
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
        kind = "feature_importance"
    else:
        mlflow.set_tag("feature_importance_logged", "false")
        return

    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df["abs_importance"] = df["importance"].abs()
    df = df.sort_values("abs_importance", ascending=False).head(top_k)

    out_path = "feature_importance_top.csv"
    df.to_csv(out_path, index=False)
    mlflow.log_artifact(out_path)
    mlflow.set_tag("feature_importance_logged", "true")
    mlflow.log_param("feature_importance_kind", kind)


def top_reasons_logreg(pipe: Pipeline, x_row: pd.DataFrame, top_k: int = 3):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []

    feat_names = pre.get_feature_names_out()
    Xtr = pre.transform(x_row)

    # one row vector
    v = Xtr[0]
    vec = v.toarray().ravel() if hasattr(v, "toarray") else np.asarray(v).ravel()
    contribs = vec * clf.coef_.ravel()

    idx = np.argsort(np.abs(contribs))[::-1][:top_k]
    return [{"feature": str(feat_names[i]), "contribution": float(contribs[i])} for i in idx]


# -------------------------
# Splits / thresholding
# -------------------------
def make_split(X, y, test_size, val_size, seed, date_col=None):
    if date_col and date_col in X.columns:
        df = X.copy()
        df["_y"] = y
        df = df.sort_values(date_col)

        n = len(df)
        n_test = int(n * test_size)
        test = df.iloc[-n_test:]
        trainval = df.iloc[:-n_test]

        n_val = int(len(trainval) * val_size)
        val = trainval.iloc[-n_val:]
        train = trainval.iloc[:-n_val]

        X_train, y_train = train.drop(columns=["_y"]), train["_y"]
        X_val, y_val = val.drop(columns=["_y"]), val["_y"]
        X_test, y_test = test.drop(columns=["_y"]), test["_y"]
        split_type = "time-based"
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval
        )
        split_type = "stratified"

    return X_train, X_val, X_test, y_train, y_val, y_test, split_type


def threshold_table(y_true, y_prob, fp_cost=1.0, fn_cost=5.0):
    rows = []
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        cost = fp_cost * fp + fn_cost * fn
        approval_rate = float(np.mean(y_pred == 0))  # approve if predicted good (0)
        rows.append({
            "threshold": float(t), "fp": fp, "fn": fn, "tp": tp, "tn": tn,
            "cost": float(cost), "approval_rate": approval_rate
        })
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


# -------------------------
# CV (manual loop -> easy fold logs + works with xgb early stopping)
# -------------------------
def run_cv(
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
        fit_pipeline(pipe, Xtr, ytr, X_val=None, y_val=None, model_name=model_name, seed=seed, es_val_fraction=0.1)

        p_tr = pipe.predict_proba(Xtr)[:, 1]
        p_te = pipe.predict_proba(Xte)[:, 1]

        fold_metrics = {
            "train_roc_auc": float(roc_auc_score(ytr, p_tr)),
            "train_pr_auc": float(average_precision_score(ytr, p_tr)),
            "test_roc_auc": float(roc_auc_score(yte, p_te)),
            "test_pr_auc": float(average_precision_score(yte, p_te)),
        }

        # --- nested run per fold (ładnie wygląda w MLflow + łatwo debugować)
        with mlflow.start_run(run_name=f"cv-fold-{fold}", nested=True):
            mlflow.log_param("fold", fold)
            mlflow.log_param("cv_folds", folds)
            if model_name == "xgb":
                mlflow.log_param("xgb_es_val_fraction", 0.1)
            mlflow.log_metrics(fold_metrics)

            # opcjonalnie: log best_iteration dla xgb w każdym foldzie
            if model_name == "xgb":
                clf = pipe.named_steps["clf"]
                best_iter = getattr(clf, "best_iteration", None)
                if best_iter is not None:
                    mlflow.log_metric("xgb_best_iteration", float(best_iter))

        rows.append({"fold": fold, **fold_metrics})

    return pd.DataFrame(rows)


# -------------------------
# Main
# -------------------------
def main():
    # avoid "already active" if previous run crashed
    if mlflow.active_run() is not None:
        mlflow.end_run(status="KILLED")

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="bank-credit-risk")
    parser.add_argument("--run-name", default="day1-baseline")
    parser.add_argument("--model", choices=["logreg", "hgb", "rf", "xgb"], default="logreg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--fp-cost", type=float, default=1.0)
    parser.add_argument("--fn-cost", type=float, default=5.0)
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
    )

    X, y = load_credit_g()

    X_train, X_val, X_test, y_train, y_val, y_test, split_type = make_split(
        X, y, cfg.test_size, cfg.val_size, cfg.seed, date_col=cfg.date_col
    )

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    mlflow.set_experiment(cfg.experiment)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        # --- params / dataset meta
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("split_type", split_type)
        mlflow.log_param("n_rows", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_numeric", len(numeric_cols))
        mlflow.log_param("n_categorical", len(categorical_cols))
        mlflow.set_tag("positive_class", "1=default/bad")
        mlflow.set_tag("decision_rule", "decline if score >= threshold")
        mlflow.set_tag("threshold_selected_on", "validation")
        print("RUN_ID =", run.info.run_id)
        mlflow.set_tag("run_id", run.info.run_id)
        if cfg.model == "xgb":
            mlflow.log_param("xgb_es_val_fraction", 0.1)

        # --- CV (optional) + fold logs
        if cfg.cv_folds and cfg.cv_folds >= 2:
            cv_df = run_cv(X_train, y_train, numeric_cols, categorical_cols, cfg.model, cfg.cv_folds, cfg.seed)
            cv_df.to_csv("cv_results.csv", index=False)
            mlflow.log_artifact("cv_results.csv")

            # agregaty
            mlflow.log_metric("cv_test_roc_auc_mean", float(cv_df["test_roc_auc"].mean()))
            mlflow.log_metric("cv_test_roc_auc_std", float(cv_df["test_roc_auc"].std(ddof=0)))
            mlflow.log_metric("cv_test_pr_auc_mean", float(cv_df["test_pr_auc"].mean()))
            mlflow.log_metric("cv_test_pr_auc_std", float(cv_df["test_pr_auc"].std(ddof=0)))

            mlflow.log_metric("cv_train_roc_auc_mean", float(cv_df["train_roc_auc"].mean()))
            mlflow.log_metric("cv_train_pr_auc_mean", float(cv_df["train_pr_auc"].mean()))

            gap = float(cv_df["train_roc_auc"].mean() - cv_df["test_roc_auc"].mean())
            mlflow.log_metric("cv_train_test_roc_gap", gap)
            if gap > 0.08:
                mlflow.set_tag("overfitting_warning", f"High ROC gap: {gap:.3f}")

        # --- Train final model (use VAL for xgb early stopping)
        pipe = build_pipeline(cfg.model, numeric_cols, categorical_cols)
        fit_pipeline(pipe, X_train, y_train, X_val, y_val, model_name=cfg.model)

        if cfg.model == "xgb":
            clf = pipe.named_steps["clf"]
            best_iter = getattr(clf, "best_iteration", None)
            if best_iter is not None:
                mlflow.log_metric("xgb_best_iteration", float(best_iter))

        # --- Predict probs
        y_val_prob = pipe.predict_proba(X_val)[:, 1]
        y_test_prob = pipe.predict_proba(X_test)[:, 1]

        # --- Choose threshold on VAL (NO leakage)
        val_tbl = threshold_table(y_val.to_numpy(), y_val_prob, fp_cost=cfg.fp_cost, fn_cost=cfg.fn_cost)
        val_tbl.to_csv("threshold_cost_table_val.csv", index=False)
        mlflow.log_artifact("threshold_cost_table_val.csv")

        best = find_best_threshold_from_table(val_tbl)
        mlflow.log_metric("best_threshold_val", best["threshold"])
        mlflow.log_metric("val_cost_at_best_threshold", best["cost"])
        mlflow.log_metric("val_fp_at_best_threshold", best["fp"])
        mlflow.log_metric("val_fn_at_best_threshold", best["fn"])
        mlflow.log_metric("val_approval_rate_at_best_threshold", best["approval_rate"])

        # --- Evaluate on TEST using val-chosen threshold
        thr = best["threshold"]
        y_test_pred = (y_test_prob >= thr).astype(int)
        test_tbl = threshold_table(y_test.to_numpy(), y_test_prob, fp_cost=cfg.fp_cost, fn_cost=cfg.fn_cost)
        test_tbl.to_csv("threshold_cost_table_test.csv", index=False)
        mlflow.log_artifact("threshold_cost_table_test.csv")

        fp = int(np.sum((y_test_pred == 1) & (y_test.to_numpy() == 0)))
        fn = int(np.sum((y_test_pred == 0) & (y_test.to_numpy() == 1)))
        test_cost = cfg.fp_cost * fp + cfg.fn_cost * fn
        mlflow.log_metric("test_cost_at_val_threshold", float(test_cost))
        mlflow.log_metric("test_fp_at_val_threshold", float(fp))
        mlflow.log_metric("test_fn_at_val_threshold", float(fn))
        mlflow.log_metric("threshold_used_for_test_pred", float(thr))

        # --- Core metrics
        mlflow.log_metric("val_roc_auc", float(roc_auc_score(y_val, y_val_prob)))
        mlflow.log_metric("val_pr_auc", float(average_precision_score(y_val, y_val_prob)))
        mlflow.log_metric("test_roc_auc", float(roc_auc_score(y_test, y_test_prob)))
        mlflow.log_metric("test_pr_auc", float(average_precision_score(y_test, y_test_prob)))

        # --- Plots
        plot_roc(y_test.to_numpy(), y_test_prob)
        plot_pr(y_test.to_numpy(), y_test_prob)
        plot_confmat(y_test.to_numpy(), y_test_pred)
        plot_calibration(y_test.to_numpy(), y_test_prob)

        # --- Explainability
        log_feature_importance(pipe, top_k=30)

        # Permutation importance on RAW columns (slow-ish but OK for 1k rows)
        perm = permutation_importance(
            pipe, X_val, y_val, n_repeats=10, random_state=cfg.seed, n_jobs=-1, scoring="roc_auc"
        )
        imp = pd.DataFrame({
            "feature": X_val.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False).head(30)
        imp.to_csv("perm_importance_top30.csv", index=False)
        mlflow.log_artifact("perm_importance_top30.csv")

        # Local reasons (logreg)
        if cfg.model == "logreg":
            sample = X_test.iloc[[0]]
            reasons = top_reasons_logreg(pipe, sample, top_k=3)
            pd.DataFrame(reasons).to_csv("sample_reasons.csv", index=False)
            mlflow.log_artifact("sample_reasons.csv")

        # --- Log model (pipeline)
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=None,
        )

        print("Done.")
        print(f"val_roc_auc={roc_auc_score(y_val, y_val_prob):.4f}, val_pr_auc={average_precision_score(y_val, y_val_prob):.4f}")
        print(f"test_roc_auc={roc_auc_score(y_test, y_test_prob):.4f}, test_pr_auc={average_precision_score(y_test, y_test_prob):.4f}")
        print(f"best_threshold(val)={thr:.2f}, test_cost_at_val_threshold={test_cost:.1f}, fp={fp}, fn={fn}")


if __name__ == "__main__":
    main()
