import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import os

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.inspection import permutation_importance

from src.data import load_credit_g


FIG_DIR = Path("reports/figures")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
mlflow.set_tracking_uri(tracking_uri)


@dataclass
class TrainConfig:
    experiment: str
    run_name: str
    model: str
    seed: int
    test_size: float
    val_size: float


@dataclass
class CostConfig:
    fp_cost: float = 1.0   # reject good client cost
    fn_cost: float = 5.0   # accept bad client cost (default)


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
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None,
        )
    elif model_name == "hgb":
        # Fast and strong baseline, no extra deps
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
    else:
        raise ValueError("Unknown model. Use: logreg | hgb | rf")

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", clf),
    ])
    return pipe


def log_feature_importance(pipe: Pipeline, top_k: int = 30):
    """
    Logs top feature importances / coefficients with *post-transform* feature names
    (after OneHotEncoder), so it's interpretable.

    Supports:
    - LogisticRegression: coef_
    - RandomForestClassifier: feature_importances_
    - HistGradientBoostingClassifier: feature_importances_ (if available)
    """
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # feature names after preprocessing (num + one-hot)
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        # fallback (should be rare)
        feature_names = np.array([f"f{i}" for i in range(clf.n_features_in_)])

    if hasattr(clf, "coef_"):
        importance = clf.coef_.ravel()
        kind = "coef"
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
        kind = "feature_importance"
    else:
        # nothing to log for this model
        mlflow.log_param("feature_importance_logged", False)
        return

    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df["abs_importance"] = df["importance"].abs()
    df = df.sort_values("abs_importance", ascending=False).head(top_k)

    out_path = "feature_importance_top.csv"
    df.to_csv(out_path, index=False)
    mlflow.log_artifact(out_path)

    mlflow.log_params({
        "feature_importance_logged": True,
        "feature_importance_kind": kind,
        "feature_importance_top_k": top_k,
    })



def save_and_log_plot(fig, filename: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(path))


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, fp_cost=1.0, fn_cost=5.0):
    """
    Minimize expected cost:
      FP -> reject good client
      FN -> accept bad client
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best = None
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        cost = fp_cost * fp + fn_cost * fn
        if best is None or cost < best["cost"]:
            best = {"threshold": float(t), "fp": int(fp), "fn": int(fn), "cost": float(cost)}
    return best


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


def make_split(X, y, test_size, val_size, seed, date_col=None):
    if date_col and date_col in X.columns:
        df = X.copy()
        df["_y"] = y
        df = df.sort_values(date_col)

        n = len(df)
        n_test = int(n * test_size)
        test = df.iloc[-n_test:]
        trainval = df.iloc[:-n_test]

        # val is last chunk of trainval
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


def top_reasons_logreg(pipe: Pipeline, x_row: pd.DataFrame, top_k: int = 3):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    if not hasattr(clf, "coef_"):
        return []

    feat_names = pre.get_feature_names_out()
    Xtr = pre.transform(x_row)

    # Xtr może być sparse
    contrib = (Xtr @ clf.coef_.ravel()).A1 if hasattr(Xtr, "A1") else np.asarray(Xtr @ clf.coef_.ravel()).ravel()
    # Ale tu jest 1 wiersz -> wektor cech
    # dla 1 wiersza lepiej:
    v = Xtr[0]
    vec = v.toarray().ravel() if hasattr(v, "toarray") else np.asarray(v).ravel()
    contribs = vec * clf.coef_.ravel()

    idx = np.argsort(np.abs(contribs))[::-1][:top_k]
    reasons = [{"feature": feat_names[i], "contribution": float(contribs[i])} for i in idx]
    return reasons


def threshold_table(y_true, y_prob, fp_cost=1.0, fn_cost=5.0):
    rows = []
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        cost = fp_cost * fp + fn_cost * fn
        approval_rate = float(np.mean(y_pred == 0))  # jeśli 1=default, 0=good -> dopasuj jeśli masz odwrotnie
        rows.append({"threshold": float(t), "fp": fp, "fn": fn, "tp": tp, "tn": tn, "cost": float(cost), "approval_rate": approval_rate})
    return pd.DataFrame(rows).sort_values("cost")


def main():
    mlflow.end_run(status="KILLED")
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="bank-credit-risk")
    parser.add_argument("--run-name", default="day1-baseline")
    parser.add_argument("--model", choices=["logreg", "hgb", "rf"], default="logreg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)  # fraction of train (after test split)
    parser.add_argument("--date-col", default=None,
                        help="Optional column for time-based split (e.g., application_date)")
    parser.add_argument("--cv-folds", type=int, default=0, help="0 disables CV; e.g. 5 enables 5-fold CV on trainval")

    args = parser.parse_args()

    cfg = TrainConfig(
        experiment=args.experiment,
        run_name=args.run_name,
        model=args.model,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    X, y = load_credit_g()

    # # split: train+val vs test
    # X_trainval, X_test, y_trainval, y_test = train_test_split(
    #     X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    # )
    #
    # # split: train vs val
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_trainval, y_trainval, test_size=cfg.val_size, random_state=cfg.seed, stratify=y_trainval
    # )
    X_train, X_val, X_test, y_train, y_val, y_test, split_type = make_split(
        X, y, cfg.test_size, cfg.val_size, cfg.seed, date_col=args.date_col
    )
    mlflow.log_param("split_type", split_type)

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    pipe = build_pipeline(cfg.model, numeric_cols, categorical_cols)

    if args.cv_folds and args.cv_folds >= 2:
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=cfg.seed)
        scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True
        )



        mlflow.log_metrics({
            "cv_roc_auc_mean": float(np.mean(cv_res["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_res["test_roc_auc"])),
            "cv_pr_auc_mean": float(np.mean(cv_res["test_pr_auc"])),
            "cv_pr_auc_std": float(np.std(cv_res["test_pr_auc"])),
            "cv_train_roc_auc_mean": float(np.mean(cv_res["train_roc_auc"])),
            "cv_train_pr_auc_mean": float(np.mean(cv_res["train_pr_auc"])),
        })

        gap = float(np.mean(cv_res["train_roc_auc"]) - np.mean(cv_res["test_roc_auc"]))
        mlflow.log_metric("cv_train_test_roc_gap", gap)
        if gap > 0.08:
            mlflow.set_tag("overfitting_warning", f"High ROC gap: {gap:.3f}")

    mlflow.set_experiment(cfg.experiment)

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("n_rows", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_numeric", len(numeric_cols))
        mlflow.log_param("n_categorical", len(categorical_cols))

        pipe.fit(X_train, y_train)

        perm = permutation_importance(
            pipe, X_val, y_val, n_repeats=10, random_state=cfg.seed, n_jobs=-1, scoring="roc_auc"
        )
        imp = pd.DataFrame({
            "feature": X_val.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False).head(30)

        path = "perm_importance_top30.csv"
        imp.to_csv(path, index=False)
        mlflow.log_artifact(path)

        # predict proba for metrics
        y_val_prob = pipe.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_prob >= 0.5).astype(int)

        y_test_prob = pipe.predict_proba(X_test)[:, 1]

        best = find_best_threshold(
            y_test.to_numpy(), y_test_prob,
            fp_cost=1.0, fn_cost=5.0
        )
        mlflow.log_params({"fp_cost": 1.0, "fn_cost": 5.0})
        mlflow.log_metrics({
            "best_threshold": best["threshold"],
            "test_fp": best["fp"],
            "test_fn": best["fn"],
            "test_cost": best["cost"],
        })
        print(f"best_threshold={best['threshold']:.2f} test_cost={best['cost']} fp={best['fp']} fn={best['fn']}")

        tbl = threshold_table(y_test.to_numpy(), y_test_prob, fp_cost=1.0, fn_cost=5.0)
        tbl.to_csv("threshold_cost_table.csv", index=False)
        mlflow.log_artifact("threshold_cost_table.csv")

        y_test_pred = (y_test_prob >= best["threshold"]).astype(int)
        mlflow.log_metric("threshold_used_for_test_pred", best["threshold"])

        # Core metrics
        val_roc = roc_auc_score(y_val, y_val_prob)
        val_pr = average_precision_score(y_val, y_val_prob)
        test_roc = roc_auc_score(y_test, y_test_prob)
        test_pr = average_precision_score(y_test, y_test_prob)

        mlflow.log_metric("val_roc_auc", float(val_roc))
        mlflow.log_metric("val_pr_auc", float(val_pr))
        mlflow.log_metric("test_roc_auc", float(test_roc))
        mlflow.log_metric("test_pr_auc", float(test_pr))

        # Plots
        plot_roc(y_test.to_numpy(), y_test_prob)
        plot_pr(y_test.to_numpy(), y_test_prob)
        plot_confmat(y_test.to_numpy(), y_test_pred)
        plot_calibration(y_test.to_numpy(), y_test_prob)
        log_feature_importance(pipe, top_k=30)

        # Log model (sklearn pipeline)
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=None,  # registry later
        )

        if cfg.model == "logreg":
            sample = X_test.iloc[[0]]
            reasons = top_reasons_logreg(pipe, sample, top_k=3)
            pd.DataFrame(reasons).to_csv("sample_reasons.csv", index=False)
            mlflow.log_artifact("sample_reasons.csv")

        print("Done.")
        print(f"val_roc_auc={val_roc:.4f}, val_pr_auc={val_pr:.4f}")
        print(f"test_roc_auc={test_roc:.4f}, test_pr_auc={test_pr:.4f}")


if __name__ == "__main__":
    main()
