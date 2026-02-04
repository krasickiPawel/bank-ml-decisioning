"""
MVP: trening modelu credit scoring (German Credit) i zapis do plików.
Bez MLflow — tylko joblib + JSON. Uruchom z katalogu mvp: python train.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from sklearn.datasets import fetch_openml
except ImportError:
    fetch_openml = None

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15  # z train


def load_credit_g():
    """German Credit — OpenML 31 lub CSV z mvp/data/raw/credit-g.csv."""
    base = Path(__file__).resolve().parent
    raw = base / "data" / "raw" / "credit-g.csv"
    if raw.exists():
        df = pd.read_csv(raw)
    elif fetch_openml is not None:
        data = fetch_openml(data_id=31, as_frame=True, parser="auto")
        df = data.frame
        raw.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw, index=False)
    else:
        raise FileNotFoundError("Brak mvp/data/raw/credit-g.csv. Zainstaluj scikit-learn i uruchom train.py.")

    if "class" not in df.columns:
        raise ValueError("Oczekiwana kolumna 'class'.")
    y = (df["class"].astype(str).str.lower() == "bad").astype(int)
    X = df.drop(columns=["class"])
    return X, y


def main():
    X, y = load_credit_g()
    feature_cols = list(X.columns)
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        verbose=0,
    )
    pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=8)),
        ]
    )
    pipe.fit(X_tr, y_tr)

    # Threshold na validation: min. cost = FP*1 + FN*5, approval = score < thr
    proba_val = pipe.predict_proba(X_val)[:, 1]
    best_thr = 0.5
    best_cost = float("inf")
    for thr in [0.2, 0.25, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5]:
        pred = (proba_val >= thr).astype(int)
        fn = ((y_val == 1) & (pred == 0)).sum()
        fp = ((y_val == 0) & (pred == 1)).sum()
        cost = fp * 1 + fn * 5
        if cost < best_cost:
            best_cost = cost
            best_thr = thr

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ARTIFACTS_DIR / "model.pkl")

    (ARTIFACTS_DIR / "expected_columns.json").write_text(
        json.dumps({"expected_columns": feature_cols}, indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / "threshold.json").write_text(
        json.dumps({"threshold": best_thr}, indent=2),
        encoding="utf-8",
    )

    example = {c: None for c in feature_cols}
    example.update({"duration": 12, "credit_amount": 1000, "age": 35, "checking_status": "<0"})
    schema = {
        "expected_columns": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "example_features": example,
    }
    (ARTIFACTS_DIR / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)
    fn = ((y_test == 1) & (test_pred == 0)).sum()
    fp = ((y_test == 0) & (test_pred == 1)).sum()
    test_cost = fp * 1 + fn * 5

    print(f"Artifacts saved to: {ARTIFACTS_DIR}")
    print(f"Threshold (validation): {best_thr}")
    print(f"Test cost at this threshold: {test_cost} (FP={fp}, FN={fn})")


if __name__ == "__main__":
    main()
