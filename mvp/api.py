"""
MVP: FastAPI — model ładowany z dysku przy starcie. Brak MLflow, brak sieci.
"""
from pathlib import Path
import json
import joblib
from typing import Any, Dict, List

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

app = FastAPI(title="Credit Risk MVP API", version="0.1.0")

# Stan ładowany przy starcie
_model = None
_expected_columns: List[str] = []
_threshold: float = 0.5
_example_features: Dict[str, Any] = {}


def _load_artifacts() -> None:
    global _model, _expected_columns, _threshold, _example_features
    model_path = ARTIFACTS_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Brak modelu: {model_path}. Uruchom najpierw: python train.py")

    _model = joblib.load(model_path)
    ec_path = ARTIFACTS_DIR / "expected_columns.json"
    if ec_path.exists():
        data = json.loads(ec_path.read_text(encoding="utf-8"))
        _expected_columns = data.get("expected_columns", list(_model.feature_names_in_))
    else:
        _expected_columns = list(getattr(_model, "feature_names_in_", []))

    thr_path = ARTIFACTS_DIR / "threshold.json"
    if thr_path.exists():
        _threshold = float(json.loads(thr_path.read_text(encoding="utf-8")).get("threshold", 0.5))

    schema_path = ARTIFACTS_DIR / "schema.json"
    if schema_path.exists():
        sch = json.loads(schema_path.read_text(encoding="utf-8"))
        _example_features = sch.get("example_features", {c: None for c in _expected_columns})
    else:
        _example_features = {c: None for c in _expected_columns}


@app.on_event("startup")
def startup():
    try:
        _load_artifacts()
    except Exception as e:
        # API wystartuje, ale /schema i /predict zwrócą 503 bez modelu
        print(f"Startup: model nie załadowany — {e}")


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    score: float
    decision: str
    threshold: float


class SchemaResponse(BaseModel):
    expected_columns: List[str]
    example_payload: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    threshold: float
    clf_name: str


def _decision(score: float, threshold: float) -> str:
    return "decline" if score >= threshold else "approve"


@app.get("/health", response_model=HealthResponse)
def health():
    if _model is None:
        return HealthResponse(status="model_not_loaded", threshold=0.5, clf_name="—")
    clf = getattr(_model, "named_steps", {}) or {}
    clf = clf.get("clf", _model)
    clf_name = getattr(clf, "__class__", type(clf)).__name__
    return HealthResponse(
        status="ok",
        threshold=_threshold,
        clf_name=clf_name,
    )


@app.get("/schema", response_model=SchemaResponse)
def schema():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model nie załadowany. Uruchom train.py.")
    return SchemaResponse(
        expected_columns=_expected_columns,
        example_payload={"features": _example_features},
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model nie załadowany. Uruchom train.py.")
    expected = _expected_columns or list(getattr(_model, "feature_names_in_", []))
    if not expected:
        raise HTTPException(status_code=500, detail="Brak expected_columns.")

    filled = {c: req.features.get(c) for c in expected}
    X = pd.DataFrame([filled], columns=expected)
    try:
        score = float(_model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predykcja nie powiodła się: {e}")

    decision = _decision(score, _threshold)
    return PredictResponse(score=score, decision=decision, threshold=_threshold)
