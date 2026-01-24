import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictRequest, PredictResponse, HealthResponse, Reason
from src.api.model_loader import load_model, LoadedModel, _get_tracking_uri


app = FastAPI(title="Bank ML Decisioning API", version="0.1.0")

LOADED: LoadedModel | None = None
THRESHOLD: float = float(os.getenv("MODEL_THRESHOLD", "0.5"))


def _ensure_loaded() -> LoadedModel:
    global LOADED
    if LOADED is not None:
        return LOADED

    run_id = os.getenv("MODEL_RUN_ID")  # preferowane
    local_path = os.getenv("MODEL_LOCAL_PATH", "artifacts/model")  # fallback

    LOADED = load_model(
        preferred_run_id=run_id,
        local_fallback_path=local_path,
        artifact_path="model",
    )
    return LOADED


def _predict_score(model, df: pd.DataFrame) -> float:
    proba = model.predict_proba(df)[:, 1]
    return float(proba[0])


def _decision(score: float, threshold: float) -> str:
    # w Twoim projekcie: positive_class=1 default/bad,
    # decision_rule=decline if score >= threshold
    return "decline" if score >= threshold else "approve"


def _top_reasons_logreg(model, df: pd.DataFrame, top_k: int = 3) -> List[Reason]:
    """
    Lokalna explainability tylko dla logreg:
    - liczymy wkłady cech po one-hot (coef * feature_value)
    """
    try:
        pre = model.named_steps["pre"]
        clf = model.named_steps["clf"]
    except Exception:
        return []

    if not hasattr(clf, "coef_"):
        return []

    feat_names = pre.get_feature_names_out()
    Xtr = pre.transform(df)

    v = Xtr[0]
    vec = v.toarray().ravel() if hasattr(v, "toarray") else np.asarray(v).ravel()
    contribs = vec * clf.coef_.ravel()

    idx = np.argsort(np.abs(contribs))[::-1][:top_k]
    out: List[Reason] = []
    for i in idx:
        out.append(Reason(feature=str(feat_names[i]), contribution=float(contribs[i])))
    return out


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _ensure_loaded()
    return HealthResponse(
        status="ok",
        model_version=loaded.model_version,
        tracking_uri=_get_tracking_uri(),
        source=loaded.source,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    loaded = _ensure_loaded()

    # 1) dict -> DataFrame 1-row
    try:
        df = pd.DataFrame([req.features])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid features payload: {e}")

    # 2) score
    try:
        score = _predict_score(loaded.model, df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model predict failed: {e}")

    # 3) decision
    decision = _decision(score, THRESHOLD)

    # 4) reasons (only for logreg)
    reasons = _top_reasons_logreg(loaded.model, df, top_k=3)

    return PredictResponse(
        score=score,
        decision=decision,
        threshold=THRESHOLD,
        reasons=reasons,
        model_version=loaded.model_version,
    )
