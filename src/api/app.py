# src/api/app.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.model_loader import ModelBundle, load_model_bundle
from src.api.request_logger import RequestLogEvent, log_event, now_ts, LOG_PATH
from src.monitoring.drift import compute_drift, load_requests_jsonl
from src.rag.service import RAGService


app = FastAPI(title="Credit Risk Decisioning API", version="0.1.0")

# German Credit (credit-g) numeric columns (stable + demo-friendly).
# Pandas dtype inference from JSON logs is unreliable (mixed None/str/int => object),
# so we keep a fixed list.
NUMERIC_COLS = {
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
}


# -------------------------
# Pydantic models
# -------------------------
class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(default_factory=dict)
    top_k_reasons: int = 3


class ReasonCode(BaseModel):
    feature: str
    contribution: float


class PredictResponse(BaseModel):
    score: float
    decision: str  # "approve" | "decline"
    threshold: float
    reasons: List[ReasonCode]
    run_id: str
    source: str
    selected_by: str
    missing_columns_filled: List[str]


class SchemaResponse(BaseModel):
    expected_columns: List[str]
    example_payload: Dict[str, Any]
    run_id: str
    source: str


class HealthResponse(BaseModel):
    status: str
    tracking_uri: str
    source: str
    selected_by: str
    run_id: Optional[str] = None
    threshold: Optional[float] = None

    clf_name: str
    reload_status: str
    reload_error: Optional[str] = None

    # Optional metadata when using Registry
    model_name: Optional[str] = None
    model_stage: Optional[str] = None
    model_alias: Optional[str] = None
    model_version: Optional[str] = None

    # Deep-link to MLflow run (best-effort)
    run_url: Optional[str] = None

    rag_indexed_run_id: Optional[str] = None
    rag_error: Optional[str] = None



class RAGRequest(BaseModel):
    question: str
    top_k: int = 4


class RAGResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    meta: Dict[str, Any]


# -------------------------
# App state
# -------------------------
@app.on_event("startup")
def _startup() -> None:
    _start_initial_load()
    # app.state.rag = RAGService(
    #     index_dir=Path("data/rag_index"),
    #     docs_dir=Path("rag_docs"),
    # )


def _set_bundle(b: ModelBundle) -> None:
    app.state.bundle = b
    app.state.rag = RAGService().load_or_build(
        docs_dir="rag_docs",
        tracking_uri=b.tracking_uri,
        run_id=b.run_id,
    )


def _get_bundle() -> Optional[ModelBundle]:
    return getattr(app.state, "bundle", None)


def _set_reload_status(status: str) -> None:
    app.state.reload_status = status


def _get_reload_status() -> str:
    return getattr(app.state, "reload_status", "idle")


def _set_reload_error(msg: Optional[str]) -> None:
    app.state.reload_error = msg


def _get_reload_error() -> Optional[str]:
    return getattr(app.state, "reload_error", None)


def _ensure_loaded() -> ModelBundle:
    b = _get_bundle()
    if b is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return b


def _start_initial_load() -> None:
    """
    Do not block Uvicorn startup.
    Streamlit can call /health and see reload_status=loading.
    """
    _set_reload_status("loading")
    _set_reload_error(None)

    def _worker() -> None:
        try:
            bundle = load_model_bundle()
            _set_bundle(bundle)
            _set_reload_status("idle")
            rag: RAGService = getattr(app.state, "rag", None)
            if rag is None:
                app.state.rag = RAGService().load_or_build(
                    docs_dir="rag_docs",
                    tracking_uri=bundle.tracking_uri,
                    run_id=bundle.run_id,
                )
            else:
                rag._tracking_uri = bundle.tracking_uri  # keep simple
                rag._run_id = bundle.run_id
                rag.rebuild_index()
        except Exception as e:
            _set_reload_error(str(e))
            _set_reload_status("error")
            app.state.rag_error = str(e)
        # try:
        #     bundle = load_model_bundle()
        #     _set_bundle(bundle)
        #     _set_reload_status("idle")
        # except Exception as e:
        #     _set_reload_error(str(e))
        #     _set_reload_status("error")

    threading.Thread(target=_worker, daemon=True).start()


# -------------------------
# Helpers
# -------------------------
def _decision(score: float, threshold: float) -> str:
    # Business rule used in this demo: decline if score >= threshold
    return "decline" if score >= threshold else "approve"


def _build_filled_features(
    features: Dict[str, Any],
    expected_cols: List[str],
) -> tuple[Dict[str, Any], List[str]]:
    missing = [c for c in expected_cols if c not in features]
    filled = {c: features.get(c, None) for c in expected_cols}
    return filled, missing


def _build_row_df(filled: Dict[str, Any], expected_cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame([filled], columns=expected_cols)


def _top_reasons_logreg(pipe, x_row_df: pd.DataFrame, top_k: int = 3) -> List[ReasonCode]:
    """
    Lightweight reasons for LogisticRegression:
    contribution_i = x_i * coef_i in transformed feature space.
    """
    try:
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
    except Exception:
        return []

    if not hasattr(clf, "coef_"):
        return []

    feat_names = pre.get_feature_names_out()
    x_tr = pre.transform(x_row_df)
    v = x_tr[0]
    vec = v.toarray().ravel() if hasattr(v, "toarray") else np.asarray(v).ravel()
    contribs = vec * clf.coef_.ravel()

    idx = np.argsort(np.abs(contribs))[::-1][:top_k]
    return [ReasonCode(feature=str(feat_names[i]), contribution=float(contribs[i])) for i in idx]


def _top_reasons_xgb_contribs(pipe, x_row_df: pd.DataFrame, top_k: int = 3) -> List[ReasonCode]:
    """
    XGBoost contribution breakdown without shap:
    booster.predict(pred_contribs=True)
    """
    try:
        import xgboost as xgb
    except Exception:
        return []

    try:
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
    except Exception:
        return []

    if not hasattr(clf, "get_booster"):
        return []

    feat_names = list(pre.get_feature_names_out())
    x_tr = pre.transform(x_row_df)
    dmat = xgb.DMatrix(x_tr, feature_names=feat_names)

    contrib = clf.get_booster().predict(dmat, pred_contribs=True)[0]  # last is bias
    names = feat_names + ["bias"]

    pairs = [(names[i], float(contrib[i])) for i in range(len(names)) if names[i] != "bias"]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return [ReasonCode(feature=f, contribution=c) for f, c in pairs[:top_k]]


def _top_reasons_feature_importance(pipe, x_row_df: pd.DataFrame, top_k: int = 3) -> List[ReasonCode]:
    """
    Fallback for tree models with feature_importances_ (RandomForest etc.).
    This is global importance, not local explanation, but still useful for demo.
    """
    try:
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
    except Exception:
        return []

    if not hasattr(clf, "feature_importances_"):
        return []

    feat_names = pre.get_feature_names_out()
    importances = np.asarray(clf.feature_importances_, dtype=float)

    if importances.size != len(feat_names) or importances.size == 0:
        return []

    idx = np.argsort(importances)[::-1][:top_k]
    return [ReasonCode(feature=str(feat_names[i]), contribution=float(importances[i])) for i in idx]


def _run_url(bundle: ModelBundle) -> Optional[str]:
    tracking_uri = getattr(bundle, "tracking_uri", None)
    experiment_id = getattr(bundle, "experiment_id", None)
    run_id = getattr(bundle, "run_id", None)
    if tracking_uri and experiment_id and run_id:
        return f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
    return None


# -------------------------
# Endpoints
# -------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    b = _get_bundle()
    reload_status = _get_reload_status()
    rag = getattr(app.state, "rag", None)
    rag_indexed_run_id = getattr(rag, "run_id", None) if rag is not None else None

    if b is None:
        return HealthResponse(
            status="ok",
            tracking_uri=str(getattr(app.state, "tracking_uri", "unknown")),
            source="not_loaded",
            selected_by="not_loaded",
            run_id=None,
            threshold=None,
            clf_name="unknown",
            reload_status=reload_status,
            reload_error=_get_reload_error(),
        )

    clf = getattr(b.model, "named_steps", {}).get("clf", None)
    clf_name = clf.__class__.__name__ if clf is not None else "unknown"

    return HealthResponse(
        status="ok",
        tracking_uri=b.tracking_uri,
        source=getattr(b, "source", "unknown"),
        selected_by=getattr(b, "selected_by", getattr(b, "source", "unknown")),
        run_id=getattr(b, "run_id", None),
        threshold=float(getattr(b, "threshold", 0.0)) if getattr(b, "threshold", None) is not None else None,
        clf_name=clf_name,
        reload_status=reload_status,
        reload_error=_get_reload_error(),
        model_name=getattr(b, "model_name", None),
        model_stage=getattr(b, "model_stage", None),
        model_alias=getattr(b, "model_alias", None),
        model_version=getattr(b, "model_version", None),
        run_url=_run_url(b),
        rag_indexed_run_id=rag_indexed_run_id,
        rag_error=getattr(app.state, "rag_error", None),
    )


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    if _get_reload_status() in {"loading", "reloading"}:
        raise HTTPException(status_code=503, detail="Model is loading/reloading. Try again in a moment.")

    b = _ensure_loaded()
    expected = list(getattr(b, "expected_columns", []))
    if not expected:
        raise HTTPException(status_code=500, detail="Model bundle has no expected_columns.")

    example_features = getattr(b, "example_features", None)
    if isinstance(example_features, dict) and example_features:
        example = dict(example_features)
    else:
        example = {c: None for c in expected}
        example.update({"duration": 12, "credit_amount": 1000, "age": 35, "checking_status": "<0"})

    return SchemaResponse(
        expected_columns=expected,
        example_payload={"features": example},
        run_id=b.run_id,
        source=getattr(b, "source", "unknown"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _get_reload_status() in {"loading", "reloading"}:
        raise HTTPException(status_code=503, detail="Model is loading/reloading. Try again in a moment.")

    b = _ensure_loaded()

    expected_cols = list(getattr(b, "expected_columns", []))
    if not expected_cols:
        raise HTTPException(status_code=500, detail="Model bundle has no expected_columns.")

    filled_features, missing_cols = _build_filled_features(req.features, expected_cols)
    x_df = _build_row_df(filled_features, expected_cols)

    try:
        score = float(b.model.predict_proba(x_df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model predict failed: {e}")

    threshold = float(getattr(b, "threshold", 0.5))
    decision = _decision(score, threshold)

    # Reasons (MVP): LogReg local, XGB local, RF global importances.
    reasons: List[ReasonCode] = []
    try:
        pipe = b.model
        clf = getattr(pipe, "named_steps", {}).get("clf", None)
        clf_name = clf.__class__.__name__ if clf is not None else ""

        if clf_name == "LogisticRegression":
            reasons = _top_reasons_logreg(pipe, x_df, top_k=req.top_k_reasons)
        elif clf_name == "XGBClassifier":
            reasons = _top_reasons_xgb_contribs(pipe, x_df, top_k=req.top_k_reasons)
        else:
            # RF / others with feature_importances_
            reasons = _top_reasons_feature_importance(pipe, x_df, top_k=req.top_k_reasons)
    except Exception:
        reasons = []

    # Request logging (must never break inference in demo)
    try:
        log_event(
            RequestLogEvent(
                ts=now_ts(),
                run_id=b.run_id,
                source=getattr(b, "source", "unknown"),
                threshold=threshold,
                score=float(score),
                decision=decision,
                features=filled_features,
                missing_filled=missing_cols,
            )
        )
    except Exception:
        pass

    return PredictResponse(
        score=float(score),
        decision=decision,
        threshold=threshold,
        reasons=reasons,
        run_id=b.run_id,
        source=getattr(b, "source", "unknown"),
        selected_by=getattr(b, "selected_by", getattr(b, "source", "unknown")),
        missing_columns_filled=missing_cols,
    )


@app.post("/reload", response_model=None)
def reload_model():
    """
    DEMO ONLY:
    Reload model bundle in the background so API stays responsive.
    Streamlit can poll /health -> reload_status.
    """
    if _get_reload_status() == "reloading":
        return {"status": "already_reloading"}

    prev = _get_bundle()
    prev_run_id = getattr(prev, "run_id", None) if prev else None

    _set_reload_status("reloading")
    _set_reload_error(None)

    def _worker() -> None:
        try:
            new_bundle = load_model_bundle()
            _set_bundle(new_bundle)
            _set_reload_status("idle")
        except Exception as e:
            _set_reload_error(str(e))
            _set_reload_status("error")

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "previous_run_id": prev_run_id}


@app.get("/metrics/drift", response_model=None)
def drift(limit: int = 200):
    """
    Drift based on request logs.

    Reference:
    - preferred: drift_reference_train.csv logged in the run (b.drift_reference_path)
    - fallback: first N requests as a proxy reference (demo-only)
    Always returns JSON (never 500) so Streamlit can safely call .json().
    """
    b = _get_bundle()
    if b is None:
        return {"status": "error", "message": "Model not loaded.", "results": []}

    cur_df = load_requests_jsonl(Path(LOG_PATH), limit=limit)
    if cur_df.empty:
        return {
            "status": "ok",
            "message": "No requests logged yet.",
            "run_id": getattr(b, "run_id", None),
            "limit": limit,
            "results": [],
        }

    expected = list(getattr(b, "expected_columns", []))
    if not expected:
        return {"status": "error", "message": "Bundle has no expected_columns.", "results": []}

    drift_ref = getattr(b, "drift_reference_path", None)
    if drift_ref and Path(drift_ref).exists():
        ref_df = pd.read_csv(drift_ref)
        reference_source = "run_artifact"
    else:
        # demo fallback: use older part of current window as reference
        n_ref = min(max(len(cur_df) // 2, 50), len(cur_df))
        ref_df = cur_df.head(n_ref).copy()
        reference_source = "fallback_from_requests"

    # Align columns defensively
    ref_df = ref_df.reindex(columns=expected)
    cur_df = cur_df.reindex(columns=expected)

    numeric_cols = [c for c in expected if c in NUMERIC_COLS]
    categorical_cols = [c for c in expected if c not in NUMERIC_COLS]

    results = compute_drift(
        ref_df=ref_df,
        cur_df=cur_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    out = [
        {
            "feature": r.feature,
            "type": r.feature_type,
            "psi": r.psi,
            "ks": r.ks,
            "n_ref": r.n_ref,
            "n_cur": r.n_cur,
        }
        for r in results
    ]
    out.sort(key=lambda x: float(max(x["psi"] or 0.0, x["ks"] or 0.0)), reverse=True)

    return {
        "status": "ok",
        "run_id": getattr(b, "run_id", None),
        "limit": limit,
        "n_current": int(len(cur_df)),
        "reference_source": reference_source,
        "rules_of_thumb": {
            "psi": "0.1 small, 0.2 moderate, 0.3+ large",
            "ks": "0.1 small, 0.2 moderate, 0.3+ large",
        },
        "results": out,
    }


@app.get("/rag/health", response_model=None)
def rag_health():
    svc: RAGService = app.state.rag
    return {"status": "ok", "rag": svc.status()}


class RAGAskRequest(BaseModel):
    question: str
    top_k: int = 4


@app.post("/rag/ask", response_model=None)
def rag_ask(req: RAGAskRequest):
    rag: RAGService = getattr(app.state, "rag", None)
    if rag is None:
        return {"status": "error", "message": "RAG not initialized"}
    return rag.ask(req.question, top_k=req.top_k)


@app.post("/rag/answer", response_model=None)
def rag_answer(req: RAGRequest):
    svc: RAGService = app.state.rag
    res = svc.answer(req.question, top_k=req.top_k)

    citations = [
        {
            "score": h.score,
            "source": h.chunk.source,
            "chunk_id": h.chunk.chunk_id,
            "text": h.chunk.text[:600],
        }
        for h in res.hits
    ]
    return {"answer": res.answer, "citations": citations, "meta": res.meta}
