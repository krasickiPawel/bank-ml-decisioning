from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient


DEFAULT_TRACKING_URI = "http://localhost:5050"
DEFAULT_EXPERIMENT_NAME = "bank-credit-risk-http"
DEFAULT_MODEL_ARTIFACT_PATH = "model"

DEFAULT_BEST_MODEL_JSON = "best_model.json"
DEFAULT_FALLBACK_MODEL_PATH = "model.pkl"

DEFAULT_CACHE_DIR = "data/mlflow_cache"


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    run_id: str
    experiment_id: str
    tracking_uri: str

    source: str  # "mlflow-run" | "mlflow-registry" | "fallback-local"
    selected_by: str  # "registry" | "run_id" | "best_model_json" | "latest_finished" | "fallback"
    threshold: float
    expected_columns: list[str]

    # Optional run artifacts used by API/UI
    example_features: Optional[dict[str, Any]]
    drift_reference_path: Optional[str]

    # Registry metadata (optional)
    model_name: Optional[str]
    model_stage: Optional[str]
    model_alias: Optional[str]
    model_version: Optional[str]


def _set_tracking_uri() -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def _cache_dir() -> Path:
    return Path(os.getenv("MODEL_CACHE_DIR", DEFAULT_CACHE_DIR))


def _read_best_model_json(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    # support both keys (you used both variants in history)
    run_id = data.get("run_id") or data.get("best_run_id")
    return str(run_id) if run_id else None


def _download_artifact_if_exists(client: MlflowClient, run_id: str, artifact_path: str, dst_dir: Path) -> Optional[Path]:
    """
    Returns local path if artifact exists and was downloaded, else None.
    """
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        local_path = client.download_artifacts(run_id, artifact_path, dst_path=str(dst_dir))
        return Path(local_path)
    except Exception:
        return None


def _load_run_model_cached(client: MlflowClient, run_id: str, artifact_path: str) -> Any:
    """
    Avoids repeated MLflow fetches:
    - download model artifacts once into local cache dir
    - then load from local path
    """
    cache_root = _cache_dir()
    run_root = cache_root / "runs" / run_id
    model_dir = run_root / artifact_path

    # If already cached and looks like MLflow model dir → load directly.
    if (model_dir / "MLmodel").exists():
        return mlflow.sklearn.load_model(str(model_dir))

    # Otherwise download artifacts into run_root
    run_root.mkdir(parents=True, exist_ok=True)
    _download_artifact_if_exists(client, run_id, artifact_path, run_root)

    if not (model_dir / "MLmodel").exists():
        raise RuntimeError(f"Cached model dir is missing MLmodel: {model_dir}")

    return mlflow.sklearn.load_model(str(model_dir))


def _load_expected_columns(client: MlflowClient, run_id: str, model: Any) -> list[str]:
    cache_root = _cache_dir()
    run_root = cache_root / "runs" / run_id

    p = _download_artifact_if_exists(client, run_id, "expected_columns.json", run_root)
    if p and p.exists():
        cols = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols]

    cols2 = getattr(model, "feature_names_in_", None)
    if cols2 is not None:
        return [str(c) for c in cols2]

    raise RuntimeError(
        "Cannot determine expected columns. "
        "Train must log expected_columns.json or model must expose feature_names_in_."
    )


def _load_example_features(client: MlflowClient, run_id: str) -> Optional[dict[str, Any]]:
    cache_root = _cache_dir()
    run_root = cache_root / "runs" / run_id

    p = _download_artifact_if_exists(client, run_id, "example_payload.json", run_root)
    if not p or not p.exists():
        return None

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        feats = payload.get("features")
        return feats if isinstance(feats, dict) else None
    except Exception:
        return None


def _load_drift_reference_path(client: MlflowClient, run_id: str) -> Optional[str]:
    cache_root = _cache_dir()
    run_root = cache_root / "runs" / run_id

    p = _download_artifact_if_exists(client, run_id, "drift_reference_train.csv", run_root)
    if p and p.exists():
        return str(p)
    return None


def _load_threshold(client: MlflowClient, run_id: str) -> float:
    run = client.get_run(run_id)
    m = run.data.metrics
    for key in ("best_threshold_val", "threshold_used_for_test_pred"):
        if key in m:
            return float(m[key])
    return 0.5


def _latest_finished_run_id(experiment_name: str, max_results: int = 200) -> str:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {experiment_name}")

    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
        output_format="pandas",
    )
    if df.empty:
        raise RuntimeError(f"No FINISHED runs in experiment={experiment_name}")

    return str(df.iloc[0]["run_id"])


def _resolve_registry_run_id(client: MlflowClient, model_name: str, stage: str) -> tuple[str, str]:
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise RuntimeError(f"No registry versions for model={model_name} stage={stage}")
    mv = versions[0]
    return str(mv.run_id), str(mv.version)


def _load_fallback_local(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Fallback model not found at: {path}")

    if p.is_dir():
        return mlflow.sklearn.load_model(str(p))

    import joblib
    return joblib.load(str(p))


def load_model_bundle() -> ModelBundle:
    """
    Priority (demo-friendly):
    1) MODEL_NAME + MODEL_STAGE -> pick run_id from registry (manual promotion in UI)
       and load *run artifacts* from cache (fast + consistent).
    2) MODEL_RUN_ID -> load that run
    3) best_model.json -> load run_id from file
    4) latest FINISHED run in experiment
    5) fallback-local
    """
    tracking_uri = _set_tracking_uri()
    client = MlflowClient()

    artifact_path = os.getenv("MODEL_ARTIFACT_PATH", DEFAULT_MODEL_ARTIFACT_PATH)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)

    model_name = os.getenv("MODEL_NAME")
    model_stage = os.getenv("MODEL_STAGE", "Production")

    run_id: Optional[str] = None
    selected_by = "unknown"

    registry_version: Optional[str] = None
    source = "mlflow-run"

    if model_name:
        run_id, registry_version = _resolve_registry_run_id(client, model_name, model_stage)
        selected_by = "registry"
        source = "mlflow-registry"
    else:
        explicit = os.getenv("MODEL_RUN_ID")
        if explicit:
            run_id = explicit
            selected_by = "run_id"
        else:
            best_path = os.getenv("BEST_MODEL_JSON", DEFAULT_BEST_MODEL_JSON)
            run_id = _read_best_model_json(best_path)
            if run_id:
                selected_by = "best_model_json"
            else:
                run_id = _latest_finished_run_id(experiment_name)
                selected_by = "latest_finished"

    try:
        assert run_id is not None
        model = _load_run_model_cached(client, run_id, artifact_path)
        expected_columns = _load_expected_columns(client, run_id, model)
        threshold = _load_threshold(client, run_id)

        run_info = client.get_run(run_id).info
        experiment_id = str(run_info.experiment_id)

        example_features = _load_example_features(client, run_id)
        drift_reference_path = _load_drift_reference_path(client, run_id)

        return ModelBundle(
            model=model,
            run_id=run_id,
            experiment_id=experiment_id,
            tracking_uri=tracking_uri,
            source=source,
            selected_by=selected_by,
            threshold=float(threshold),
            expected_columns=expected_columns,
            example_features=example_features,
            drift_reference_path=drift_reference_path,
            model_name=model_name,
            model_stage=model_stage if model_name else None,
            model_alias=None,
            model_version=registry_version if model_name else None,
        )
    except Exception:
        # absolute fallback for demo
        fb = os.getenv("MODEL_FALLBACK_PATH", DEFAULT_FALLBACK_MODEL_PATH)
        model = _load_fallback_local(fb)
        cols = list(getattr(model, "feature_names_in_", []))
        return ModelBundle(
            model=model,
            run_id="local",
            experiment_id="0",
            tracking_uri=tracking_uri,
            source="fallback-local",
            selected_by="fallback",
            threshold=0.5,
            expected_columns=cols,
            example_features=None,
            drift_reference_path=None,
            model_name=None,
            model_stage=None,
            model_alias=None,
            model_version=None,
        )
