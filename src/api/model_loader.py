import os
from dataclasses import dataclass
from typing import Optional, Tuple

import mlflow
import mlflow.sklearn


@dataclass
class LoadedModel:
    model: object  # sklearn Pipeline
    model_version: str  # run_id albo "local"
    source: str  # "mlflow" | "local"


def _get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")


def load_model_from_mlflow(run_id: str, artifact_path: str = "model") -> object:
    tracking_uri = _get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.sklearn.load_model(uri)


def load_model_local(local_path: str) -> object:
    # local_path może być np. "artifacts/model"
    # mlflow.sklearn.load_model działa też na ścieżce lokalnej do modelu MLflow
    return mlflow.sklearn.load_model(local_path)


def load_model(
    preferred_run_id: Optional[str],
    local_fallback_path: str,
    artifact_path: str = "model",
) -> LoadedModel:
    """
    Preferowane: MLflow run_id (ENV: MODEL_RUN_ID).
    Fallback: lokalna ścieżka do modelu (ENV: MODEL_LOCAL_PATH).
    """
    tracking_uri = _get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    if preferred_run_id:
        try:
            model = load_model_from_mlflow(preferred_run_id, artifact_path=artifact_path)
            return LoadedModel(model=model, model_version=preferred_run_id, source="mlflow")
        except Exception as e:
            # spadamy do fallback
            pass

    model = load_model_local(local_fallback_path)
    return LoadedModel(model=model, model_version="local", source="local")
