# src/rag/mlflow_artifacts.py
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


DEFAULT_ARTIFACT_PATHS: list[str] = [
    "feature_importance_top.csv",
    "perm_importance_top30.csv",
    "threshold_cost_table_val.csv",
    "threshold_cost_table_test.csv",
    "cv_results.csv",
    "sample_reasons.csv",
    "expected_columns.json",
    "example_payload.json",
    "drift_reference_train.csv",  # we don't index the whole csv content by default (see below)
]


@dataclass(frozen=True)
class ArtifactDoc:
    source: str  # e.g. "mlflow:RUN_ID/feature_importance_top.csv"
    text: str


def _read_text_file(path: Path, max_chars: int = 40_000) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars] + "\n\n[truncated]"


def _render_csv_as_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    head = df.head(max_rows).copy()
    return head.to_markdown(index=False)


def _download_artifact(client: MlflowClient, run_id: str, artifact_path: str) -> Optional[Path]:
    """
    Returns local path if download succeeded, else None.
    """
    try:
        with tempfile.TemporaryDirectory() as td:
            local = client.download_artifacts(run_id, artifact_path, dst_path=td)
            return Path(local)
    except Exception:
        return None


def _artifact_to_doc_text(path: Path, artifact_path: str) -> Optional[str]:
    """
    Convert artifact to a compact text block for RAG indexing.
    Keep it readable and stable for demos.
    """
    suffix = path.suffix.lower()

    # For drift reference: it's often large and mostly raw data.
    # Better to index a short "header" + few rows for explainability.
    if artifact_path == "drift_reference_train.csv":
        try:
            df = pd.read_csv(path)
            cols = list(df.columns)
            sample = df.head(5)
            return (
                "# Drift reference (train)\n"
                "This file is used as baseline distribution for PSI/KS drift.\n\n"
                f"Columns ({len(cols)}): {cols}\n\n"
                "Sample rows:\n"
                f"{_render_csv_as_markdown(sample, max_rows=5)}\n"
            )
        except Exception:
            return None

    if suffix == ".csv":
        try:
            df = pd.read_csv(path)
            return f"# Artifact: {artifact_path}\n\n{_render_csv_as_markdown(df, max_rows=40)}\n"
        except Exception:
            return None

    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            pretty = json.dumps(data, indent=2, ensure_ascii=False)
            return f"# Artifact: {artifact_path}\n\n```json\n{pretty}\n```\n"
        except Exception:
            return None

    if suffix in {".md", ".txt", ".log"}:
        txt = _read_text_file(path)
        return f"# Artifact: {artifact_path}\n\n{txt}\n"

    return None


def load_mlflow_artifact_docs(
    tracking_uri: str,
    run_id: str,
    artifact_paths: Optional[Iterable[str]] = None,
) -> List[ArtifactDoc]:
    """
    Downloads selected artifacts from a run and converts them to text docs for RAG.

    Design goals:
    - best-effort: if some artifacts are missing (older runs), we still index what exists
    - keep artifacts readable (tables truncated, json pretty-printed)
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    paths = list(artifact_paths) if artifact_paths is not None else DEFAULT_ARTIFACT_PATHS
    out: list[ArtifactDoc] = []

    for ap in paths:
        local_path = _download_artifact(client, run_id, ap)
        if not local_path:
            continue

        doc_text = _artifact_to_doc_text(local_path, ap)
        if not doc_text:
            continue

        source = f"mlflow:{run_id}/{ap}"
        out.append(ArtifactDoc(source=source, text=doc_text))

    return out
