# src/monitoring/drift.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-6


@dataclass(frozen=True)
class DriftResult:
    feature: str
    feature_type: str  # "numeric" | "categorical"
    psi: Optional[float]
    ks: Optional[float]
    n_ref: int
    n_cur: int


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPS, 1.0)
    actual = np.clip(actual, EPS, 1.0)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_hist(ref: pd.Series, cur: pd.Series, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ref.dropna(), q))

    # If too few unique bin edges (e.g. constant values), fallback to min/max bins.
    if len(edges) < 3:
        mn = float(ref.min()) if pd.notna(ref.min()) else 0.0
        mx = float(ref.max()) if pd.notna(ref.max()) else 1.0
        if mn == mx:
            mx = mn + 1.0
        edges = np.linspace(mn, mx, n_bins + 1)

    ref_counts, _ = np.histogram(ref.dropna(), bins=edges)
    cur_counts, _ = np.histogram(cur.dropna(), bins=edges)

    ref_p = ref_counts / max(ref_counts.sum(), 1)
    cur_p = cur_counts / max(cur_counts.sum(), 1)
    return ref_p, cur_p


def _categorical_freq(ref: pd.Series, cur: pd.Series, top_k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    ref = ref.fillna("__NA__").astype(str)
    cur = cur.fillna("__NA__").astype(str)

    ref_counts = ref.value_counts()
    keep = set(ref_counts.head(top_k).index.tolist())

    def squash(s: pd.Series) -> pd.Series:
        s = s.copy()
        s[~s.isin(list(keep))] = "__OTHER__"
        return s

    ref2 = squash(ref).value_counts()
    cur2 = squash(cur).value_counts()

    cats = sorted(set(ref2.index) | set(cur2.index))
    ref_p = np.array([ref2.get(c, 0) for c in cats], dtype=float)
    cur_p = np.array([cur2.get(c, 0) for c in cats], dtype=float)

    ref_p = ref_p / max(ref_p.sum(), 1)
    cur_p = cur_p / max(cur_p.sum(), 1)
    return ref_p, cur_p


def compute_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[DriftResult]:
    results: List[DriftResult] = []

    for col in numeric_cols:
        r = pd.to_numeric(ref_df[col], errors="coerce")
        c = pd.to_numeric(cur_df[col], errors="coerce")

        n_ref = int(r.dropna().shape[0])
        n_cur = int(c.dropna().shape[0])

        psi_val = None
        ks_val = None

        if n_ref >= 30 and n_cur >= 30:
            ref_p, cur_p = _numeric_hist(r, c, n_bins=10)
            psi_val = _psi(ref_p, cur_p)
            ks_val = float(ks_2samp(r.dropna().values, c.dropna().values).statistic)

        results.append(DriftResult(col, "numeric", psi_val, ks_val, n_ref, n_cur))

    for col in categorical_cols:
        r = ref_df[col]
        c = cur_df[col]

        n_ref = int(r.dropna().shape[0])
        n_cur = int(c.dropna().shape[0])

        psi_val = None
        if n_ref >= 30 and n_cur >= 30:
            ref_p, cur_p = _categorical_freq(r, c, top_k=30)
            psi_val = _psi(ref_p, cur_p)

        results.append(DriftResult(col, "categorical", psi_val, None, n_ref, n_cur))

    return results


def load_requests_jsonl(path: Path, limit: int = 200) -> pd.DataFrame:
    """
    Reads JSONL created by request_logger and returns a flat DataFrame of features,
    plus metadata columns (ts, run_id, decision, score).
    """
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    feat = pd.json_normalize(df["features"])

    feat["ts"] = df["ts"]
    feat["run_id"] = df["run_id"]
    feat["decision"] = df["decision"]
    feat["score"] = df["score"]

    feat = feat.sort_values("ts").tail(limit)
    return feat
