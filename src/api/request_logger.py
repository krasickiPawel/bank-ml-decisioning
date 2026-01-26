# src/api/request_logger.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

LOG_PATH = Path(os.getenv("REQUEST_LOG_PATH", "data/requests.jsonl"))


@dataclass(frozen=True)
class RequestLogEvent:
    ts: float
    run_id: str
    source: str
    threshold: float
    score: float
    decision: str
    features: Dict[str, Any]
    missing_filled: list[str]
    request_id: Optional[str] = None


def now_ts() -> float:
    return time.time()


def log_event(evt: RequestLogEvent) -> None:
    """
    Append one prediction event to a JSONL file.

    This is intentionally simple for demo:
    - no PII handling
    - no encryption
    - no retention policy
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "ts": evt.ts,
        "run_id": evt.run_id,
        "source": evt.source,
        "threshold": evt.threshold,
        "score": evt.score,
        "decision": evt.decision,
        "features": evt.features,
        "missing_filled": evt.missing_filled,
        "request_id": evt.request_id,
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
