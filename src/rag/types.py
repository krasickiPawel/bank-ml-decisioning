from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float  # higher = more similar


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    hits: List[RetrievalHit]
    meta: Dict[str, Any]
