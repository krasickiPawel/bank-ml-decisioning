from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.types import Chunk, RetrievalHit


@dataclass
class RAGIndex:
    model_name: str
    dim: int
    chunks: List[Chunk]
    index: faiss.Index

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(out_dir / "faiss.index"))
        meta = {
            "model_name": self.model_name,
            "dim": self.dim,
            "chunks": [
                {"chunk_id": c.chunk_id, "source": c.source, "text": c.text}
                for c in self.chunks
            ],
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(dir_path: Path) -> "RAGIndex":
        dir_path = Path(dir_path)
        meta = json.loads((dir_path / "meta.json").read_text(encoding="utf-8"))
        idx = faiss.read_index(str(dir_path / "faiss.index"))
        chunks = [Chunk(**c) for c in meta["chunks"]]
        return RAGIndex(
            model_name=meta["model_name"],
            dim=int(meta["dim"]),
            chunks=chunks,
            index=idx,
        )


def build_index(
    chunks: List[Chunk],
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> RAGIndex:
    """
    Uses cosine similarity via normalized vectors + inner product.
    """
    model = SentenceTransformer(embed_model_name)

    texts = [c.text for c in chunks]
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")

    faiss.normalize_L2(emb)  # cosine via inner product
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    return RAGIndex(model_name=embed_model_name, dim=dim, chunks=chunks, index=index)


def search(
    rag_index: RAGIndex,
    query: str,
    top_k: int = 4,
) -> List[RetrievalHit]:
    model = SentenceTransformer(rag_index.model_name)
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)

    scores, ids = rag_index.index.search(q, top_k)
    hits: List[RetrievalHit] = []

    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        hits.append(RetrievalHit(chunk=rag_index.chunks[int(idx)], score=float(score)))

    return hits
