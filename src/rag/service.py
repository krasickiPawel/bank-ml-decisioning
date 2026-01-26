from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.mlflow_artifacts import ArtifactDoc, load_mlflow_artifact_docs
from src.rag.index import RAGIndex, build_index, search
from src.rag.io import load_chunks_from_dir
from src.rag.generator import build_prompt, extractive_fallback_answer
from src.rag.types import RAGAnswer, RetrievalHit


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_id: str
    text: str


# @dataclass
# class RAGService:
#     index_dir: Path
#     docs_dir: Path
#     embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
#
#     _index: Optional[RAGIndex] = None
#
#     def load_or_build(self) -> None:
#         if (self.index_dir / "faiss.index").exists() and (self.index_dir / "meta.json").exists():
#             self._index = RAGIndex.load(self.index_dir)
#             return
#
#         chunks = load_chunks_from_dir(self.docs_dir, glob_pattern="*.md")
#         idx = build_index(chunks, embed_model_name=self.embed_model_name)
#         idx.save(self.index_dir)
#         self._index = idx
#
#     def status(self) -> Dict[str, Any]:
#         if self._index is None:
#             return {"loaded": False}
#         return {
#             "loaded": True,
#             "model_name": self._index.model_name,
#             "chunks": len(self._index.chunks),
#             "dim": self._index.dim,
#             "index_dir": str(self.index_dir),
#         }
#
#     def answer(self, question: str, top_k: int = 4) -> RAGAnswer:
#         if self._index is None:
#             self.load_or_build()
#
#         assert self._index is not None
#         hits = search(self._index, query=question, top_k=top_k)
#
#         # MVP: no LLM yet -> extractive answer
#         answer = extractive_fallback_answer(question, hits)
#
#         return RAGAnswer(
#             answer=answer,
#             hits=hits,
#             meta={"top_k": top_k, "mode": "extractive"},
#         )

class RAGService:
    """
    Simple, demo-friendly RAG:
    - loads docs from rag_docs/
    - optionally loads MLflow artifacts for a given run_id
    - chunks text and builds a TF-IDF index
    - answers in an extractive way + returns citations
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

        self._docs_dir: Optional[Path] = None
        self._tracking_uri: Optional[str] = None
        self._run_id: Optional[str] = None

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    def load_or_build(
        self,
        docs_dir: str | Path,
        tracking_uri: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> "RAGService":
        self._docs_dir = Path(docs_dir)
        self._tracking_uri = tracking_uri
        self._run_id = run_id
        self.rebuild_index()
        return self

    def rebuild_index(self) -> None:
        if not self._docs_dir:
            raise RuntimeError("docs_dir not set")

        docs: list[tuple[str, str]] = []
        docs.extend(self._load_markdown_docs(self._docs_dir))

        if self._tracking_uri and self._run_id:
            art_docs = load_mlflow_artifact_docs(
                tracking_uri=self._tracking_uri,
                run_id=self._run_id,
            )
            docs.extend([(d.source, d.text) for d in art_docs])

        self._chunks = self._chunk_docs(docs)
        self._build_index(self._chunks)

    def ask(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        if not self._vectorizer or self._matrix is None or not self._chunks:
            return {
                "status": "error",
                "message": "RAG index is empty. Rebuild the index first.",
                "question": question,
            }

        top_k = max(1, int(top_k))

        q = question.strip()
        if not q:
            return {"status": "error", "message": "Empty question"}

        q_vec = self._vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self._matrix)[0]  # shape: (n_chunks,)
        idx = np.argsort(sims)[::-1][:top_k]

        selected = [self._chunks[i] for i in idx]
        citations = [
            {
                "score": float(sims[i]),
                "source": self._chunks[i].source,
                "chunk_id": self._chunks[i].chunk_id,
                "text": self._chunks[i].text,
            }
            for i in idx
        ]

        answer = self._extractive_answer(selected)
        return {
            "status": "ok",
            "question": question,
            "top_k": top_k,
            "answer_type": "extractive",
            "answer": answer,
            "citations": citations,
            "indexed_run_id": self._run_id,
        }

    # -------------------------
    # Internals
    # -------------------------

    def _load_markdown_docs(self, docs_dir: Path) -> list[tuple[str, str]]:
        if not docs_dir.exists():
            return []

        out: list[tuple[str, str]] = []
        for p in sorted(docs_dir.glob("*.md")):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            out.append((str(p.as_posix()), txt))
        return out

    def _chunk_docs(self, docs: list[tuple[str, str]]) -> list[Chunk]:
        """
        Chunk by paragraphs, keep chunks reasonably small.
        This is stable and predictable for demos.
        """
        chunks: list[Chunk] = []
        for source, text in docs:
            parts = self._split_paragraphs(text)
            buff: list[str] = []
            char_count = 0
            chunk_idx = 0

            for part in parts:
                if not part.strip():
                    continue

                if char_count + len(part) > 1200 and buff:
                    joined = "\n\n".join(buff).strip()
                    chunks.append(
                        Chunk(
                            source=source,
                            chunk_id=f"{Path(source).name}::chunk{chunk_idx:03d}",
                            text=joined,
                        )
                    )
                    chunk_idx += 1
                    buff = []
                    char_count = 0

                buff.append(part)
                char_count += len(part)

            if buff:
                joined = "\n\n".join(buff).strip()
                chunks.append(
                    Chunk(
                        source=source,
                        chunk_id=f"{Path(source).name}::chunk{chunk_idx:03d}",
                        text=joined,
                    )
                )

        return chunks

    def _build_index(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]

        # No language-specific stopwords on purpose (docs may be PL/EN).
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50_000,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def _split_paragraphs(self, text: str) -> list[str]:
        # Normalize newlines and split on blank lines
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = re.split(r"\n\s*\n+", t)
        return [p.strip() for p in parts if p.strip()]

    def _extractive_answer(self, chunks: list[Chunk]) -> str:
        """
        Very simple extractive answer: take top chunks and keep only the most informative lines.
        """
        lines: list[str] = []
        seen = set()

        for ch in chunks:
            for ln in ch.text.splitlines():
                ln2 = ln.strip()
                if not ln2:
                    continue
                if ln2 in seen:
                    continue
                if len(ln2) < 3:
                    continue
                seen.add(ln2)
                lines.append(ln2)

        # Keep it short enough for the UI
        if len(lines) > 40:
            lines = lines[:40] + ["...", "[truncated]"]

        return "\n".join(lines)