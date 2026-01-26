from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.rag.types import Chunk


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """
    Super simple chunking:
    - split by paragraphs first
    - then pack into ~chunk_size chars
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: List[str] = []
    buf = ""

    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip()
            continue

        if buf:
            out.append(buf)

        if overlap and out:
            tail = out[-1][-overlap:]
            buf = (tail + "\n\n" + p).strip()
        else:
            buf = p

    if buf:
        out.append(buf)

    return out


def load_chunks_from_dir(
    docs_dir: Path,
    glob_pattern: str = "*.md",
    chunk_size: int = 900,
    overlap: int = 120,
) -> List[Chunk]:
    docs_dir = Path(docs_dir)
    paths = sorted(docs_dir.glob(glob_pattern))

    chunks: List[Chunk] = []
    for path in paths:
        text = _read_text_file(path)
        parts = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{path.name}::chunk{i:03d}",
                    source=str(path),
                    text=part,
                )
            )
    return chunks
