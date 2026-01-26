from __future__ import annotations

from typing import List

from src.rag.types import RetrievalHit


def build_prompt(question: str, hits: List[RetrievalHit]) -> str:
    """
    Simple prompt. Works even if you later swap in a real LLM.
    """
    ctx = []
    for i, h in enumerate(hits, start=1):
        ctx.append(f"[{i}] source={h.chunk.source} id={h.chunk.chunk_id}\n{h.chunk.text}")

    context_block = "\n\n---\n\n".join(ctx)

    return (
        "You are an assistant for a credit risk demo system.\n"
        "Answer using ONLY the provided context. If the answer is not in context, say you don't know.\n"
        "Keep it short and practical.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        "ANSWER:"
    )


def extractive_fallback_answer(question: str, hits: List[RetrievalHit]) -> str:
    """
    No-LLM fallback: return best snippet with a short header.
    Good enough for MVP if you don't want OpenAI/Ollama yet.
    """
    if not hits:
        return "I don't know (no relevant context found)."

    best = hits[0].chunk.text.strip()
    return f"(Extractive answer)\n\n{best}"
