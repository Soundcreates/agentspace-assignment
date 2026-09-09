from __future__ import annotations

from app.rag.models import SourceChunk
from app.rag.util import env_int


def chunk_text(text: str, metadata: dict[str, object]) -> list[SourceChunk]:
    size = env_int("RAG_CHUNK_SIZE", 1000)
    overlap = min(env_int("RAG_CHUNK_OVERLAP", 200), size // 2)
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[SourceChunk] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(SourceChunk(cleaned[start:end], dict(metadata)))
        if end == len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks
