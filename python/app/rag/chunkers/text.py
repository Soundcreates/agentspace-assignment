"""Text chunking and content deduplication."""

from __future__ import annotations

import hashlib

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


def deduplicate(chunks: list[SourceChunk]) -> list[SourceChunk]:
    seen: set[str] = set()
    unique: list[SourceChunk] = []
    for chunk in chunks:
        digest = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(chunk)
    return unique
