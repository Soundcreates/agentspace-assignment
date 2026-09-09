from __future__ import annotations

import hashlib

from app.rag.models import SourceChunk


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
