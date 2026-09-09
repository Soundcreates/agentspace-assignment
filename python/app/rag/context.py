"""Context assembly for the LLM prompt."""

from __future__ import annotations

from app.rag.models import SourceChunk
from app.rag.util import env_int


def build_context(documents: list[SourceChunk]) -> tuple[str, set[str]]:
    """Build labeled source context for the LLM prompt."""
    max_doc_chars = env_int("RAG_CONTEXT_DOC_CHARS", 1400)
    max_total_chars = env_int("RAG_CONTEXT_TOTAL_CHARS", 9000)
    current_total = 0
    sections: list[str] = []
    source_ids: set[str] = set()
    for document in documents:
        chunk_id = str(document.metadata.get("chunk_id", ""))
        if not chunk_id:
            continue
        label = document.metadata.get("filename") or document.metadata.get("source") or "source"
        entry = f"[Source {chunk_id}: {label}]\n{document.text[:max_doc_chars]}"
        if current_total + len(entry) > max_total_chars:
            break
        sections.append(entry)
        source_ids.add(chunk_id)
        current_total += len(entry)
    return "\n\n".join(sections), source_ids
