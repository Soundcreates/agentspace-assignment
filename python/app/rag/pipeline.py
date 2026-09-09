"""RAG orchestration: load → index → enhance query → retrieve → answer."""

from __future__ import annotations

from pathlib import Path

from app.rag.context import build_context
from app.rag.llm.generate import ABSTAIN, SYSTEM_PROMPT, generate_answer
from app.rag.loaders.local import SUPPORTED_EXTENSIONS, load_file, load_sources
from app.rag.models import Index, SourceChunk
from app.rag.query.enhance import enhance_queries, rewrite_query, split_query
from app.rag.chunkers.text import chunk_text, deduplicate
from app.rag.vectorstores.chroma import build_index, retrieve, retrieve_multi


def ask(
    index: Index,
    question: str,
    k: int = 6,
    llm=None,
    on_token=None,
) -> dict[str, object]:
    queries = enhance_queries(question)
    retrieved = retrieve_multi(index, queries, k=k) if queries else []
    context, source_ids = build_context(retrieved)
    result_answer = generate_answer(question, context, llm=llm, on_token=on_token)
    return {
        "question": question,
        "answer": result_answer,
        "queries": queries,
        "documents_count": len(index.chunks),
        "partial": index.partial,
        "source_paths": index.source_paths,
        "source_ids": sorted(source_ids),
        "retrieved_chunks": [
            {
                "id": chunk.metadata["chunk_id"],
                "filename": chunk.metadata.get("filename")
                or Path(str(chunk.metadata.get("source", ""))).name,
                "rank": rank,
                "text": chunk.text,
            }
            for rank, chunk in enumerate(retrieved, start=1)
        ],
    }


def build_from_paths(paths: list[Path], max_source_chars: int | None = None) -> Index:
    chunks, partial = load_sources(paths, max_source_chars=max_source_chars)
    resolved = [str(path.expanduser().resolve()) for path in paths]
    return build_index(chunks, partial=partial, source_paths=resolved)


# Backward-compatible re-exports for tests and callers.
__all__ = [
    "ABSTAIN",
    "SYSTEM_PROMPT",
    "SUPPORTED_EXTENSIONS",
    "Index",
    "SourceChunk",
    "ask",
    "build_context",
    "build_from_paths",
    "build_index",
    "chunk_text",
    "deduplicate",
    "enhance_queries",
    "generate_answer",
    "load_file",
    "load_sources",
    "retrieve",
    "retrieve_multi",
    "rewrite_query",
    "split_query",
]
