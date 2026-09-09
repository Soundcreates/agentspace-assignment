"""RAG orchestration: load → index → LangGraph ask workflow."""

from __future__ import annotations

from pathlib import Path

from app.rag.context import build_context
from app.rag.graph import build_ask_graph, get_ask_graph, run_ask
from app.rag.llm.generate import ABSTAIN, SYSTEM_PROMPT, generate_answer
from app.rag.llm.judge import RAG_SCORE_THRESHOLD, judge_rag_need
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
    judge_llm=None,
) -> dict[str, object]:
    """Run the LangGraph ask workflow (judge → optional RAG → answer)."""
    return run_ask(
        index,
        question,
        k=k,
        llm=llm,
        on_token=on_token,
        judge_llm=judge_llm,
    )


def build_from_paths(paths: list[Path], max_source_chars: int | None = None) -> Index:
    chunks, partial = load_sources(paths, max_source_chars=max_source_chars)
    resolved = [str(path.expanduser().resolve()) for path in paths]
    return build_index(chunks, partial=partial, source_paths=resolved)


# Backward-compatible re-exports for tests and callers.
__all__ = [
    "ABSTAIN",
    "SYSTEM_PROMPT",
    "SUPPORTED_EXTENSIONS",
    "RAG_SCORE_THRESHOLD",
    "Index",
    "SourceChunk",
    "ask",
    "build_ask_graph",
    "build_context",
    "build_from_paths",
    "build_index",
    "chunk_text",
    "deduplicate",
    "enhance_queries",
    "generate_answer",
    "get_ask_graph",
    "judge_rag_need",
    "load_file",
    "load_sources",
    "retrieve",
    "retrieve_multi",
    "rewrite_query",
    "run_ask",
    "split_query",
]
