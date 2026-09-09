"""Query rewriting and multi-part query splitting for better retrieval."""

from __future__ import annotations

import os
import re


def split_query(question: str) -> list[str]:
    """Split a question into retrieval-friendly sub-queries / sentences."""
    cleaned = " ".join(question.strip().split())
    if not cleaned:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\s*;\s*|\s+\band\b\s+|\s+\bas well as\b\s+", cleaned, flags=re.I)
    subqueries: list[str] = []
    for part in parts:
        piece = part.strip(" ,;")
        if not piece:
            continue
        if not piece.endswith("?"):
            # Keep short clause fragments usable as retrieval queries.
            if len(piece.split()) >= 3:
                subqueries.append(piece)
        else:
            subqueries.append(piece)

    # Also split on commas for "X, Y, and Z" style multi-aspect questions when long enough.
    if len(subqueries) <= 1 and "," in cleaned:
        for part in re.split(r",\s*", cleaned):
            piece = part.strip(" ?")
            if len(piece.split()) >= 4:
                subqueries.append(piece if piece.endswith("?") else piece)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in [cleaned, *subqueries]:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def rewrite_query(question: str, llm=None) -> str:
    """Rewrite the user question into a clearer retrieval query."""
    cleaned = " ".join(question.strip().split())
    if not cleaned:
        return ""

    if llm is None and not os.getenv("OPENROUTER_API_KEY"):
        return _heuristic_rewrite(cleaned)

    try:
        chat = llm or _default_llm()
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(
                content=(
                    "Rewrite the user's question into one concise search query for document retrieval. "
                    "Keep key entities and intents. Do not answer the question. "
                    "Return only the rewritten query text."
                )
            ),
            HumanMessage(content=cleaned),
        ]
        response = chat.invoke(messages)
        text = response.content if isinstance(response.content, str) else str(response.content)
        rewritten = " ".join(text.strip().split())
        return rewritten or _heuristic_rewrite(cleaned)
    except Exception:
        return _heuristic_rewrite(cleaned)


def _heuristic_rewrite(question: str) -> str:
    text = question
    text = re.sub(r"^(can you|could you|please|tell me|explain|what about)\s+", "", text, flags=re.I)
    text = re.sub(r"\?+$", "", text).strip()
    return text or question


def _default_llm():
    from langchain_openai import ChatOpenAI

    headers: dict[str, str] = {}
    title = os.getenv("OPENROUTER_TITLE", "agentspace-rag")
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    if title:
        headers["X-Title"] = title
    if referer:
        headers["HTTP-Referer"] = referer
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0,
        streaming=False,
        default_headers=headers or None,
    )


def enhance_queries(question: str, llm=None) -> list[str]:
    """Build a set of retrieval queries: original, rewrite, and split parts."""
    cleaned = " ".join(question.strip().split())
    if not cleaned:
        return []

    queries = split_query(cleaned)
    rewritten = rewrite_query(cleaned, llm=llm)
    if rewritten and rewritten.lower() not in {q.lower() for q in queries}:
        queries.insert(1, rewritten)

    # Cap to keep retrieval latency bounded.
    max_queries = int(os.getenv("RAG_MAX_QUERY_VARIANTS", "5"))
    return queries[:max_queries]
