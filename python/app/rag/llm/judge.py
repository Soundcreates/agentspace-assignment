"""Judge LLM: score whether the question needs help from the RAG service."""

from __future__ import annotations

import json
import os
import re
from typing import Any

RAG_SCORE_THRESHOLD = 60.0

JUDGE_SYSTEM_PROMPT = """You are a retrieval-routing judge for a local document Q&A system.

Given the user's question (and optional source file names), decide how much the answer would benefit from retrieving passages from the user's uploaded files via the RAG service.

Score from 0 to 100:
- 0–40: general knowledge / chit-chat / math / definitions the model already knows well; RAG unlikely to help
- 41–60: mixed; RAG optional
- 61–100: project-specific, document-specific, or likely answered only from the user's files; RAG should run

Respond with ONLY a compact JSON object, no markdown:
{"score": <number 0-100>, "reason": "<short phrase>"}
"""


def _message_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


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


def _parse_score(raw: str) -> tuple[float, str] | None:
    text = raw.strip()
    if not text:
        return None
    # Strip optional markdown fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", text, flags=re.S)
        if brace:
            text = brace.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"score[\"']?\s*[:=]\s*(\d+(?:\.\d+)?)", raw, flags=re.I)
        if not match:
            return None
        return float(match.group(1)), "parsed from free text"
    if not isinstance(data, dict):
        return None
    score_raw = data.get("score")
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        return None
    reason = str(data.get("reason") or "").strip() or "no reason"
    return max(0.0, min(100.0, score)), reason


def _heuristic_judgment(question: str, source_names: list[str] | None) -> dict[str, Any]:
    """Offline / fallback: lean toward RAG when sources exist and the question looks document-bound."""
    cleaned = " ".join(question.strip().split())
    lower = cleaned.lower()
    general = (
        "hello",
        "hi ",
        "what is 2+",
        "who are you",
        "tell me a joke",
        "capital of",
    )
    if any(lower.startswith(g) or g in lower for g in general) and not source_names:
        score = 20.0
        reason = "looks like general knowledge"
    elif any(
        token in lower
        for token in (
            "this document",
            "this file",
            "according to",
            "in the pdf",
            "in my notes",
            "from the source",
            "what does the",
            "summarize",
            "based on",
        )
    ):
        score = 85.0
        reason = "question references user documents"
    elif source_names:
        score = 70.0
        reason = "sources attached; default toward RAG"
    else:
        score = 55.0
        reason = "ambiguous; optional RAG"
    return {
        "score": score,
        "use_rag": score > RAG_SCORE_THRESHOLD,
        "reason": reason,
    }


def judge_rag_need(
    question: str,
    *,
    source_names: list[str] | None = None,
    llm=None,
) -> dict[str, Any]:
    """Return ``{score, use_rag, reason}``. ``use_rag`` is True only when score > 60."""
    cleaned = " ".join(question.strip().split())
    if not cleaned:
        return {"score": 0.0, "use_rag": False, "reason": "empty question"}

    if llm is None and not os.getenv("OPENROUTER_API_KEY"):
        return _heuristic_judgment(cleaned, source_names)

    try:
        chat = llm or _default_llm()
        from langchain_core.messages import HumanMessage, SystemMessage

        names = ", ".join(source_names) if source_names else "(none listed)"
        user_prompt = (
            f"SOURCE FILE NAMES: {names}\n"
            f"QUESTION: {cleaned}\n\n"
            'Return JSON only: {"score": <0-100>, "reason": "<short>"}'
        )
        response = chat.invoke(
            [
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        raw = (
            response.content
            if isinstance(response.content, str)
            else _message_text(response.content)
        )
        parsed = _parse_score(raw)
        if parsed is None:
            return _heuristic_judgment(cleaned, source_names)
        score, reason = parsed
        return {
            "score": score,
            "use_rag": score > RAG_SCORE_THRESHOLD,
            "reason": reason,
        }
    except Exception:
        return _heuristic_judgment(cleaned, source_names)
