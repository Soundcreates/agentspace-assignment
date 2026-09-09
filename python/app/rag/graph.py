"""LangGraph workflow for ask: judge → (retrieve | skip) → answer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.rag.context import build_context
from app.rag.llm.generate import generate_answer
from app.rag.llm.judge import judge_rag_need
from app.rag.models import Index, SourceChunk
from app.rag.query.enhance import enhance_queries
from app.rag.vectorstores.chroma import retrieve_multi


class AskState(TypedDict, total=False):
    question: str
    index: Index
    k: int
    llm: Any
    judge_llm: Any
    on_token: Any
    source_names: list[str]
    rag_score: float
    use_rag: bool
    rag_reason: str
    queries: list[str]
    retrieved: list[SourceChunk]
    context: str
    source_ids: list[str]
    answer: str


def _judge_node(state: AskState) -> dict[str, Any]:
    index = state["index"]
    source_names = [
        Path(path).name for path in (index.source_paths or []) if path
    ]
    judge_llm = state.get("judge_llm")
    llm = state.get("llm")
    judgment = judge_rag_need(
        state["question"],
        source_names=source_names,
        llm=judge_llm if judge_llm is not None else llm,
    )
    return {
        "source_names": source_names,
        "rag_score": float(judgment["score"]),
        "use_rag": bool(judgment["use_rag"]),
        "rag_reason": str(judgment["reason"]),
    }


def _route_after_judge(state: AskState) -> Literal["retrieve", "skip_retrieve"]:
    return "retrieve" if state.get("use_rag") else "skip_retrieve"


def _retrieve_node(state: AskState) -> dict[str, Any]:
    question = state["question"]
    queries = enhance_queries(question)
    retrieved = retrieve_multi(state["index"], queries, k=state.get("k", 6)) if queries else []
    context, source_ids = build_context(retrieved)
    return {
        "queries": queries,
        "retrieved": retrieved,
        "context": context,
        "source_ids": sorted(source_ids),
    }


def _skip_retrieve_node(_state: AskState) -> dict[str, Any]:
    return {
        "queries": [],
        "retrieved": [],
        "context": "",
        "source_ids": [],
    }


def _answer_node(state: AskState) -> dict[str, Any]:
    answer = generate_answer(
        state["question"],
        state.get("context") or "",
        llm=state.get("llm"),
        on_token=state.get("on_token"),
    )
    return {"answer": answer}


def build_ask_graph():
    """Compile the ask StateGraph (judge → conditional retrieve → answer)."""
    graph = StateGraph(AskState)
    graph.add_node("judge", _judge_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("skip_retrieve", _skip_retrieve_node)
    graph.add_node("answer", _answer_node)

    graph.add_edge(START, "judge")
    graph.add_conditional_edges(
        "judge",
        _route_after_judge,
        {
            "retrieve": "retrieve",
            "skip_retrieve": "skip_retrieve",
        },
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip_retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


_ASK_GRAPH = None


def get_ask_graph():
    global _ASK_GRAPH
    if _ASK_GRAPH is None:
        _ASK_GRAPH = build_ask_graph()
    return _ASK_GRAPH


def run_ask(
    index: Index,
    question: str,
    k: int = 6,
    llm=None,
    on_token=None,
    judge_llm=None,
) -> dict[str, object]:
    """Invoke the LangGraph ask workflow and return the API-shaped result dict."""
    final = get_ask_graph().invoke(
        {
            "question": question,
            "index": index,
            "k": k,
            "llm": llm,
            "judge_llm": judge_llm,
            "on_token": on_token,
        }
    )
    retrieved: list[SourceChunk] = list(final.get("retrieved") or [])
    return {
        "question": question,
        "answer": final.get("answer") or "",
        "queries": list(final.get("queries") or []),
        "documents_count": len(index.chunks),
        "partial": index.partial,
        "source_paths": index.source_paths,
        "source_ids": list(final.get("source_ids") or []),
        "rag_score": final.get("rag_score"),
        "use_rag": bool(final.get("use_rag")),
        "rag_reason": final.get("rag_reason"),
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
