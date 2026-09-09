"""Local RAG: load files, chunk, local Chroma retrieve, OpenRouter LLM answer."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# #region agent log
_DEBUG_LOG = Path(__file__).resolve().parents[3] / ".cursor" / "debug-c95187.log"
def _dbg(hypothesis_id: str, location: str, message: str, data: dict | None = None, run_id: str = "pre-fix") -> None:
    try:
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG.open("a", encoding="utf-8") as _fh:
            _fh.write(json.dumps({"sessionId": "c95187", "runId": run_id, "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data or {}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass
_import_t0 = time.perf_counter()
_dbg("C", "pipeline.py:before_pypdf", "about to import pypdf")
# #endregion

from pypdf import PdfReader

# #region agent log
_dbg("C", "pipeline.py:after_pypdf", "pypdf imported", {"elapsed_ms": round((time.perf_counter() - _import_t0) * 1000, 1)})
# #endregion

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
ABSTAIN = "I couldn't generate an answer."

SYSTEM_PROMPT = """You are a helpful technical Q&A assistant.

Your own knowledge is the primary source for answering. Retrieved document passages are optional helper context — an addon, not the main constraint.

Rules:
- Keep the core answer short: a few concise sentences (or a short bullet list) from your own knowledge.
- If helper passages are present and relevant, you may add a brief extra section that pulls in useful details from them. Do not dump large passages verbatim.
- If the passages are incomplete, off-topic, or missing, still give the short answer from your own knowledge. Do not refuse just because retrieval was weak.
- Prefer passage details for project-specific names, claims, and facts when they clearly apply.
- Prefer brevity over completeness. Avoid long essays unless the user explicitly asks for depth.
"""


@dataclass
class SourceChunk:
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Index:
    """Local Chroma-backed index that can be reused across questions."""

    chunks: list[SourceChunk]
    collection: object
    chunk_by_id: dict[str, SourceChunk]
    partial: bool = False
    source_paths: list[str] = field(default_factory=list)
    collection_name: str = ""


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def load_file(path: Path, max_chars: int) -> tuple[str, dict[str, object]]:
    """Load a local .txt/.md/.pdf file up to max_chars. Returns (text, coverage meta)."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported file type '{suffix}' (supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")

    if suffix in {".txt", ".md"}:
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"could not decode {path.name} as UTF-8") from exc
        text = raw[:max_chars]
        meta = {
            "name": path.name,
            "kind": "text_file",
            "partial": len(raw) > max_chars,
            "chars_processed": len(text),
            "path": str(path),
        }
        return text, meta

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    page_limit = min(total_pages, _env_int("RAG_MAX_PDF_PAGES", 120))
    if page_limit <= 0:
        return "", {"name": path.name, "kind": "pdf", "partial": False, "chars_processed": 0, "path": str(path)}
    if page_limit == 1:
        indexes = [0]
    else:
        indexes = [round(i * (total_pages - 1) / (page_limit - 1)) for i in range(page_limit)]

    parts: list[str] = []
    used = 0
    for page_index in indexes:
        if used >= max_chars:
            break
        page_text = (reader.pages[page_index].extract_text() or "")[: max_chars - used]
        if page_text.strip():
            parts.append(page_text)
            used += len(page_text)
    text = "\n\n".join(parts)
    meta = {
        "name": path.name,
        "kind": "pdf",
        "partial": len(indexes) < total_pages or used >= max_chars,
        "pages_processed": min(len(indexes), total_pages),
        "pages_total": total_pages,
        "chars_processed": used,
        "path": str(path),
    }
    return text, meta


def load_sources(paths: list[Path], max_source_chars: int | None = None) -> tuple[list[SourceChunk], bool]:
    """Load and chunk multiple local files under a shared character budget."""
    budget = max_source_chars if max_source_chars is not None else _env_int("RAG_MAX_SOURCE_CHARS", 150_000)
    chunks: list[SourceChunk] = []
    partial = False
    remaining = budget

    for path in paths:
        if remaining <= 0:
            partial = True
            LOGGER.info("loader: character budget exhausted; skipping remaining files")
            break
        text, meta = load_file(path, remaining)
        if not text.strip():
            raise ValueError(f"no readable text in {path}")
        remaining -= meta["chars_processed"]
        partial = partial or bool(meta.get("partial"))
        file_chunks = chunk_text(
            text,
            {
                "source": str(path.expanduser().resolve()),
                "filename": path.name,
                "path": str(path.expanduser().resolve()),
            },
        )
        if not file_chunks:
            raise ValueError(f"no readable text in {path}")
        chunks.extend(file_chunks)
        LOGGER.info("loader: %s -> %d chunk(s), %d chars", path.name, len(file_chunks), meta["chars_processed"])

    if not chunks:
        raise ValueError("no readable source content was found")
    return chunks, partial


def chunk_text(text: str, metadata: dict[str, object]) -> list[SourceChunk]:
    size = _env_int("RAG_CHUNK_SIZE", 1000)
    overlap = min(_env_int("RAG_CHUNK_OVERLAP", 200), size // 2)
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


def _chroma_metadata(metadata: dict[str, object]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif value is None:
            continue
        else:
            cleaned[key] = str(value)
    return cleaned


def _chroma_client():
    import chromadb

    path = os.getenv("CHROMA_PATH", str(Path(__file__).resolve().parents[2] / ".chroma"))
    Path(path).mkdir(parents=True, exist_ok=True)
    # #region agent log
    _dbg(
        "CHROMA",
        "pipeline.py:_chroma_client",
        "creating PersistentClient",
        {"path": path},
        run_id="post-fix",
    )
    # #endregion
    return chromadb.PersistentClient(path=path)


def build_index(chunks: list[SourceChunk], partial: bool = False, source_paths: list[str] | None = None) -> Index:
    unique = deduplicate(chunks)
    if not unique:
        raise ValueError("no readable source content was found")

    for index, chunk in enumerate(unique):
        source = str(chunk.metadata.get("path") or chunk.metadata.get("source") or "")
        chunk.metadata = {
            **chunk.metadata,
            "chunk_id": hashlib.sha256(f"{source}:{index}:{chunk.text}".encode("utf-8")).hexdigest(),
        }

    client = _chroma_client()
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    embedding_fn = DefaultEmbeddingFunction()
    collection_name = f"rag-{uuid.uuid4().hex[:12]}"
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(chunk.metadata["chunk_id"]) for chunk in unique]
    documents = [chunk.text for chunk in unique]
    metadatas = [_chroma_metadata(chunk.metadata) for chunk in unique]
    batch_size = _env_int("CHROMA_UPSERT_BATCH", 100)
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    # #region agent log
    _dbg(
        "CHROMA",
        "pipeline.py:build_index",
        "upserted to local Chroma",
        {"collection": collection_name, "chunks": len(unique)},
        run_id="post-fix",
    )
    # #endregion
    LOGGER.info("index: upserted %d chunk(s) to local Chroma collection=%s", len(unique), collection_name)
    chunk_by_id = {str(chunk.metadata["chunk_id"]): chunk for chunk in unique}
    return Index(
        chunks=unique,
        collection=collection,
        chunk_by_id=chunk_by_id,
        partial=partial,
        source_paths=source_paths or [],
        collection_name=collection_name,
    )


def retrieve(index: Index, query: str, k: int = 6) -> list[SourceChunk]:
    if not index.chunks:
        # #region agent log
        _dbg("A", "pipeline.py:retrieve", "empty index", {"query": query})
        # #endregion
        return []
    n_results = min(k, len(index.chunks))
    result = index.collection.query(query_texts=[query], n_results=n_results)
    ids = (result.get("ids") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]

    retrieved: list[SourceChunk] = []
    for chunk_id, distance, document, metadata in zip(ids, distances, documents, metadatas):
        existing = index.chunk_by_id.get(str(chunk_id))
        if existing is not None:
            retrieved.append(existing)
            continue
        retrieved.append(
            SourceChunk(
                text=document or "",
                metadata={**(metadata or {}), "chunk_id": str(chunk_id), "distance": distance},
            )
        )

    # #region agent log
    _dbg(
        "A",
        "pipeline.py:retrieve",
        "chroma retrieval",
        {
            "query": query,
            "chunk_count": len(index.chunks),
            "returned": len(retrieved),
            "top_distance": float(distances[0]) if distances else None,
            "collection": index.collection_name,
        },
        run_id="post-fix",
    )
    # #endregion
    return retrieved


def build_context(documents: list[SourceChunk]) -> tuple[str, set[str]]:
    """Build labeled source context for the LLM prompt."""
    max_doc_chars = _env_int("RAG_CONTEXT_DOC_CHARS", 1400)
    max_total_chars = _env_int("RAG_CONTEXT_TOTAL_CHARS", 9000)
    current_total = 0
    sections: list[str] = []
    source_ids: set[str] = set()
    missing_ids = 0
    for document in documents:
        chunk_id = str(document.metadata.get("chunk_id", ""))
        if not chunk_id:
            missing_ids += 1
            continue
        label = document.metadata.get("filename") or document.metadata.get("source") or "source"
        entry = f"[Source {chunk_id}: {label}]\n{document.text[:max_doc_chars]}"
        if current_total + len(entry) > max_total_chars:
            break
        sections.append(entry)
        source_ids.add(chunk_id)
        current_total += len(entry)
    context = "\n\n".join(sections)
    # #region agent log
    _dbg(
        "B",
        "pipeline.py:build_context",
        "context built",
        {
            "input_docs": len(documents),
            "missing_ids": missing_ids,
            "source_count": len(source_ids),
            "context_chars": len(context),
        },
        run_id="post-fix",
    )
    # #endregion
    LOGGER.info("context: %d source(s), %d character(s)", len(source_ids), len(context))
    return context, source_ids


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


def generate_answer(
    question: str,
    context: str,
    llm=None,
    on_token=None,
) -> str:
    """Stream/generate an answer via LangChain ChatOpenAI over OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if llm is None and not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export your OpenRouter API key before asking questions."
        )

    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    # #region agent log
    _llm_t0 = time.perf_counter()
    _dbg(
        "LLM",
        "pipeline.py:generate_answer",
        "calling LangChain ChatOpenAI stream",
        {
            "question_len": len(question),
            "context_len": len(context),
            "model": model,
            "streaming": on_token is not None,
        },
        run_id="post-fix",
    )
    # #endregion

    if llm is None:
        from langchain_openai import ChatOpenAI

        headers: dict[str, str] = {}
        title = os.getenv("OPENROUTER_TITLE", "agentspace-rag")
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if title:
            headers["X-Title"] = title
        if referer:
            headers["HTTP-Referer"] = referer
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0,
            streaming=True,
            default_headers=headers or None,
        )

    from langchain_core.messages import HumanMessage, SystemMessage

    helper = context.strip() if context.strip() else "(no helper passages retrieved)"
    user_prompt = (
        "OPTIONAL HELPER PASSAGES from the user's files (addon context, not the only source):\n"
        f"{helper}\n\n"
        f"QUESTION: {question}\n\n"
        "Give a short answer from your own knowledge first. "
        "Only if the helper passages clearly add useful project-specific detail, append a brief "
        "context-backed note — keep that short too, even if the passages are long."
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    if on_token is not None:
        parts: list[str] = []
        for chunk in llm.stream(messages):
            piece = chunk.content if isinstance(chunk.content, str) else _message_text(chunk.content)
            if piece:
                on_token(piece)
                parts.append(piece)
        text = "".join(parts).strip()
    else:
        response = llm.invoke(messages)
        text = (response.content if isinstance(response.content, str) else _message_text(response.content)).strip()

    # #region agent log
    _dbg(
        "LLM",
        "pipeline.py:generate_answer_done",
        "LangChain generation finished",
        {"elapsed_ms": round((time.perf_counter() - _llm_t0) * 1000, 1), "answer_len": len(text)},
        run_id="post-fix",
    )
    # #endregion
    LOGGER.info("llm: generated answer (%d chars) via LangChain/OpenRouter model=%s", len(text), model)
    return text or ABSTAIN


def ask(
    index: Index,
    question: str,
    k: int = 6,
    llm=None,
    on_token=None,
) -> dict[str, object]:
    retrieved = retrieve(index, question, k=k)
    context, source_ids = build_context(retrieved)
    # #region agent log
    _dbg(
        "D",
        "pipeline.py:ask",
        "before llm decision",
        {
            "retrieved": len(retrieved),
            "context_chars": len(context),
            "will_call_llm": bool(context),
            "streaming": on_token is not None,
        },
        run_id="post-fix",
    )
    # #endregion
    result_answer = generate_answer(question, context, llm=llm, on_token=on_token)
    return {
        "question": question,
        "answer": result_answer,
        "documents_count": len(index.chunks),
        "partial": index.partial,
        "source_paths": index.source_paths,
        "source_ids": sorted(source_ids),
        "retrieved_chunks": [
            {
                "id": chunk.metadata["chunk_id"],
                "filename": chunk.metadata.get("filename") or Path(str(chunk.metadata.get("source", ""))).name,
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
