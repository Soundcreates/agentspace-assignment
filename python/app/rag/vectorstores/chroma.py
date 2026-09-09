from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from app.rag.deduplicators import deduplicate
from app.rag.models import Index, SourceChunk
from app.rag.util import env_int


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

    path = os.getenv("CHROMA_PATH", str(Path(__file__).resolve().parents[3] / ".chroma"))
    Path(path).mkdir(parents=True, exist_ok=True)
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
    batch_size = env_int("CHROMA_UPSERT_BATCH", 100)
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
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
            chunk = SourceChunk(existing.text, {**existing.metadata, "distance": distance})
            retrieved.append(chunk)
            continue
        retrieved.append(
            SourceChunk(
                text=document or "",
                metadata={**(metadata or {}), "chunk_id": str(chunk_id), "distance": distance},
            )
        )
    return retrieved


def retrieve_multi(index: Index, queries: list[str], k: int = 6) -> list[SourceChunk]:
    if not queries:
        return []
    per_query = max(1, (k + len(queries) - 1) // len(queries))
    best: dict[str, tuple[float, SourceChunk]] = {}
    for query in queries:
        for chunk in retrieve(index, query, k=per_query):
            chunk_id = str(chunk.metadata.get("chunk_id", ""))
            if not chunk_id:
                continue
            distance = chunk.metadata.get("distance")
            score = float(distance) if isinstance(distance, (int, float)) else 1.0
            previous = best.get(chunk_id)
            if previous is None or score < previous[0]:
                best[chunk_id] = (score, chunk)
    ranked = sorted(best.values(), key=lambda item: item[0])
    return [chunk for _, chunk in ranked[:k]]
