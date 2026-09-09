"""HTTP API for the local RAG pipeline."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import uuid
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag.pipeline import SUPPORTED_EXTENSIONS, ask, build_from_paths
from app.rag.models import Index

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if not raw:
        return list(_DEFAULT_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(title="AgentSpace RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("uvicorn.error")
_sessions: dict[str, Index] = {}


class AskBody(BaseModel):
    session_id: str
    question: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=24)


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness/readiness probe for local runs and Docker healthchecks."""
    openrouter = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    chroma = bool(os.getenv("CHROMA_API_KEY", "").strip())
    ready = openrouter and chroma
    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "openrouter_configured": openrouter,
        "chroma_configured": chroma,
        "sessions": len(_sessions),
    }


@app.post("/api/session")
async def create_session(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")
    if not os.getenv("CHROMA_API_KEY"):
        raise HTTPException(status_code=500, detail="CHROMA_API_KEY is not set")

    dest_dir = Path(mkdtemp(prefix="agentspace-"))
    saved: list[Path] = []
    for upload in files:
        name = Path(upload.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"unsupported file type '{suffix}' for {name} (supported: {supported})",
            )
        path = dest_dir / name
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"empty file: {name}")
        path.write_bytes(content)
        saved.append(path)

    try:
        index = build_from_paths(saved)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id = uuid.uuid4().hex
    _sessions[session_id] = index
    return {
        "session_id": session_id,
        "chunk_count": len(index.chunks),
        "source_count": len(index.source_paths),
        "partial": index.partial,
    }


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/ask")
def ask_question(body: AskBody) -> StreamingResponse:
    index = _sessions.get(body.session_id)
    if index is None:
        raise HTTPException(status_code=404, detail="session not found; attach files first")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

    events: queue.Queue[tuple[str, Any]] = queue.Queue()

    def on_token(token: str) -> None:
        events.put(("token", token))

    def run() -> None:
        try:
            result = ask(index, body.question.strip(), k=body.top_k, on_token=on_token)
            events.put(("done", result))
        except Exception as exc:  # noqa: BLE001 — surface any LLM/RAG failure to the client
            log.exception("ask failed")
            events.put(("error", str(exc)))

    def stream():
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        while True:
            kind, payload = events.get()
            if kind == "token":
                yield _sse({"type": "token", "text": payload})
            elif kind == "done":
                yield _sse(
                    {
                        "type": "done",
                        "answer": payload.get("answer", ""),
                        "retrieved_chunks": payload.get("retrieved_chunks", []),
                        "partial": payload.get("partial", False),
                        "source_ids": payload.get("source_ids", []),
                        "rag_score": payload.get("rag_score"),
                        "use_rag": payload.get("use_rag"),
                        "rag_reason": payload.get("rag_reason"),
                    }
                )
                break
            else:
                yield _sse({"type": "error", "message": str(payload)})
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)
