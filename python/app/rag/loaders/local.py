from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.rag.chunkers.text import chunk_text
from app.rag.models import SourceChunk
from app.rag.util import env_int

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_file(path: Path, max_chars: int) -> tuple[str, dict[str, object]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported file type '{suffix}' (supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

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
    page_limit = min(total_pages, env_int("RAG_MAX_PDF_PAGES", 120))
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
    budget = max_source_chars if max_source_chars is not None else env_int("RAG_MAX_SOURCE_CHARS", 150_000)
    chunks: list[SourceChunk] = []
    partial = False
    remaining = budget

    for path in paths:
        if remaining <= 0:
            partial = True
            break
        text, meta = load_file(path, remaining)
        if not text.strip():
            raise ValueError(f"no readable text in {path}")
        remaining -= int(meta["chars_processed"])
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

    if not chunks:
        raise ValueError("no readable source content was found")
    return chunks, partial
