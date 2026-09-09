"""Ask questions about local text, Markdown, or PDF files."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
import time
from pathlib import Path

# Support both `python -m app.main` and `python app/main.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from app.rag.pipeline import SUPPORTED_EXTENSIONS, build_from_paths, ask



def parse_source_paths(raw: str) -> list[Path]:
    """Parse shell-escaped paths (including terminal drag-and-drop paste)."""
    try:
        tokens = shlex.split(raw.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"could not parse paths: {exc}") from exc
    if not tokens:
        raise argparse.ArgumentTypeError("at least one source path is required")
    return [Path(token) for token in tokens]


def resolve_sources(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve()
        if not candidate.is_file():
            raise SystemExit(f"error: file not found: {path}")
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise SystemExit(f"error: unsupported file type '{candidate.suffix}' for {path} (supported: {supported})")
        resolved.append(candidate)
    return resolved


def prompt_sources() -> list[Path]:
    print("Drop file(s) here or enter paths (.txt, .md, .pdf):", flush=True)
    try:
        raw = input().strip()
    except EOFError:
        raise SystemExit("error: no source paths provided") from None
    if not raw:
        raise SystemExit("error: at least one source path is required")
    return resolve_sources(parse_source_paths(raw))


def print_chunks(result: dict[str, object]) -> None:
    print("\nRetrieved chunks:")
    for chunk in result["retrieved_chunks"]:
        print(f"\n[{chunk['id']}] {chunk['filename']} (rank={chunk['rank']})\n{chunk['text']}")


def _stream_chars(text: str, delay_s: float) -> None:
    """Print text one character at a time for a natural typing feel."""
    for char in text:
        print(char, end="", flush=True)
        if delay_s > 0:
            time.sleep(delay_s)


def stream_answer(index, question: str, top_k: int) -> dict[str, object]:
    delay_s = max(0.0, float(os.getenv("STREAM_CHAR_DELAY_MS", "12")) / 1000.0)
    print("\nAnswer: ", end="", flush=True)
    streamed = False

    def on_token(token: str) -> None:
        nonlocal streamed
        streamed = True
        _stream_chars(token, delay_s)

    result = ask(index, question, k=top_k, on_token=on_token)
    if not streamed:
        # Abstain / empty-context path never called on_token.
        _stream_chars(str(result["answer"]), delay_s)
    print(flush=True)
    print_chunks(result)
    return result


def main() -> None:
    from dotenv import load_dotenv

    # Load python/.env when present (same directory as requirements.txt).
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(
        description="Ask questions about local .txt, .md, or .pdf files (drag files into the terminal to paste paths).",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="local source file path(s); omit to paste/drag-and-drop at the prompt",
    )
    parser.add_argument("--question", "-q", help="one-shot question; omit for an interactive question loop")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON (one-shot mode)")
    parser.add_argument("--top-k", type=int, default=6, help="number of chunks to retrieve (default: 6)")
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=150_000,
        help="total character budget across all sources (default: 150000)",
    )
    args = parser.parse_args()

    if args.json and not args.question:
        parser.error("--json requires --question")

    paths = resolve_sources(args.sources) if args.sources else prompt_sources()

    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("error: OPENROUTER_API_KEY is not set (required for LLM answers)")

    print(f"Loading {len(paths)} source(s)...", flush=True)
    try:
        index = build_from_paths(paths, max_source_chars=args.max_source_chars)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(
        f"Indexed {len(index.chunks)} chunk(s) from {len(index.source_paths)} file(s) "
        f"into local Chroma ({index.collection_name}).",
        flush=True,
    )
    if index.partial:
        print("Note: sources were partially indexed due to the character budget.", flush=True)

    if args.question:
        question = args.question.strip()
        if not question:
            parser.error("a question is required")
        try:
            if args.json:
                result = ask(index, question, k=args.top_k)
                print(json.dumps(result, indent=2))
                return
            stream_answer(index, question, args.top_k)
        except Exception as exc:
            raise SystemExit(f"error: {exc}") from exc
        return

    print("Ask questions about your sources (empty line or Ctrl-D to exit).", flush=True)
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except EOFError:
            print()
            break
        if not question:
            break
        try:
            stream_answer(index, question, args.top_k)
        except Exception as exc:
            print(f"error: {exc}", flush=True)
            continue


if __name__ == "__main__":
    main()