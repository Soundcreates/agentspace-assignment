# Tiny Local RAG Q&A

Ask questions about **your** local `.txt`, `.md`, or `.pdf` files.
Files are chunked locally, stored in a **local Chroma** persistent database, then answered by an **OpenRouter** model through **LangChain** with **token streaming**.

## Layout

```text
python/
  app/main.py              # CLI entry point (streams answer tokens)
  app/rag/pipeline.py      # load → chunk → local Chroma retrieve → LangChain stream
  .chroma/                 # local Chroma data (created at runtime)
  .env.example
  tests/test_rag_pipeline.py
  requirements.txt
```

## Install

```bash
cd python
python3 -m pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=sk-or-...
```

Requires Python 3.10+. OpenRouter key: https://openrouter.ai/keys

Optional: `CHROMA_PATH` to choose where Chroma persists (default `python/.chroma`).

## Usage

### Interactive (drag-and-drop)

```bash
cd python
python3 -m app.main
```

When prompted, drag one or more files into the terminal, then ask questions. Answers stream **character by character** (paced from LangChain token chunks; tune with `STREAM_CHAR_DELAY_MS`, default `12`).

### One-shot

```bash
python3 -m app.main notes.txt --question "What is this project about?"
```

### JSON (non-streaming, full payload)

```bash
python3 -m app.main notes.txt --question "What is this project about?" --json
```

## How it works

1. Load local files under a character budget.
2. Chunk text and upsert embeddings into a **local Chroma** collection (`chromadb.PersistentClient`).
3. Query Chroma for top-k passages.
4. Stream an answer with LangChain `ChatOpenAI` pointed at OpenRouter (`llm.stream(...)`).
5. The model answers mainly from its own knowledge and uses retrieved passages as optional helper context.

## Tests

```bash
cd python
python3 -m unittest discover -s tests -p 'test_*.py'
```
