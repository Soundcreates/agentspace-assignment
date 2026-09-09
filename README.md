# Tiny Local RAG Q&A

Ask questions about **your** local `.txt`, `.md`, or `.pdf` files.
Files are chunked locally, stored in a **local Chroma** persistent database, then answered by an **OpenRouter** model through a **LangGraph** ask workflow (LangChain) with **token streaming**.

## Layout

```text
python/
  app/main.py                # CLI
  app/server.py              # FastAPI: /api/session, /api/ask (SSE), /api/health
  app/rag/
    pipeline.py              # build_from_paths + ask() wrapper
    graph.py                 # LangGraph ask workflow (judge → retrieve|skip → answer)
    models.py / context.py / util.py
    loaders/local.py
    chunkers/text.py
    vectorstores/chroma.py
    query/enhance.py         # rewrite + split + multi-query retrieve helpers
    llm/generate.py
    llm/judge.py             # RAG-need score judge
  .chroma/
  .env.example
  tests/test_rag_pipeline.py
  requirements.txt
frontend/                    # React + Tailwind + GSAP prompt page
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

### Web UI (prompt page)

Two terminals:

```bash
# 1. API (use the venv interpreter — a bare `uvicorn` may hit a broken system install)
cd python
python -m app.server
```

```bash
# 2. Frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Attach `.txt` / `.md` / `.pdf` files, then ask from the prompt bar. The Vite dev server proxies `/api` to port 8000.

Optional: set `VITE_API_BASE_URL` in `frontend/.env` if the API is not on the same origin / proxy (default empty so the proxy is used).

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
3. Run the **LangGraph** ask graph:
   - `judge` — score whether RAG help is needed (0–100); continue to retrieve only when score **> 60**
   - `retrieve` or `skip_retrieve` — enhance queries + multi-query Chroma retrieval, or empty context
   - `answer` — stream with LangChain `ChatOpenAI` over OpenRouter
4. The model answers mainly from its own knowledge and uses retrieved passages as optional helper context when RAG ran.

## Tests

```bash
cd python
python3 -m unittest discover -s tests -p 'test_*.py'
```
