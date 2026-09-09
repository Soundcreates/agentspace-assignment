# Tiny Local RAG Q&A

Ask questions about **your** local `.txt`, `.md`, or `.pdf` files.
Files are chunked locally, stored in a **local Chroma** persistent database, then answered by an **OpenRouter** model through a **LangGraph** ask workflow (LangChain) with **token streaming**.

## Layout

```text
python/
  app/main.py                # CLI
  app/server.py              # FastAPI: /api/session, /api/ask (SSE), /health
  app/rag/
    pipeline.py              # build_from_paths + ask() wrapper
    graph.py                 # LangGraph ask workflow (judge → retrieve|skip → answer)
    models.py / context.py / util.py
    loaders/local.py
    chunkers/text.py
    vectorstores/chroma.py   # local Chroma PersistentClient
    query/enhance.py
    llm/generate.py
    llm/judge.py
  Dockerfile
  docker-compose.yml
  .env.example
  tests/test_rag_pipeline.py
  requirements.txt
frontend/                    # React + Tailwind + GSAP prompt page
  .env.example               # VITE_API_BASE_URL
```

## Install

```bash
cd python
python3 -m pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=sk-or-...

cd ../frontend
cp .env.example .env
# VITE_API_BASE_URL=http://localhost:8000
```

Requires Python 3.10+. OpenRouter key: https://openrouter.ai/keys

Local Chroma path: `CHROMA_PATH` (default `.chroma`).

## Docker

From `python/`:

```bash
# ensure .env has OPENROUTER_API_KEY
docker compose up --build
```

API listens on http://localhost:8000 (`GET /health`). Chroma data persists in the `chroma_data` volume.

Or without Compose:

```bash
docker build -t agentspace-api .
docker run --rm -p 8000:8000 --env-file .env -v agentspace-chroma:/app/.chroma agentspace-api
```

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
# 1. API (local) — or use Docker Compose above
cd python
python -m app.server
```

```bash
# 2. Frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Attach `.txt` / `.md` / `.pdf` files, then ask from the prompt bar.

Frontend talks to the API via `VITE_API_BASE_URL` (default `http://localhost:8000`).

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
2. Chunk text and upsert embeddings into a **local Chroma** collection (`chromadb.PersistentClient`, embeddings via `all-MiniLM-L6-v2` / ONNX).
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
