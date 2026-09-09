# Tiny RAG Q&A

This is a small retrieval-augmented Q&A demo over [RFC 9110 — HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.html).

It uses only Python's standard library:

1. Downloads and caches the public RFC text.
2. Removes RFC page noise, groups text by section, and chunks it with overlap.
3. Builds sparse TF–IDF vectors as local embeddings and retrieves the top six chunks with cosine similarity.
4. Extracts an answer from the retrieved evidence, or abstains when the question is not sufficiently supported.

Run it with Python 3.10+:

```bash
python3 rag_qa.py
```

For machine-readable answers and the exact evidence returned by retrieval:

```bash
python3 rag_qa.py --json > answers.json
```

The built-in questions are deliberately shaped as requested:

- Q1: HTTPS's default port (factual).
- Q2: safe methods plus idempotency/retry behavior (combines two sections).
- Q3: a summary of Section 9.2.2.
- Q4: HTTP/2 vs HTTP/3 performance over 5G, which RFC 9110 does not answer and should trigger abstention.

No API/UI wrapper is included; the JSON CLI already exposes the core result cleanly.
# agentspace-assignment
