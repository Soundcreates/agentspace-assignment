"""Tests for local dynamic RAG. Run: python3 -m unittest discover -s tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.main import parse_source_paths
from app.rag.pipeline import (
    Index,
    SourceChunk,
    ask,
    build_context,
    chunk_text,
    enhance_queries,
    generate_answer,
    load_file,
    load_sources,
    retrieve,
    retrieve_multi,
    rewrite_query,
    split_query,
)


def _fake_index(chunks: list[SourceChunk]) -> Index:
    for index, chunk in enumerate(chunks):
        chunk.metadata = {
            **chunk.metadata,
            "chunk_id": chunk.metadata.get("chunk_id") or f"chunk-{index}",
        }
    chunk_by_id = {str(chunk.metadata["chunk_id"]): chunk for chunk in chunks}
    collection = Mock()
    collection.query.return_value = {
        "ids": [[str(chunk.metadata["chunk_id"]) for chunk in chunks[:2]]],
        "distances": [[0.1, 0.2][: len(chunks[:2])]],
        "documents": [[chunk.text for chunk in chunks[:2]]],
        "metadatas": [[dict(chunk.metadata) for chunk in chunks[:2]]],
    }
    return Index(
        chunks=chunks,
        collection=collection,
        chunk_by_id=chunk_by_id,
        source_paths=["inline"],
        collection_name="rag-test",
    )


class PathParsingTests(unittest.TestCase):
    def test_parse_shell_escaped_drag_drop_paths(self):
        paths = parse_source_paths("/tmp/my\\ notes.txt '/Users/me/Docs/Report Final.pdf'")
        self.assertEqual(
            [str(path) for path in paths],
            ["/tmp/my notes.txt", "/Users/me/Docs/Report Final.pdf"],
        )


class LoaderTests(unittest.TestCase):
    def test_load_text_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt = root / "notes.txt"
            md = root / "guide.md"
            txt.write_text("Alpha protocol uses port 443 for secure traffic.", encoding="utf-8")
            md.write_text("# Guide\n\nSafe methods are read-only requests.", encoding="utf-8")

            text, meta = load_file(txt, max_chars=10_000)
            self.assertIn("port 443", text)
            self.assertEqual(meta["kind"], "text_file")
            self.assertFalse(meta["partial"])

            chunks, partial = load_sources([txt, md], max_source_chars=10_000)
            self.assertFalse(partial)
            self.assertGreaterEqual(len(chunks), 2)
            filenames = {chunk.metadata["filename"] for chunk in chunks}
            self.assertEqual(filenames, {"notes.txt", "guide.md"})

    def test_load_pdf(self):
        from pypdf import PdfWriter

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "sample.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            with pdf_path.open("wb") as handle:
                writer.write(handle)

            with patch("app.rag.loaders.local.PdfReader") as reader_cls:
                page = Mock()
                page.extract_text.return_value = "Idempotent methods may be retried after failure."
                reader_cls.return_value.pages = [page]
                text, meta = load_file(pdf_path, max_chars=10_000)

            self.assertIn("Idempotent", text)
            self.assertEqual(meta["kind"], "pdf")

    def test_reject_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(b"\x89PNG")
            with self.assertRaises(ValueError) as ctx:
                load_file(path, max_chars=100)
            self.assertIn("unsupported", str(ctx.exception).lower())

    def test_character_budget_marks_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.txt"
            second = root / "b.txt"
            first.write_text("A" * 50, encoding="utf-8")
            second.write_text("B" * 50, encoding="utf-8")
            chunks, partial = load_sources([first, second], max_source_chars=40)
            self.assertTrue(partial)
            self.assertTrue(chunks)
            self.assertTrue(all(chunk.metadata["filename"] == "a.txt" for chunk in chunks))


class PipelineTests(unittest.TestCase):
    def test_retrieve_context_and_llm_answer(self):
        text = (
            "GET and HEAD are safe methods because they are essentially read-only. "
            "GET and HEAD are idempotent. A client can retry an idempotent request after a communication failure."
        )
        chunks = chunk_text(text, {"source": "inline", "filename": "demo.txt", "path": "inline"})
        index = _fake_index(chunks)
        question = "Which methods are safe and idempotent, and why can they be retried?"
        retrieved = retrieve(index, question)
        context, source_ids = build_context(retrieved)
        self.assertTrue(context)
        self.assertTrue(source_ids)

        llm = Mock()
        llm.invoke.return_value = Mock(content="GET and HEAD are safe and idempotent; they can be retried.")
        result = generate_answer(question, context, llm=llm)
        self.assertIn("GET", result)
        messages = llm.invoke.call_args.args[0]
        self.assertEqual(messages[0].type, "system")
        self.assertIn("OPTIONAL HELPER PASSAGES", messages[1].content)
        self.assertIn(question, messages[1].content)

        empty_llm = Mock()
        empty_llm.invoke.return_value = Mock(content="Answer from general knowledge.")
        self.assertEqual(generate_answer(question, "", llm=empty_llm), "Answer from general knowledge.")
        self.assertIn("no helper passages", empty_llm.invoke.call_args.args[0][1].content)

    def test_stream_tokens(self):
        llm = Mock()
        llm.stream.return_value = [Mock(content="Hello "), Mock(content="world")]
        tokens: list[str] = []
        result = generate_answer("What?", "Some context about hello world.", llm=llm, on_token=tokens.append)
        self.assertEqual(result, "Hello world")
        self.assertEqual(tokens, ["Hello ", "world"])
        llm.stream.assert_called_once()

    def test_ask_returns_evidence_metadata(self):
        chunks = chunk_text(
            "GET and PUT can be repeated after a communication failure. "
            "Safe methods do not change server state.",
            {"source": "protocol.txt", "filename": "protocol.txt", "path": "protocol.txt"},
        )
        index = _fake_index(chunks)
        judge = Mock()
        judge.invoke.return_value = Mock(content='{"score": 82, "reason": "doc-specific"}')
        llm = Mock()
        llm.invoke.return_value = Mock(content="GET and PUT can be repeated after failure.")
        result = ask(index, "Which methods can be repeated?", llm=llm, judge_llm=judge)
        self.assertIn("GET", result["answer"])
        self.assertTrue(result["use_rag"])
        self.assertGreater(result["rag_score"], 60)
        self.assertTrue(result["source_ids"])
        self.assertEqual(result["retrieved_chunks"][0]["filename"], "protocol.txt")
        self.assertEqual(result["documents_count"], len(index.chunks))
        judge.invoke.assert_called_once()

    def test_ask_skips_retrieval_when_judge_score_low(self):
        chunks = chunk_text(
            "GET and PUT can be repeated after a communication failure.",
            {"source": "protocol.txt", "filename": "protocol.txt", "path": "protocol.txt"},
        )
        index = _fake_index(chunks)
        judge = Mock()
        judge.invoke.return_value = Mock(content='{"score": 25, "reason": "general knowledge"}')
        llm = Mock()
        llm.invoke.return_value = Mock(content="Paris is the capital of France.")
        result = ask(index, "What is the capital of France?", llm=llm, judge_llm=judge)
        self.assertFalse(result["use_rag"])
        self.assertEqual(result["rag_score"], 25)
        self.assertEqual(result["queries"], [])
        self.assertEqual(result["retrieved_chunks"], [])
        self.assertEqual(result["source_ids"], [])
        index.collection.query.assert_not_called()
        self.assertIn("Paris", result["answer"])

    def test_index_reuse_across_questions(self):
        chunks = [
            SourceChunk(
                "Alpha service listens on port 8080 by default.",
                {"source": "a.txt", "filename": "a.txt", "path": "a.txt"},
            )
        ]
        index = _fake_index(chunks)
        collection_id = id(index.collection)
        judge = Mock()
        judge.invoke.return_value = Mock(content='{"score": 75, "reason": "project fact"}')
        llm = Mock()
        llm.invoke.return_value = Mock(content="Alpha service uses port 8080.")
        first = ask(index, "What port does Alpha service use?", llm=llm, judge_llm=judge)
        second = ask(index, "What is Alpha service default port?", llm=llm, judge_llm=judge)
        self.assertIn("8080", first["answer"])
        self.assertIn("8080", second["answer"])
        self.assertEqual(first["documents_count"], second["documents_count"])
        self.assertEqual(id(index.collection), collection_id)
        self.assertEqual(llm.invoke.call_count, 2)
        self.assertEqual(judge.invoke.call_count, 2)
        self.assertGreaterEqual(index.collection.query.call_count, 2)
        self.assertTrue(first["queries"])
        self.assertTrue(second["queries"])


class JudgeTests(unittest.TestCase):
    def test_parse_score_and_threshold(self):
        from app.rag.llm.judge import judge_rag_need

        llm = Mock()
        llm.invoke.return_value = Mock(content='{"score": 61, "reason": "borderline high"}')
        high = judge_rag_need("What does section 3 say?", source_names=["notes.txt"], llm=llm)
        self.assertEqual(high["score"], 61)
        self.assertTrue(high["use_rag"])

        llm.invoke.return_value = Mock(content='{"score": 60, "reason": "at threshold"}')
        edge = judge_rag_need("Hello there", source_names=["notes.txt"], llm=llm)
        self.assertEqual(edge["score"], 60)
        self.assertFalse(edge["use_rag"])

    def test_heuristic_without_api(self):
        from app.rag.llm.judge import judge_rag_need

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False):
            # Ensure key absence path: pass llm=None and empty key
            env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
            with patch.dict("os.environ", env, clear=True):
                result = judge_rag_need(
                    "According to this document, what is the timeout?",
                    source_names=["guide.md"],
                    llm=None,
                )
        self.assertTrue(result["use_rag"])
        self.assertGreater(result["score"], 60)


class QueryEnhanceTests(unittest.TestCase):
    def test_split_and_rewrite_multipart_question(self):
        question = "What is custody-role disambiguation and how does fail-closed policy validation work?"
        parts = split_query(question)
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(any("custody" in p.lower() for p in parts))
        self.assertTrue(any("fail-closed" in p.lower() or "policy" in p.lower() for p in parts))

        rewritten = rewrite_query(question, llm=None)
        self.assertTrue(rewritten)
        self.assertNotIn("?", rewritten)

        enhanced = enhance_queries(question, llm=None)
        self.assertGreaterEqual(len(enhanced), 2)
        self.assertEqual(enhanced[0], " ".join(question.split()))

    def test_retrieve_multi_merges_unique_chunks(self):
        chunks = [
            SourceChunk("Section A about custody roles.", {"chunk_id": "a", "filename": "a.txt"}),
            SourceChunk("Section B about fail-closed policy.", {"chunk_id": "b", "filename": "b.txt"}),
        ]
        index = _fake_index(chunks)

        def _query_side_effect(query_texts, n_results):
            q = query_texts[0].lower()
            if "custody" in q:
                chosen = [chunks[0]]
            elif "fail" in q or "policy" in q:
                chosen = [chunks[1]]
            else:
                chosen = chunks
            return {
                "ids": [[c.metadata["chunk_id"] for c in chosen]],
                "distances": [[0.1] * len(chosen)],
                "documents": [[c.text for c in chosen]],
                "metadatas": [[dict(c.metadata) for c in chosen]],
            }

        index.collection.query.side_effect = _query_side_effect
        merged = retrieve_multi(
            index,
            ["custody roles", "fail-closed policy validation"],
            k=6,
        )
        ids = {c.metadata["chunk_id"] for c in merged}
        self.assertEqual(ids, {"a", "b"})


if __name__ == "__main__":
    unittest.main()