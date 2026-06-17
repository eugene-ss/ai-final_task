"""Tests for the BM25 keyword index"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
import pytest

from chatbot.rag.retrieval.bm25_index import BM25ChunkIndex, chunk_uid_for_document

def _doc(text: str, did: str, chunk_uid: str | None = None) -> Document:
    meta = {"id": did, "chunk_uid": chunk_uid or f"{did}:0", "category": "X", "source": "csv"}
    return Document(page_content=text, metadata=meta)

def test_add_and_search_returns_relevant_first():
    idx = BM25ChunkIndex()
    idx.add_documents(
        [
            _doc("python machine learning developer", "1"),
            _doc("registered nurse with patient care", "2"),
            _doc("python data engineer cloud aws", "3"),
        ]
    )
    results = idx.search("python developer", k=2)
    ids = [r.metadata["id"] for r in results]
    assert "1" in ids and "3" in ids
    assert "2" not in ids

def test_search_ranked_uids_matches_search_docs():
    idx = BM25ChunkIndex()
    idx.add_documents([_doc("python coding", "1"), _doc("financial analysis", "2")])
    docs = idx.search("python", k=2)
    uids = idx.search_ranked_uids("python", k=2)
    assert [chunk_uid_for_document(d) for d in docs] == uids

def test_remove_by_doc_id():
    idx = BM25ChunkIndex()
    idx.add_documents(
        [
            _doc("a b c", "1", chunk_uid="1:0"),
            _doc("d e f", "1", chunk_uid="1:1"),
            _doc("g h i", "2", chunk_uid="2:0"),
        ]
    )
    removed = idx.remove_by_doc_id("1")
    assert removed == 2
    assert len(idx) == 1
    # The remaining document (id=2) does not contain "a"; verify it's the only doc.
    remaining = idx.search("g h i", k=5)
    assert len(remaining) == 1
    assert remaining[0].metadata["id"] == "2"

def test_upsert_replaces_existing_doc():
    idx = BM25ChunkIndex()
    idx.add_documents([_doc("original text", "10", chunk_uid="10:0")])
    idx.upsert_documents([_doc("replacement text", "10", chunk_uid="10:0")])
    results = idx.search("replacement", k=1)
    assert results and results[0].metadata["id"] == "10"
    # Old content gone.
    assert idx.search("original", k=1)[0].page_content == "replacement text"

def test_empty_query_returns_empty():
    idx = BM25ChunkIndex()
    idx.add_documents([_doc("foo", "1")])
    assert idx.search("", k=5) == []
    assert idx.search_ranked_uids("", k=5) == []

def test_search_before_add_returns_empty():
    idx = BM25ChunkIndex()
    assert idx.search("anything", k=5) == []

def test_clear_resets_state():
    idx = BM25ChunkIndex()
    idx.add_documents([_doc("foo", "1")])
    idx.clear()
    assert len(idx) == 0
    assert idx.search("foo", k=1) == []

def test_save_and_load_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BM25_HMAC_KEY", "deterministic-key")
    idx = BM25ChunkIndex()
    idx.add_documents([_doc("python", "1"), _doc("finance", "2")])
    save_path = tmp_path / "bm25"
    idx.save(save_path)
    assert (tmp_path / "bm25.json").is_file()
    assert (tmp_path / "bm25.json.hmac").is_file()

    loaded = BM25ChunkIndex()
    assert loaded.load(save_path) is True
    assert len(loaded) == 2
    # Search works after reload.
    assert loaded.search("python", k=1)[0].metadata["id"] == "1"

def test_load_with_tampered_hmac_refuses(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BM25_HMAC_KEY", "deterministic-key")
    idx = BM25ChunkIndex()
    idx.add_documents([_doc("hello world", "1")])
    save_path = tmp_path / "bm25"
    idx.save(save_path)
    (tmp_path / "bm25.json.hmac").write_text("00" * 32, encoding="utf-8")
    loaded = BM25ChunkIndex()
    assert loaded.load(save_path) is False
    assert len(loaded) == 0

def test_load_without_hmac_refuses(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BM25_HMAC_KEY", "deterministic-key")
    idx = BM25ChunkIndex()
    idx.add_documents([_doc("hello world", "1")])
    save_path = tmp_path / "bm25"
    idx.save(save_path)
    (tmp_path / "bm25.json.hmac").unlink()
    loaded = BM25ChunkIndex()
    assert loaded.load(save_path) is False
    assert len(loaded) == 0

def test_save_requires_non_default_hmac_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BM25_HMAC_KEY", raising=False)
    idx = BM25ChunkIndex()
    idx.add_documents([_doc("secure text", "1")])
    with pytest.raises(RuntimeError, match="BM25_HMAC_KEY"):
        idx.save(tmp_path / "bm25")

def test_load_missing_file_returns_false(tmp_path: Path):
    idx = BM25ChunkIndex()
    assert idx.load(tmp_path / "missing") is False