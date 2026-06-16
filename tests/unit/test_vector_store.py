"""Tests for the Chroma-backed vector store wrapper (Chroma + Azure mocked)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

class _FakeCollection:
    def __init__(self) -> None:
        self._metas: List[Dict[str, Any]] = []
        self._ids: List[str] = []
        self.deleted: List[Dict[str, Any]] = []

    def count(self) -> int:
        return len(self._ids)

    def get(self, include: Optional[List[str]] = None, where: Optional[Dict] = None):
        if where:
            ids = [
                i for i, m in zip(self._ids, self._metas)
                if m.get("id") == where.get("id")
            ]
            return {"ids": ids}
        return {"metadatas": list(self._metas), "ids": list(self._ids)}

    def delete(self, where: Optional[Dict] = None, ids: Optional[List[str]] = None) -> None:
        self.deleted.append({"where": where, "ids": ids})

class _FakeChroma:
    def __init__(self, collection_name: str, **_) -> None:
        self.collection_name = collection_name
        self._collection = _FakeCollection()
        self._added: List[Document] = []

    def add_documents(self, docs: List[Document]) -> None:
        for d in docs:
            meta = d.metadata or {}
            self._collection._metas.append(dict(meta))
            self._collection._ids.append(str(meta.get("chunk_uid") or len(self._collection._ids)))
        self._added.extend(docs)

    def similarity_search(self, query: str, k: int = 10, filter: Dict | None = None):
        return self._added[:k]

    def similarity_search_with_score(self, query: str, k: int = 10):
        return [(d, 0.5) for d in self._added[:k]]


@pytest.fixture
def vs(monkeypatch, config_manager):
    monkeypatch.setattr(
        "chatbot.rag.retrieval.vector_store.Chroma", _FakeChroma
    )
    monkeypatch.setattr(
        "chatbot.rag.retrieval.vector_store.build_azure_embeddings",
        lambda cm: MagicMock(),
    )
    from chatbot.rag.retrieval.vector_store import VectorStore

    return VectorStore(config_manager)


def _doc(text: str, chunk_uid: str, did: str = "1") -> Document:
    return Document(
        page_content=text,
        metadata={
            "id": did,
            "chunk_uid": chunk_uid,
            "category": "Earthquake",
            "source": "emdat",
        },
    )

def test_add_documents_deduplicates_by_chunk_uid(vs):
    vs.add_documents([_doc("a", "1:0"), _doc("b", "1:1")])
    # Re-adding the same uids should skip.
    vs.add_documents([_doc("a", "1:0")])
    assert vs.get_collection_stats()["document_count"] == 2

def test_search_empty_query(vs):
    assert vs.search("") == []

def test_search_returns_documents(vs):
    vs.add_documents([_doc("x", "1:0"), _doc("y", "1:1")])
    out = vs.search("anything", k=5)
    assert len(out) == 2

def test_search_with_invalid_k_defaults(vs):
    vs.add_documents([_doc("x", "1:0")])
    out = vs.search("anything", k=0)
    assert isinstance(out, list)

def test_has_document(vs):
    vs.add_documents([_doc("text", "1:0", did="abc")])
    assert vs.has_document("abc") is True
    assert vs.has_document("missing") is False


def test_delete_documents_single_and_many(vs):
    vs.add_documents([_doc("text", "1:0", did="abc")])
    assert vs.delete_documents(["abc"]) is True
    assert vs.delete_documents(["abc", "def"]) is True
    # Last call used $in
    last = vs.vectorstore._collection.deleted[-1]
    assert "$in" in str(last)

def test_delete_documents_empty_returns_false(vs):
    assert vs.delete_documents([]) is False
    assert vs.delete_documents([""]) is False

def test_get_collection_stats(vs):
    stats = vs.get_collection_stats()
    assert stats["status"] in {"active", "unknown"}
    assert "document_count" in stats
