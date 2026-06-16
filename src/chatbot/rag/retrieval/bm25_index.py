"""BM25 keyword index with HMAC-protected JSON persistence."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from chatbot.rag.ingestion.resume_text import tokenize_for_bm25

logger = logging.getLogger(__name__)

_HMAC_KEY_ENV = "BM25_HMAC_KEY"
_HMAC_SUFFIX = ".hmac"

def _get_hmac_key() -> bytes:
    return os.environ.get(_HMAC_KEY_ENV, "default-bm25-integrity-key").encode("utf-8")

def _compute_hmac(data: bytes) -> str:
    return hmac.new(_get_hmac_key(), data, hashlib.sha256).hexdigest()

def _verify_hmac(data: bytes, expected: str) -> bool:
    actual = hmac.new(_get_hmac_key(), data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(actual, expected)

def chunk_uid_for_document(doc: Document) -> str:
    meta = doc.metadata if isinstance(doc.metadata, dict) else {}
    uid = meta.get("chunk_uid")
    if uid:
        return str(uid)
    rid = meta.get("id", "unknown")
    return f"{rid}::__{hash(doc.page_content) & 0xFFFFFFFF:x}"

class BM25ChunkIndex:
    def __init__(self) -> None:
        self._uids: List[str] = []
        self._tokenized: List[List[str]] = []
        self._documents: List[Document] = []
        self._bm25: Optional[BM25Okapi] = None

    def clear(self) -> None:
        self._uids.clear()
        self._tokenized.clear()
        self._documents.clear()
        self._bm25 = None

    def __len__(self) -> int:
        return len(self._documents)

    def _rebuild_bm25(self) -> None:
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def _append_one(self, doc: Document) -> None:
        uid = chunk_uid_for_document(doc)
        toks = tokenize_for_bm25(doc.page_content or "")
        if not toks:
            toks = ["empty"]
        self._uids.append(uid)
        self._tokenized.append(toks)
        self._documents.append(doc)

    def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            self._append_one(doc)
        self._rebuild_bm25()
        logger.info("BM25 index size: %s chunks", len(self._documents))

    def remove_by_doc_id(self, doc_id: str) -> int:
        keep = [
            i for i, d in enumerate(self._documents)
            if (d.metadata if isinstance(d.metadata, dict) else {}).get("id") != doc_id
        ]
        removed = len(self._documents) - len(keep)
        if removed == 0:
            return 0
        self._uids = [self._uids[i] for i in keep]
        self._tokenized = [self._tokenized[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        self._rebuild_bm25()
        logger.info("Removed %s BM25 entries for doc_id=%s", removed, doc_id)
        return removed

    def upsert_documents(self, documents: List[Document]) -> None:
        ids_to_replace: set[str] = set()
        for doc in documents:
            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            rid = meta.get("id")
            if rid:
                ids_to_replace.add(str(rid))
        for rid in ids_to_replace:
            self.remove_by_doc_id(rid)
        for doc in documents:
            self._append_one(doc)
        self._rebuild_bm25()

    def search(self, query: str, k: int) -> List[Document]:
        if not query or not query.strip() or not self._bm25:
            return []
        q = tokenize_for_bm25(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._documents[i] for i in ranked]

    def search_ranked_uids(self, query: str, k: int) -> List[str]:
        if not query or not query.strip() or not self._bm25:
            return []
        q = tokenize_for_bm25(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._uids[i] for i in ranked]

    def uid_to_document(self) -> Dict[str, Document]:
        return dict(zip(self._uids, self._documents))

    def save(self, path: Path) -> None:
        path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "uids": self._uids,
            "tokenized": self._tokenized,
            "metadatas": [
                d.metadata if isinstance(d.metadata, dict) else {} for d in self._documents
            ],
            "contents": [d.page_content for d in self._documents],
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with open(path, "wb") as fh:
            fh.write(data)
        hmac_path = path.with_suffix(path.suffix + _HMAC_SUFFIX)
        hmac_path.write_text(_compute_hmac(data), encoding="utf-8")
        logger.info("Saved BM25 index to %s (with HMAC integrity)", path)

    def load(self, path: Path) -> bool:
        json_path = path.with_suffix(".json")
        if not json_path.is_file():
            return False
        data = json_path.read_bytes()
        hmac_path = json_path.with_suffix(json_path.suffix + _HMAC_SUFFIX)
        if hmac_path.is_file():
            expected = hmac_path.read_text(encoding="utf-8").strip()
            if not _verify_hmac(data, expected):
                logger.error("BM25 index HMAC FAILED for %s; refusing to load", json_path)
                return False
        else:
            logger.warning("No HMAC for BM25 index at %s; loading without integrity check", json_path)
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse BM25 JSON index: %s", exc)
            return False
        self.clear()
        self._uids = list(payload["uids"])
        self._tokenized = list(payload["tokenized"])
        for meta, content in zip(payload["metadatas"], payload["contents"]):
            self._documents.append(Document(page_content=content, metadata=dict(meta)))
        self._rebuild_bm25()
        logger.info("Loaded BM25 index from %s (%s chunks)", json_path, len(self._documents))
        return True

    def rebuild_from_chroma(self, vectorstore) -> None:
        coll = getattr(vectorstore, "_collection", None)
        if coll is None:
            logger.error("Chroma collection missing; cannot rebuild BM25")
            return
        batch = coll.get(include=["documents", "metadatas"])
        docs_raw = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        self.clear()
        for content, meta in zip(docs_raw, metas):
            if content is None:
                continue
            self._append_one(Document(page_content=str(content), metadata=dict(meta or {})))
        self._rebuild_bm25()
        logger.info("Rebuilt BM25 from Chroma (%s chunks)", len(self._documents))
