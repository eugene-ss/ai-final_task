"""Chroma-backed dense vector store with idempotent upsert by chunk uid."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from langchain_chroma import Chroma
from langchain_core.documents import Document

from chatbot.rag.retrieval.embeddings import build_azure_embeddings
from chatbot.security.guardrails import sanitize_for_log

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, config_manager) -> None:
        self.config = config_manager
        self.embeddings = build_azure_embeddings(config_manager)
        self.vectorstore = Chroma(
            collection_name=config_manager.chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=config_manager.chroma_persist_dir,
        )
        logger.info("Initialized vector store at %s", config_manager.chroma_persist_dir)

    def _chroma_collection(self):
        return getattr(self.vectorstore, "_collection", None)

    def _existing_chunk_uids(self) -> Set[str]:
        coll = self._chroma_collection()
        if coll is None or coll.count() == 0:
            return set()
        batch = coll.get(include=["metadatas"])
        metas = batch.get("metadatas") or []
        return {str(m.get("chunk_uid")) for m in metas if m and m.get("chunk_uid")}

    def has_document(self, doc_id: str) -> bool:
        coll = self._chroma_collection()
        if coll is None:
            return False
        result = coll.get(where={"id": doc_id}, include=[])
        return bool(result and result.get("ids"))

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        existing = self._existing_chunk_uids()
        new_docs: List[Document] = []
        for doc in documents:
            uid = (doc.metadata or {}).get("chunk_uid")
            if uid and str(uid) in existing:
                continue
            new_docs.append(doc)
        if not new_docs:
            logger.info("All %d documents already indexed; nothing to add", len(documents))
            return
        skipped = len(documents) - len(new_docs)
        if skipped:
            logger.info("Skipping %d already-indexed chunks", skipped)
        total_batches = (len(new_docs) + batch_size - 1) // batch_size
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            self.vectorstore.add_documents(batch)
            logger.info(
                "Added batch %d/%d (%d documents)",
                i // batch_size + 1,
                total_batches,
                len(batch),
            )

    def search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if not query or not query.strip():
            return []
        if k <= 0:
            k = 10
        if filter_dict:
            results = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        logger.info("Found %d docs for query: %s", len(results), sanitize_for_log(query, 50))
        return results

    def delete_documents(self, doc_ids: List[str]) -> bool:
        valid_ids = [d for d in doc_ids if d and d.strip()]
        if not valid_ids:
            return False
        coll = self._chroma_collection()
        if coll is None:
            return False
        if len(valid_ids) == 1:
            coll.delete(where={"id": valid_ids[0]})
        else:
            coll.delete(where={"id": {"$in": valid_ids}})
        logger.info("Deleted chunks for %d logical document id(s)", len(valid_ids))
        return True

    def get_collection_stats(self) -> Dict[str, Any]:
        coll = self._chroma_collection()
        if coll is not None and hasattr(coll, "count"):
            return {"status": "active", "document_count": coll.count()}
        return {"status": "unknown", "document_count": 0}
