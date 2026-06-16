"""Hybrid retriever: dense (Chroma) + sparse (BM25), fused via RRF or weighted rank."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from chatbot.model.schemas import (
    DocumentMetadata,
    ResumeDocument,
    SearchResult,
    User,
)
from chatbot.rag.retrieval.bm25_index import BM25ChunkIndex, chunk_uid_for_document
from chatbot.rag.retrieval.vector_store import VectorStore
from chatbot.rag.security_facade import AccessControl
from chatbot.security.guardrails import Guardrails, GuardrailViolation, sanitize_for_log
from chatbot.settings.app_config import ConfigManager

logger = logging.getLogger(__name__)

def _rrf(ranked_lists: List[List[str]], rrf_k: int) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, uid in enumerate(ranking, start=1):
            if uid:
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def _weighted_fusion(
    ranked_lists: List[List[str]],
    weights: List[float],
) -> List[Tuple[str, float]]:
    if not ranked_lists or len(weights) != len(ranked_lists):
        return []
    scores: Dict[str, float] = {}
    for ranking, w in zip(ranked_lists, weights):
        if not ranking or w <= 0:
            continue
        n = len(ranking)
        for rank, uid in enumerate(ranking, start=1):
            if uid:
                scores[uid] = scores.get(uid, 0.0) + w * (1.0 / rank) / max(1.0, float(n))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class HybridRetriever:
    def __init__(
        self,
        config: ConfigManager,
        vector_store: VectorStore,
        bm25_index: BM25ChunkIndex,
        access_control: AccessControl,
        guardrails: Optional[Guardrails] = None,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.access_control = access_control
        self._guardrails = guardrails or Guardrails(config)

    @staticmethod
    def documents_to_search_results(
        documents: List[Any], rank_method: str
    ) -> List[SearchResult]:
        results: List[SearchResult] = []
        n = len(documents)
        for i, doc in enumerate(documents):
            if isinstance(doc.metadata, dict):
                metadata = DocumentMetadata(**doc.metadata)
            else:
                metadata = DocumentMetadata(id=str(i), category="Unknown", source="unknown")
            text = (doc.page_content or "").strip() or " "
            resume_doc = ResumeDocument(page_content=text, metadata=metadata)
            score = max(0.01, 1.0 - i * (0.99 / max(n, 1)))
            results.append(
                SearchResult(document=resume_doc, score=float(score), method=rank_method)
            )
        return results

    def search(
        self,
        query: str,
        k: int = 10,
        user: Optional[User] = None,
    ) -> List[SearchResult]:
        self._guardrails.enforce_user(user, "search")
        if not query or not query.strip():
            return []
        if user is not None and not isinstance(user, User):
            return []
        if k <= 0:
            k = 10
        try:
            q = self._guardrails.validate_user_input(query)
        except GuardrailViolation as exc:
            logger.warning("Search query blocked by guardrails: %s", exc)
            return []

        filt = self.access_control.create_filter(user) if user else None
        k_fetch = k * 2
        hy = self.config.app_settings.hybrid_search

        docs = self.vector_store.search(q, k=k_fetch, filter_dict=filt)
        rank_method = "dense"

        if hy.enabled and len(self.bm25_index) > 0:
            rank_method = "hybrid"
            sk = hy.sparse_k
            dense_uids = [chunk_uid_for_document(d) for d in docs]
            sparse_uids = self.bm25_index.search_ranked_uids(q, sk)
            by_uid: Dict[str, Any] = {chunk_uid_for_document(d): d for d in docs}
            for d in self.bm25_index.search(q, sk):
                by_uid.setdefault(chunk_uid_for_document(d), d)
            fused = (
                _weighted_fusion([dense_uids, sparse_uids], [hy.dense_weight, hy.sparse_weight])
                if hy.fusion == "weighted"
                else _rrf([dense_uids, sparse_uids], hy.rrf_k)
            )
            merged: List[Any] = []
            seen: set[str] = set()
            for uid, _ in fused:
                if uid in by_uid and uid not in seen:
                    merged.append(by_uid[uid])
                    seen.add(uid)
                if len(merged) >= k_fetch:
                    break
            docs = merged
        elif hy.enabled:
            logger.warning("Hybrid search enabled but BM25 index empty; dense only")

        if not docs:
            return []

        results = self.documents_to_search_results(docs, rank_method)

        if user:
            payload = [
                {"document": r.document, "score": r.score, "method": r.method} for r in results
            ]
            filtered = self.access_control.filter_results(user, payload)
            results = [
                SearchResult(document=x["document"], score=x["score"], method=x["method"])
                for x in filtered
            ]
            self.access_control.log_access(
                user, "search", sanitize_for_log(query, 200), True
            )

        deduped: List[SearchResult] = []
        seen_ids: set[str] = set()
        for r in results:
            rid = getattr(r.document.metadata, "id", None)
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                deduped.append(r)

        out = deduped if deduped else results
        return out[:k]
