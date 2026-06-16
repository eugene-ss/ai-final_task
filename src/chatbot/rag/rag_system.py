"""High-level orchestrator wrapping disaster-narrative ingestion + hybrid
retrieval + grounded answer generation.

The corpus is built on demand from the EM-DAT disaster repository (and,
optionally, any PDF reports dropped under ``<data_dir>/disaster_reports/``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from chatbot.disasters.document_builder import (
    DEFAULT_MAX_ROWS,
    DisasterDocumentBuilder,
)
from chatbot.disasters.repository import DisasterRepository
from chatbot.model.schemas import Permission, User
from chatbot.rag.answer_generator import AnswerGenerator
from chatbot.rag.ingestion.loader import DocumentLoader
from chatbot.rag.prompts.prompt_manager import PromptManager
from chatbot.rag.retrieval.bm25_index import BM25ChunkIndex
from chatbot.rag.retrieval.hybrid_retriever import HybridRetriever
from chatbot.rag.retrieval.vector_store import VectorStore
from chatbot.rag.security_facade import AccessControl
from chatbot.security.guardrails import Guardrails, init_guardrails
from chatbot.settings.app_config import ConfigManager, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


def _build_llm(config: ConfigManager):
    """Create an ``AzureChatOpenAI`` / ``ChatOpenAI`` for answer generation."""

    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    llm_cfg = config.get_llm_config()
    if llm_cfg.api_version:
        return AzureChatOpenAI(
            azure_endpoint=config.endpoint_url,
            api_key=config.api_key,
            azure_deployment=llm_cfg.deployment_name or llm_cfg.model,
            api_version=llm_cfg.api_version,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
        )
    return ChatOpenAI(
        api_key=config.api_key,
        base_url=config.endpoint_url or None,
        model=llm_cfg.model,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
    )


class RAGSystem:
    """End-to-end RAG pipeline over disaster narrative documents."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        llm=None,
        vision_llm=None,
        disaster_repository: Optional[DisasterRepository] = None,
        document_builder: Optional[DisasterDocumentBuilder] = None,
    ) -> None:
        self.config = ConfigManager(config_path)
        self._guardrails: Guardrails = init_guardrails(self.config)
        prompts_dir = self.config.project_root / "prompts"
        self.prompt_manager = PromptManager(prompts_dir=str(prompts_dir))

        self.llm = llm if llm is not None else _build_llm(self.config)
        self.vision_llm = vision_llm or self.llm

        self.document_loader = DocumentLoader(self.config, vision_llm=self.vision_llm)
        self.vector_store = VectorStore(self.config)
        self.access_control = AccessControl(self.config)
        self.bm25_index = BM25ChunkIndex()

        dcfg = self.config.app_settings.disasters
        self.disaster_repository = disaster_repository or DisasterRepository(
            data_dir=self.config.data_dir,
            csv_files=dcfg.csv_files,
            max_limit=dcfg.max_query_limit,
        )
        self.document_builder = document_builder or DisasterDocumentBuilder(
            repository=self.disaster_repository,
            max_rows=dcfg.indexing_max_rows,
        )

        self._bm25_path = (
            self.config.vector_db_dir
            / self.config.app_settings.hybrid_search.bm25_persist_filename
        )

        self._retriever = HybridRetriever(
            self.config,
            self.vector_store,
            self.bm25_index,
            self.access_control,
            guardrails=self._guardrails,
        )
        self._answer_gen = AnswerGenerator(
            self.llm,
            self.config,
            self.prompt_manager,
            guardrails=self._guardrails,
        )
        self._init_hybrid_from_disk_or_chroma()
        logger.info("RAG System initialized successfully")

    # --------------------------------------------------------------- internals
    def _init_hybrid_from_disk_or_chroma(self) -> None:
        if not self.config.app_settings.hybrid_search.enabled:
            return
        if self.bm25_index.load(self._bm25_path):
            return
        logger.warning("BM25 index not found; rebuilding from Chroma (if any)")
        self.bm25_index.rebuild_from_chroma(self.vector_store.vectorstore)
        if len(self.bm25_index) > 0:
            self.bm25_index.save(self._bm25_path)

    def _clear_existing_index(self) -> None:
        coll = getattr(self.vector_store.vectorstore, "_collection", None)
        if coll is not None:
            batch = coll.get(include=["metadatas"])
            ids = batch.get("ids") or []
            if ids:
                coll.delete(ids=ids)
                logger.info("Deleted %d existing chunks before reindex", len(ids))
        self.bm25_index.clear()

    # ----------------------------------------------------------------- public
    def ingest_disasters(
        self,
        max_rows: Optional[int] = None,
        strategy: str = "recent",
        include_pdf_reports: bool = True,
        force: bool = False,
    ) -> Dict[str, int]:
        """Build narrative documents from the disaster repository and index them.

        Args:
            max_rows: cap on EM-DAT rows to index; defaults to the configured
                ``disasters.indexing_max_rows``.
            strategy: ``"recent"`` (default) or ``"impact"`` for row selection.
            include_pdf_reports: also ingest any PDFs under
                ``<data_dir>/disaster_reports/``.
            force: if true, drop the existing index before re-ingesting.

        Returns counts of indexed chunks split by source.
        """

        if force:
            self._clear_existing_index()

        narratives = self.document_builder.build_documents(
            max_rows=max_rows,
            strategy=strategy,
        )
        pdf_docs = self.document_loader.load_pdf_reports() if include_pdf_reports else []
        raw_docs = narratives + pdf_docs
        chunks = self.document_loader.chunk_documents(raw_docs)
        if not chunks:
            logger.warning("No disaster documents to index")
            return {"narratives": 0, "pdf_chunks": 0, "total_chunks": 0}

        self.vector_store.add_documents(chunks)
        if self.config.app_settings.hybrid_search.enabled:
            if len(self.bm25_index) == 0:
                self.bm25_index.add_documents(chunks)
            else:
                self.bm25_index.upsert_documents(chunks)
            self.bm25_index.save(self._bm25_path)

        pdf_chunks = sum(
            1 for c in chunks if (c.metadata or {}).get("source") == "pdf"
        )
        narrative_chunks = len(chunks) - pdf_chunks
        logger.info(
            "Indexed %d chunks (%d narratives, %d pdf)",
            len(chunks),
            narrative_chunks,
            pdf_chunks,
        )
        return {
            "narratives": narrative_chunks,
            "pdf_chunks": pdf_chunks,
            "total_chunks": len(chunks),
        }

    def search(self, query: str, k: int = 10, user: Optional[User] = None):
        return self._retriever.search(query, k=k, user=user)

    def generate_answer(
        self,
        query: str,
        retrieved_docs,
        user: Optional[User] = None,
    ) -> str:
        return self._answer_gen.generate_answer(
            query, retrieved_docs, user, self.access_control
        )

    def answer(self, query: str, k: int = 5, user: Optional[User] = None) -> Dict[str, Any]:
        results = self.search(query, k=k, user=user)
        answer_text = self.generate_answer(query, results, user=user) if results else ""
        return {"answer": answer_text, "results": results}

    def list_categories(self) -> List[str]:
        coll = getattr(self.vector_store.vectorstore, "_collection", None)
        if coll is None or coll.count() == 0:
            return []
        batch = coll.get(include=["metadatas"])
        cats: set[str] = set()
        for meta in batch.get("metadatas") or []:
            if meta and meta.get("category"):
                cats.add(str(meta["category"]))
        return sorted(cats)

    def delete_documents(
        self,
        doc_ids: List[str],
        user: Optional[User] = None,
    ) -> bool:
        self._guardrails.enforce_user(user, "delete_documents")
        if user and not self.access_control.check_permission(user, Permission.DELETE):
            return False
        valid_ids = [d for d in doc_ids if d and d.strip()]
        if not valid_ids:
            return False
        if not self.vector_store.delete_documents(valid_ids):
            return False
        if self.config.app_settings.hybrid_search.enabled:
            for did in valid_ids:
                self.bm25_index.remove_by_doc_id(did)
            if len(self.bm25_index) > 0:
                self.bm25_index.save(self._bm25_path)
        if user:
            self.access_control.log_access(
                user, "delete_documents", f"{len(valid_ids)} documents", True
            )
        return True

    def get_system_stats(self) -> Dict[str, Any]:
        hy = self.config.app_settings.hybrid_search
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "vector_db_directory": str(self.config.vector_db_dir),
            "data_directory": str(self.config.data_dir),
            "hybrid_search_enabled": hy.enabled,
            "bm25_chunks": len(self.bm25_index),
            "structured_output_enabled": self.config.app_settings.structured_output.enabled,
            "disaster_rows_loaded": int(len(self.disaster_repository.df))
            if self.disaster_repository._df is not None
            else 0,
        }
