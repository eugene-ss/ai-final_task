"""RAG MCP server.

Spawned as a subprocess by the orchestrator. Wraps ``RAGSystem`` (hybrid
Chroma + BM25 retrieval over disaster narrative documents) and exposes it as
MCP tools the document RAG agent can call.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mcp.server.fastmcp import FastMCP

from chatbot.rag.rag_system import RAGSystem
from chatbot.security.guardrails import validate_tool_output
from chatbot.settings.app_config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_server")

_cfg = ConfigManager()
_rag = RAGSystem(_cfg.config_path)

mcp = FastMCP("rag")

def _pack(tool_name: str, payload: object) -> str:
    text = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, default=str, indent=2)
    )
    return validate_tool_output(f"rag.{tool_name}", text)

def _error_payload(code: str, message: str, retryable: bool = False) -> dict:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }

def _serialize_result(result) -> dict:
    meta = result.document.metadata.model_dump(exclude_none=True)
    return {
        "score": round(float(result.score), 4),
        "method": result.method,
        "doc_id": meta.get("id"),
        "category": meta.get("category"),
        "source": meta.get("source"),
        "source_type": meta.get("source_type", "text"),
        "headline": meta.get("headline"),
        "snippet": (result.document.page_content or "")[:600],
    }

@mcp.tool()
async def hybrid_search(query: str, k: int = 5, category: Optional[str] = None) -> str:
    """Hybrid (dense Chroma + sparse BM25) retrieval over disaster narratives.

    Args:
        query: natural-language query.
        k: number of chunks to return (default 5).
        category: optional disaster type to filter by (post-retrieval).
    """

    try:
        results = _rag.search(query, k=max(1, int(k)), user=None)
    except Exception:
        logger.exception("hybrid_search failed")
        return _pack(
            "hybrid_search",
            _error_payload("INTERNAL_ERROR", "Hybrid retrieval failed."),
        )

    if category:
        results = [
            r for r in results
            if (getattr(r.document.metadata, "category", "") or "").lower()
            == category.lower()
        ]

    payload = {
        "query": query,
        "k": k,
        "category": category,
        "results": [_serialize_result(r) for r in results],
    }
    return _pack("hybrid_search", payload)

@mcp.tool()
async def answer_with_rag(query: str, k: int = 5) -> str:
    """Run hybrid retrieval and synthesise a grounded answer.

    Use this when the user wants a narrative answer (e.g. "tell me about the
    1995 Kobe earthquake") rather than raw aggregated numbers.

    Args:
        query: natural-language question.
        k: number of context chunks to retrieve (default 5).
    """

    try:
        bundle = _rag.answer(query, k=max(1, int(k)), user=None)
    except Exception:  # noqa: BLE001
        logger.exception("answer_with_rag failed")
        return _pack(
            "answer_with_rag",
            _error_payload("INTERNAL_ERROR", "RAG answer generation failed."),
        )

    payload = {
        "query": query,
        "answer": bundle["answer"],
        "results": [_serialize_result(r) for r in bundle["results"]],
    }
    return _pack("answer_with_rag", payload)

@mcp.tool()
async def ingest_corpus(
    force: bool = False,
    max_rows: Optional[int] = None,
    strategy: str = "recent",
) -> str:
    """Build narrative documents from EM-DAT and index them into Chroma + BM25.

    Also picks up any PDF reports dropped under ``<data_dir>/disaster_reports/``
    so the multimodal pipeline (tables, image descriptions) is exercised when
    real reports are present.

    Args:
        force: if true, delete the existing index before re-ingesting.
        max_rows: cap on EM-DAT rows to index (defaults to config value).
        strategy: ``"recent"`` (newest events first) or ``"impact"`` (deadliest first).
    """

    try:
        counts = _rag.ingest_disasters(
            max_rows=max_rows,
            strategy=strategy,
            force=force,
        )
    except Exception:
        logger.exception("ingest_corpus failed")
        return _pack(
            "ingest_corpus",
            _error_payload("INTERNAL_ERROR", "Corpus ingestion failed."),
        )
    return _pack(
        "ingest_corpus",
        {"counts": counts, "stats": _rag.get_system_stats()},
    )

@mcp.tool()
async def list_categories() -> str:
    """List the distinct disaster categories present in the indexed corpus."""

    return _pack("list_categories", {"categories": _rag.list_categories()})

if __name__ == "__main__":
    mcp.run()
