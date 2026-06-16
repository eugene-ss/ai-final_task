"""Generic document chunker.

The original task indexed a resume CSV; the chatbot now indexes disaster
narrative Documents produced by ``DisasterDocumentBuilder``. The shared piece
is the chunker: split long ``text`` documents with the configured splitter
while passing through ``table`` / ``image_description`` documents as-is.

The loader also discovers loose PDF reports under ``<data_dir>/disaster_reports/``
when ``load_pdfs=True`` so the multimodal pipeline (focus area D) is still
exercised end-to-end if the user drops disaster report PDFs in that folder.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chatbot.model.schemas import DocumentMetadata
from chatbot.rag.ingestion.multimodal_pdf import (
    extract_pdf_elements,
    get_available_backend,
)
from chatbot.rag.ingestion.resume_text import (
    extract_headline,
    normalize_resume_text,
)

logger = logging.getLogger(__name__)

PDF_REPORTS_SUBDIR = "disaster_reports"

def assign_chunk_metadata(chunks: List[Document]) -> List[Document]:
    """Attach ``chunk_index`` / ``total_chunks`` / ``chunk_uid`` to each chunk."""

    if not chunks:
        return chunks
    by_id: defaultdict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        meta = c.metadata if isinstance(c.metadata, dict) else {}
        rid = str(meta.get("id", i))
        by_id[rid].append(i)
    out = list(chunks)
    for rid, indices in by_id.items():
        n = len(indices)
        for j, idx in enumerate(indices):
            ch = out[idx]
            md = dict(ch.metadata) if isinstance(ch.metadata, dict) else {}
            md["chunk_index"] = j
            md["total_chunks"] = n
            md["chunk_uid"] = f"{rid}:{j}"
            out[idx] = Document(page_content=ch.page_content, metadata=md)
    return out

class DocumentLoader:
    """Chunks documents and (optionally) ingests loose PDF reports."""

    def __init__(self, config_manager, vision_llm=None) -> None:
        self.config = config_manager
        self.vision_llm = vision_llm
        text_cfg = config_manager.get_text_splitter_config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_cfg.chunk_size,
            chunk_overlap=text_cfg.chunk_overlap,
        )
        logger.info("PDF backend: %s", get_available_backend())

    def _doc_processing(self):
        return self.config.app_settings.document_processing

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split long text Documents; pass table / image Documents through."""

        if not documents:
            return []
        text_docs: List[Document] = []
        passthrough: List[Document] = []
        for doc in documents:
            st = (doc.metadata or {}).get("source_type", "text")
            if st in ("table", "image_description"):
                passthrough.append(doc)
            else:
                text_docs.append(doc)
        chunked = self.text_splitter.split_documents(text_docs) if text_docs else []
        chunked.extend(passthrough)
        logger.info(
            "Created %d chunks (text-split=%d, passthrough=%d) from %d documents",
            len(chunked),
            len(chunked) - len(passthrough),
            len(passthrough),
            len(documents),
        )
        return assign_chunk_metadata(chunked)

    def load_pdf_reports(self, default_category: str = "Report") -> List[Document]:
        """Ingest any PDFs found under ``<data_dir>/disaster_reports/``.

        Each PDF page / table / image becomes its own ``Document``; image
        descriptions are produced by the chat-vision LLM when one is provided.
        Returns the raw (un-chunked) Documents; pass through ``chunk_documents``
        before indexing.
        """

        reports_dir = self.config.data_dir / PDF_REPORTS_SUBDIR
        if not reports_dir.is_dir():
            return []
        out: List[Document] = []
        dp = self._doc_processing()
        for pdf_path in sorted(reports_dir.glob("*.pdf")):
            raw_docs = extract_pdf_elements(str(pdf_path), vision_llm=self.vision_llm)
            for doc in raw_docs:
                text = doc.page_content or ""
                source_type = (doc.metadata or {}).get("source_type", "text")
                if source_type == "text" and dp.normalize_text:
                    text = normalize_resume_text(text)
                headline = None
                if source_type == "text" and dp.extract_headline_skills:
                    headline = extract_headline(text)
                metadata = DocumentMetadata(
                    id=pdf_path.stem,
                    category=default_category,
                    source="pdf",
                    source_type=source_type,
                    file_path=str(pdf_path),
                    headline=headline,
                )
                doc.page_content = text
                doc.metadata = metadata.model_dump(exclude_none=True)
                out.append(doc)
            logger.info("Ingested PDF %s -> %d elements", pdf_path.name, len(raw_docs))
        return out

    @staticmethod
    def reports_directory(data_dir: Path) -> Path:
        return Path(data_dir) / PDF_REPORTS_SUBDIR
