"""Layout-aware PDF extraction with table + image handling (multimodal RAG).

Backend preference:
    1. ``unstructured`` (hi_res, tables + images)
    2. ``pymupdf`` (text + raw image bytes for vision LLM)
    3. ``pypdf`` (text only - last resort)

When a chat LLM with vision support is passed via ``vision_llm``, raw image bytes
are sent to the model so the resulting embeddings are grounded in an *actual*
visual description rather than a placeholder string.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_HAS_UNSTRUCTURED = False
_HAS_PYMUPDF = False

try:
    from unstructured.partition.pdf import partition_pdf

    _HAS_UNSTRUCTURED = True
except ImportError:
    partition_pdf = None

try:
    import fitz

    _HAS_PYMUPDF = True
except ImportError:
    fitz = None

def _html_table_to_markdown(html: str) -> str:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
    md_rows: List[str] = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
        cleaned = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        md_rows.append("| " + " | ".join(cleaned) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(cleaned)) + " |")
    return "\n".join(md_rows) if md_rows else html

def _extract_with_unstructured(pdf_path: str) -> List[Tuple[str, str, Optional[bytes]]]:
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
    )
    results: List[Tuple[str, str, Optional[bytes]]] = []
    for el in elements:
        category = getattr(el, "category", "")
        text = str(el).strip()
        if not text:
            continue
        if category == "Table":
            html = getattr(el.metadata, "text_as_html", None)
            if html:
                text = _html_table_to_markdown(html)
            results.append((text, "table", None))
        elif category == "Image":
            results.append((text, "image_description", None))
        else:
            results.append((text, "text", None))
    return results

def _extract_with_pymupdf(pdf_path: str) -> List[Tuple[str, str, Optional[bytes]]]:
    doc = fitz.open(pdf_path)
    results: List[Tuple[str, str, Optional[bytes]]] = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text and text.strip():
                results.append((text.strip(), "text", None))
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                    placeholder = (
                        f"[Image on page {page_num + 1}, index {img_index}]"
                    )
                    results.append((placeholder, "image_description", img_bytes))
                except Exception as exc:
                    logger.debug(
                        "Could not extract image %s on page %s: %s",
                        img_index,
                        page_num,
                        exc,
                    )
    finally:
        doc.close()
    return results

def _extract_with_pypdf(pdf_path: str) -> List[Tuple[str, str, Optional[bytes]]]:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    out: List[Tuple[str, str, Optional[bytes]]] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            out.append((text.strip(), "text", None))
    return out

def _describe_image_with_llm(
    placeholder: str,
    image_bytes: Optional[bytes],
    vision_llm,
) -> str:
    """Use the chat LLM (with vision support) to caption an image."""

    try:
        from langchain_core.messages import HumanMessage

        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            content = [
                {
                    "type": "text",
                    "text": (
                        "Describe this image from a document concisely. "
                        "Focus on any people, skills, certifications, charts, or data shown."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ]
        else:
            content = [
                {
                    "type": "text",
                    "text": (
                        "Provide a concise description of the following image placeholder "
                        "from a document, inferring any context from the surrounding name: "
                        + placeholder
                    ),
                }
            ]
        msg = HumanMessage(content=content)
        response = vision_llm.invoke([msg])
        desc = (getattr(response, "content", "") or "").strip()
        return desc if desc else placeholder
    except Exception as exc:  # pragma: no cover - vision is best-effort
        logger.warning("Vision LLM description failed: %s", exc)
        return placeholder

def extract_pdf_elements(
    pdf_path: str | Path,
    vision_llm=None,
) -> List[Document]:
    """Extract text, tables, and image-description Documents from a PDF."""

    path_str = str(pdf_path)
    if _HAS_UNSTRUCTURED:
        logger.info("Using unstructured for PDF extraction: %s", path_str)
        elements = _extract_with_unstructured(path_str)
    elif _HAS_PYMUPDF:
        logger.info("Using PyMuPDF for PDF extraction: %s", path_str)
        elements = _extract_with_pymupdf(path_str)
    else:
        logger.info("Falling back to pypdf for PDF extraction: %s", path_str)
        elements = _extract_with_pypdf(path_str)

    docs: List[Document] = []
    counts = {"text": 0, "table": 0, "image_description": 0}
    for content, source_type, image_bytes in elements:
        if not content or not content.strip():
            continue
        counts[source_type] = counts.get(source_type, 0) + 1
        if source_type == "image_description" and vision_llm is not None:
            content = _describe_image_with_llm(content, image_bytes, vision_llm)
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "pdf",
                    "source_type": source_type,
                    "file_path": path_str,
                },
            )
        )

    logger.info(
        "Extracted %d elements from %s (text=%d, table=%d, image=%d)",
        len(docs),
        path_str,
        counts.get("text", 0),
        counts.get("table", 0),
        counts.get("image_description", 0),
    )
    return docs

def get_available_backend() -> str:
    if _HAS_UNSTRUCTURED:
        return "unstructured"
    if _HAS_PYMUPDF:
        return "pymupdf"
    return "pypdf"
