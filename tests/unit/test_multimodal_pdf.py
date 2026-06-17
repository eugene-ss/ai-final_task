"""Tests for the multimodal PDF extractor (without requiring real backends)"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from chatbot.rag.ingestion import multimodal_pdf

def test_html_table_to_markdown_basic():
    html = "<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>"
    md = multimodal_pdf._html_table_to_markdown(html)
    lines = md.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 2 |"

def test_describe_image_with_llm_returns_llm_text():
    fake_llm = MagicMock()
    fake_response = MagicMock()
    fake_response.content = "Photo of a person holding a certificate."
    fake_llm.invoke.return_value = fake_response
    out = multimodal_pdf._describe_image_with_llm("[Image]", b"\x89PNG\r\n", fake_llm)
    assert out == "Photo of a person holding a certificate."
    fake_llm.invoke.assert_called_once()

def test_describe_image_with_llm_falls_back_on_error():
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = RuntimeError("network down")
    out = multimodal_pdf._describe_image_with_llm("[Image]", None, fake_llm)
    assert out == "[Image]"

def test_describe_image_returns_placeholder_when_response_empty():
    fake_llm = MagicMock()
    fake_response = MagicMock()
    fake_response.content = "   "
    fake_llm.invoke.return_value = fake_response
    out = multimodal_pdf._describe_image_with_llm("placeholder", None, fake_llm)
    assert out == "placeholder"

def test_extract_pdf_elements_uses_pypdf_fallback(monkeypatch):
    monkeypatch.setattr(multimodal_pdf, "_HAS_UNSTRUCTURED", False)
    monkeypatch.setattr(multimodal_pdf, "_HAS_PYMUPDF", False)

    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeReader:
        def __init__(self, path: str) -> None:
            self.pages = [_FakePage("Hello world"), _FakePage("   ")]

    monkeypatch.setattr("pypdf.PdfReader", _FakeReader)
    docs = multimodal_pdf.extract_pdf_elements("ignored.pdf")
    assert len(docs) == 1
    assert docs[0].page_content == "Hello world"
    assert docs[0].metadata["source_type"] == "text"

def test_extract_pdf_elements_applies_vision_llm_for_images(monkeypatch):
    monkeypatch.setattr(multimodal_pdf, "_HAS_UNSTRUCTURED", False)
    monkeypatch.setattr(multimodal_pdf, "_HAS_PYMUPDF", False)
    monkeypatch.setattr(
        multimodal_pdf,
        "_extract_with_pypdf",
        lambda p: [("Body text", "text", None), ("[Image on page 1]", "image_description", None)],
    )

    captioned: list[str] = []

    def _stub_describe(placeholder, image_bytes, llm):
        captioned.append(placeholder)
        return "captioned description"

    monkeypatch.setattr(multimodal_pdf, "_describe_image_with_llm", _stub_describe)
    fake_llm = MagicMock()
    docs = multimodal_pdf.extract_pdf_elements("any.pdf", vision_llm=fake_llm)
    image_doc = next(d for d in docs if d.metadata["source_type"] == "image_description")
    assert image_doc.page_content == "captioned description"
    assert captioned == ["[Image on page 1]"]

def test_get_available_backend(monkeypatch):
    monkeypatch.setattr(multimodal_pdf, "_HAS_UNSTRUCTURED", False)
    monkeypatch.setattr(multimodal_pdf, "_HAS_PYMUPDF", False)
    assert multimodal_pdf.get_available_backend() == "pypdf"
    monkeypatch.setattr(multimodal_pdf, "_HAS_PYMUPDF", True)
    assert multimodal_pdf.get_available_backend() == "pymupdf"
    monkeypatch.setattr(multimodal_pdf, "_HAS_UNSTRUCTURED", True)
    assert multimodal_pdf.get_available_backend() == "unstructured"

def test_extract_pdf_elements_drops_empty_strings(monkeypatch):
    monkeypatch.setattr(multimodal_pdf, "_HAS_UNSTRUCTURED", False)
    monkeypatch.setattr(multimodal_pdf, "_HAS_PYMUPDF", False)
    monkeypatch.setattr(
        multimodal_pdf,
        "_extract_with_pypdf",
        lambda p: [("real", "text", None), ("   ", "text", None)],
    )
    docs = multimodal_pdf.extract_pdf_elements("dummy.pdf")
    assert len(docs) == 1