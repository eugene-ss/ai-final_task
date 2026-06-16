"""Disaster knowledge layer: Pandas-backed repository over EM-DAT CSVs and a
document builder that turns rows into RAG-friendly narrative Documents."""

from chatbot.disasters.document_builder import DisasterDocumentBuilder
from chatbot.disasters.repository import DisasterRepository

__all__ = ["DisasterDocumentBuilder", "DisasterRepository"]
