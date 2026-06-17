"""Tests for the custom Healthcare NL API (LLM stubbed)"""
from __future__ import annotations

import pytest

from chatbot.healthcare.nl_api import HealthcareNLAPI
from chatbot.model.schemas import HealthcareEntities, HealthcareSummary

def test_extract_entities_returns_structured_response(stub_llm):
    api = HealthcareNLAPI(llm=stub_llm)
    parsed = api.extract_entities("Patient with hypertension on lisinopril 10mg daily.")
    assert isinstance(parsed, HealthcareEntities)
    types = {e.type for e in parsed.entities}
    assert "Condition" in types
    assert "Medication" in types
    assert parsed.icd10_suggestions[0].code == "I10"

def test_extract_entities_uses_structured_output(stub_llm):
    api = HealthcareNLAPI(llm=stub_llm)
    api.extract_entities("Patient with hypertension on lisinopril.")
    schemas = [c["schema"] for c in stub_llm.calls]
    assert schemas == ["HealthcareEntities"]

def test_summarize_text_default_clinician(stub_llm):
    api = HealthcareNLAPI(llm=stub_llm)
    summary = api.summarize_text("Long clinical note ...")
    assert isinstance(summary, HealthcareSummary)
    assert summary.audience == "clinician"
    assert len(summary.bullet_points) >= 1

def test_summarize_text_patient_audience(stub_llm, stub_llm_payloads):
    stub_llm_payloads["HealthcareSummary"]["audience"] = "patient"
    api = HealthcareNLAPI(llm=stub_llm)
    summary = api.summarize_text("note", audience="patient")
    assert summary.audience == "patient"

def test_summarize_text_invalid_audience_defaults_to_clinician(stub_llm):
    api = HealthcareNLAPI(llm=stub_llm)
    summary = api.summarize_text("note", audience="alien")
    assert summary.audience == "clinician"

def test_link_icd10_returns_suggestions(stub_llm):
    api = HealthcareNLAPI(llm=stub_llm)
    suggestions = api.link_icd10("hypertension")
    assert suggestions and suggestions[0].code == "I10"

def test_link_icd10_empty_string_returns_empty_list(stub_llm):
    api = HealthcareNLAPI(llm=stub_llm)
    assert api.link_icd10("   ") == []

def test_truncates_long_input(stub_llm, caplog):
    api = HealthcareNLAPI(llm=stub_llm, max_input_chars=20)
    with caplog.at_level("WARNING"):
        api.extract_entities("x" * 500)
    assert any("Truncating Healthcare NL input" in r.getMessage() for r in caplog.records)

def test_empty_input_raises(stub_llm):
    api = HealthcareNLAPI(llm=stub_llm)
    with pytest.raises(ValueError):
        api.extract_entities("")

def test_requires_llm():
    with pytest.raises(ValueError):
        HealthcareNLAPI(llm=None)

def test_helpers_build_entity_and_relation():
    e = HealthcareNLAPI.make_entity("aspirin", "Medication", confidence=0.95)
    assert e.type == "Medication" and e.confidence == 0.95
    r = HealthcareNLAPI.make_relation("aspirin", "treats", "pain")
    assert r.relation == "treats"

def test_empty_response_is_valid_schema():
    blank = HealthcareNLAPI.empty_response()
    assert isinstance(blank, HealthcareEntities)
    assert blank.entities == []