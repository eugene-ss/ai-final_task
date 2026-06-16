"""Shared pytest fixtures for unit + integration tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("API_KEY", "test-key-not-real")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.invalid")
os.environ.setdefault("ENDPOINT_URL", "https://example.invalid")
os.environ.setdefault("BM25_HMAC_KEY", "test-hmac-key")

@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT

@pytest.fixture
def fixtures_dir() -> Path:
    return PROJECT_ROOT / "tests" / "fixtures"

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dataset"
    d.mkdir()
    return d

@pytest.fixture
def app_config():
    from chatbot.settings.app_config import load_config

    return load_config()

@pytest.fixture
def config_manager(tmp_path: Path, monkeypatch):
    """A ConfigManager pointed at temporary directories so tests cannot
    pollute the real ``vector-db`` / ``results`` folders."""

    monkeypatch.setenv("VECTOR_DB_DIR", str(tmp_path / "vector-db"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    from chatbot.settings.app_config import ConfigManager, EnvironmentSettings

    env = EnvironmentSettings()
    return ConfigManager(env_settings=env)

@pytest.fixture
def guardrails_config():
    from chatbot.settings.app_config import GuardrailsConfig

    return GuardrailsConfig(
        enabled=True,
        require_user=False,
        max_user_input_chars=200,
        max_llm_output_chars=2000,
        max_tool_output_chars=5000,
        max_history_chars=10000,
        blocked_patterns=[r"(?i)ignore previous instructions"],
    )

@pytest.fixture
def small_disasters_repo(fixtures_dir: Path):
    from chatbot.disasters.repository import DisasterRepository

    return DisasterRepository(
        data_dir=fixtures_dir,
        csv_files=["small_disasters.csv"],
        max_limit=100,
    )

class _StubStructured:
    def __init__(self, schema, payload: Dict[str, Any]) -> None:
        self._schema = schema
        self._payload = payload

    def invoke(self, _messages):
        return self._schema.model_validate(self._payload)

class StubLLM:
    """Simple stand-in for an LLM client used by HealthcareNLAPI tests."""

    def __init__(self, payloads_by_schema: Dict[str, Dict[str, Any]]) -> None:
        self._payloads = payloads_by_schema
        self.calls: List[Dict[str, Any]] = []

    def with_structured_output(self, schema):
        payload = self._payloads.get(schema.__name__, {})
        self.calls.append({"schema": schema.__name__, "payload": payload})
        return _StubStructured(schema, payload)

    def invoke(self, messages):
        raise RuntimeError("StubLLM does not support raw invoke")

@pytest.fixture
def stub_llm_payloads() -> Dict[str, Dict[str, Any]]:
    return {
        "HealthcareEntities": {
            "entities": [
                {"text": "hypertension", "type": "Condition", "confidence": 0.95},
                {"text": "lisinopril", "type": "Medication", "confidence": 0.93, "normalized": "lisinopril"},
                {"text": "10mg daily", "type": "Dosage", "confidence": 0.9},
            ],
            "relations": [
                {"subject": "lisinopril", "relation": "treats", "object": "hypertension"},
                {"subject": "10mg daily", "relation": "dosage_of", "object": "lisinopril"},
            ],
            "risk_factors": ["family history"],
            "icd10_suggestions": [
                {
                    "code": "I10",
                    "label": "Essential (primary) hypertension",
                    "confidence": 0.82,
                    "rationale": "Standard ICD-10 code for primary hypertension.",
                }
            ],
        },
        "HealthcareSummary": {
            "audience": "clinician",
            "summary": "Patient with hypertension on lisinopril 10mg daily; family history noted.",
            "bullet_points": [
                "Diagnosis: hypertension",
                "Medication: lisinopril 10mg daily",
                "Risk factor: family history",
            ],
        },
    }

@pytest.fixture
def stub_llm(stub_llm_payloads: Dict[str, Dict[str, Any]]) -> StubLLM:
    return StubLLM(stub_llm_payloads)
