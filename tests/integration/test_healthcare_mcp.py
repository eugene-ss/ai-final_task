"""Integration test: spawn the Healthcare MCP server with a stubbed LLM.

To avoid hitting a real LLM, we patch ``HealthcareNLAPI`` inside the spawned
subprocess by setting an environment flag that the server respects. Since the
server constructs its API at import time using ``_build_llm``, we instead spawn
a tiny wrapper script that swaps the API on the module before ``mcp.run()``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

WRAPPER_TEMPLATE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, r"{src}")

    from chatbot.healthcare.nl_api import HealthcareNLAPI
    from chatbot.model.schemas import (
        HealthcareEntities,
        HealthcareEntity,
        HealthcareICD10Suggestion,
        HealthcareSummary,
    )

    class _StubStructured:
        def __init__(self, schema, payload):
            self._schema = schema
            self._payload = payload
        def invoke(self, _msgs):
            return self._schema.model_validate(self._payload)

    class StubLLM:
        def __init__(self):
            self.payloads = {{
                "HealthcareEntities": {{
                    "entities": [
                        {{"text": "hypertension", "type": "Condition", "confidence": 0.9}},
                        {{"text": "lisinopril", "type": "Medication", "confidence": 0.9}},
                    ],
                    "relations": [{{"subject": "lisinopril", "relation": "treats", "object": "hypertension"}}],
                    "risk_factors": [],
                    "icd10_suggestions": [
                        {{"code": "I10", "label": "Essential hypertension", "confidence": 0.8, "rationale": "match"}}
                    ],
                }},
                "HealthcareSummary": {{
                    "audience": "clinician",
                    "summary": "Patient with hypertension on lisinopril.",
                    "bullet_points": ["hypertension", "lisinopril"],
                }},
            }}
        def with_structured_output(self, schema):
            return _StubStructured(schema, self.payloads.get(schema.__name__, {{}}))
        def invoke(self, msgs):
            raise RuntimeError("StubLLM.invoke not implemented")

    import chatbot.mcp_servers.healthcare_server as server
    server._api = HealthcareNLAPI(llm=StubLLM(), max_input_chars=8000)
    server.mcp.run()
    """
)

async def _call_tool(name: str, arguments: dict) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    wrapper = WRAPPER_TEMPLATE.format(src=str(PROJECT_ROOT / "src"))
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(wrapper)
        script_path = fh.name
    try:
        env = dict(os.environ)
        env.setdefault("OPENAI_API_KEY", "test-key")
        env.setdefault("API_KEY", "test-key")
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        params = StdioServerParameters(
            command=sys.executable, args=[script_path], env=env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                payload = result.content[0].text
                return json.loads(payload)
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

async def test_extract_medical_entities_returns_structured_payload():
    payload = await _call_tool(
        "extract_medical_entities",
        {"text": "Patient with hypertension on lisinopril 10mg daily."},
    )
    assert "entities" in payload
    types = {e["type"] for e in payload["entities"]}
    assert "Condition" in types
    assert "Medication" in types
    assert payload["icd10_suggestions"][0]["code"] == "I10"

async def test_summarize_clinical_text_returns_summary():
    payload = await _call_tool(
        "summarize_clinical_text",
        {"text": "Long note about a patient.", "audience": "clinician"},
    )
    assert payload["audience"] == "clinician"
    assert "lisinopril" in payload["summary"]

async def test_link_to_icd10_returns_suggestions():
    payload = await _call_tool("link_to_icd10", {"entity": "hypertension"})
    assert payload["entity"] == "hypertension"
    assert payload["suggestions"][0]["code"] == "I10"

async def test_extract_medical_entities_empty_input_returns_error():
    payload = await _call_tool("extract_medical_entities", {"text": ""})
    assert "error" in payload
