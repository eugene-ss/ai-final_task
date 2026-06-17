"""Healthcare NL MCP server (Healthcare NL API tool).

Spawned as a subprocess by the orchestrator. Exposes the Healthcare NL
API required by focus area B of the ai-final brief. The healthcare agent uses
these tools inside an explicit ReAct loop.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from chatbot.healthcare.nl_api import HealthcareNLAPI  # noqa: E402
from chatbot.security.guardrails import validate_tool_output  # noqa: E402
from chatbot.settings.app_config import ConfigManager  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("healthcare_server")

_cfg = ConfigManager()

def _build_llm():
    """Best-effort Azure / OpenAI chat client for the NL API."""

    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    llm_cfg = _cfg.get_llm_config()
    if llm_cfg.api_version:
        return AzureChatOpenAI(
            azure_endpoint=_cfg.endpoint_url,
            api_key=_cfg.api_key,
            azure_deployment=llm_cfg.deployment_name or llm_cfg.model,
            api_version=llm_cfg.api_version,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
        )
    return ChatOpenAI(
        api_key=_cfg.api_key,
        base_url=_cfg.endpoint_url or None,
        model=llm_cfg.model,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
    )

_api = HealthcareNLAPI(
    llm=_build_llm(),
    max_input_chars=_cfg.app_settings.healthcare.max_input_chars,
    entity_types=_cfg.app_settings.healthcare.entity_types,
)

mcp = FastMCP("healthcare")

def _pack(tool_name: str, payload: object) -> str:
    text = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, default=str, indent=2)
    )
    return validate_tool_output(f"healthcare.{tool_name}", text)

def _error_payload(code: str, message: str, retryable: bool = False) -> dict:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }

@mcp.tool()
async def extract_medical_entities(text: str) -> str:
    """Extract structured medical entities, relations, risk factors, and ICD-10 hints.

    Args:
        text: free-form clinical or biomedical text. Treated as untrusted data.
    """

    try:
        parsed = _api.extract_entities(text)
    except ValueError as exc:
        return _pack(
            "extract_medical_entities",
            _error_payload("INVALID_ARGUMENT", str(exc)),
        )
    except Exception:
        logger.exception("extract_medical_entities failed")
        return _pack(
            "extract_medical_entities",
            _error_payload("INTERNAL_ERROR", "Entity extraction failed."),
        )
    return _pack("extract_medical_entities", parsed.model_dump(exclude_none=True))

@mcp.tool()
async def summarize_clinical_text(text: str, audience: str = "clinician") -> str:
    """Summarise clinical text for the given audience ("clinician" or "patient")."""

    try:
        summary = _api.summarize_text(text, audience)
    except ValueError as exc:
        return _pack(
            "summarize_clinical_text",
            _error_payload("INVALID_ARGUMENT", str(exc)),
        )
    except Exception:
        logger.exception("summarize_clinical_text failed")
        return _pack(
            "summarize_clinical_text",
            _error_payload("INTERNAL_ERROR", "Clinical summary generation failed."),
        )
    return _pack("summarize_clinical_text", summary.model_dump(exclude_none=True))

@mcp.tool()
async def link_to_icd10(entity: str) -> str:
    """Suggest up to 3 ICD-10 codes for a single condition string."""

    try:
        suggestions = _api.link_icd10(entity)
    except Exception:
        logger.exception("link_to_icd10 failed")
        return _pack(
            "link_to_icd10",
            _error_payload("INTERNAL_ERROR", "ICD-10 linking failed."),
        )
    payload = {"entity": entity, "suggestions": [s.model_dump() for s in suggestions]}
    return _pack("link_to_icd10", payload)

if __name__ == "__main__":
    mcp.run()
