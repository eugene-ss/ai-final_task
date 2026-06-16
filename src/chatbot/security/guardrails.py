"""Centralized security guardrails for input / output / tool validation.

Merges the RAG-style stateful ``Guardrails`` class with the agentic-task
module-level helpers (``validate_user_input``, ``validate_tool_output`` etc.)
so both layers can share one configuration.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b\+?\d[\d\s\-().]{8,}\d\b")
_SSN_RE = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")

@dataclass
class GuardrailViolation(RuntimeError):
    stage: str
    code: str
    reason: str
    user_message: str = (
        "Your request was blocked by security checks. "
        "Please rephrase and remove potentially unsafe content."
    )

    def __str__(self) -> str:
        return f"GUARDRAIL[{self.stage}:{self.code}] {self.reason}"

class Guardrails:
    """Configurable validator. Pass a :class:`GuardrailsConfig` or a
    ``ConfigManager`` / ``AppConfig`` and the right sub-config is picked up."""

    def __init__(self, config: Any) -> None:
        from chatbot.settings.app_config import (
            AppConfig,
            ConfigManager,
            GuardrailsConfig,
        )

        if isinstance(config, GuardrailsConfig):
            self._cfg = config
        elif isinstance(config, AppConfig):
            self._cfg = config.guardrails
        elif isinstance(config, ConfigManager):
            self._cfg = config.app_settings.guardrails
        elif hasattr(config, "app_settings"):
            self._cfg = config.app_settings.guardrails
        elif hasattr(config, "guardrails"):
            self._cfg = config.guardrails
        else:
            raise TypeError(
                f"Cannot derive GuardrailsConfig from {type(config).__name__}"
            )
        self._compiled: Optional[List[re.Pattern[str]]] = None

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def require_user(self) -> bool:
        return self._cfg.require_user

    def _patterns(self) -> List[re.Pattern[str]]:
        if self._compiled is None:
            self._compiled = [re.compile(p) for p in self._cfg.blocked_patterns]
        return self._compiled

    def sanitize_for_log(self, text: str, max_chars: int = 500) -> str:
        out = text or ""
        if self._cfg.pii_redaction.enabled:
            out = _EMAIL_RE.sub("[REDACTED_EMAIL]", out)
            out = _PHONE_RE.sub("[REDACTED_PHONE]", out)
            if "ssn" in self._cfg.pii_redaction.patterns:
                out = _SSN_RE.sub("[REDACTED_SSN]", out)
        out = _CTRL_CHARS.sub("", out)
        return out[:max_chars]

    def redact_pii(self, text: str) -> str:
        out = text or ""
        if not self._cfg.pii_redaction.enabled:
            return out
        pats = self._cfg.pii_redaction.patterns
        if "email" in pats:
            out = _EMAIL_RE.sub("[REDACTED_EMAIL]", out)
        if "phone" in pats:
            out = _PHONE_RE.sub("[REDACTED_PHONE]", out)
        if "ssn" in pats:
            out = _SSN_RE.sub("[REDACTED_SSN]", out)
        return out

    @staticmethod
    def _violate(stage: str, code: str, reason: str) -> None:
        raise GuardrailViolation(stage=stage, code=code, reason=reason)

    def _check_blocked_patterns(self, stage: str, text: str) -> None:
        for pattern in self._patterns():
            if pattern.search(text):
                logger.warning("Blocked pattern matched in %s", stage)
                self._violate(stage, "blocked_pattern", f"pattern matched: {pattern.pattern}")

    def _check_text(
        self,
        stage: str,
        text: str,
        max_chars: int,
        check_patterns: bool = True,
    ) -> str:
        value = (text or "").strip()
        if not value:
            self._violate(stage, "empty_input", "text is empty")
        if len(value) > max_chars:
            self._violate(stage, "too_long", f"length {len(value)} exceeds {max_chars}")
        if _CTRL_CHARS.search(value):
            self._violate(stage, "control_chars", "contains control characters")
        if check_patterns:
            self._check_blocked_patterns(stage, value)
        return value

    def validate_user_input(self, text: str) -> str:
        if not self._cfg.enabled:
            return text
        return self._check_text("user_input", text, self._cfg.max_user_input_chars)

    def validate_llm_output(self, text: str) -> str:
        if not self._cfg.enabled:
            return text
        return self._check_text(
            "llm_output", text, self._cfg.max_llm_output_chars, check_patterns=False
        )

    def validate_tool_output(self, tool_name: str, payload: str) -> str:
        if not self._cfg.enabled:
            return payload
        return self._check_text(
            f"tool_output:{tool_name}",
            payload,
            self._cfg.max_tool_output_chars,
            check_patterns=False,
        )

    def validate_context(self, text: str) -> str:
        if not self._cfg.enabled:
            return text
        return self._check_text(
            "context", text, self._cfg.max_context_chars, check_patterns=False
        )

    def validate_message_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._cfg.enabled:
            return history
        total = 0
        out: List[Dict[str, Any]] = []
        for msg in history:
            clone = dict(msg)
            content = clone.get("content")
            if isinstance(content, str):
                safe = _CTRL_CHARS.sub("", content)
                total += len(safe)
                clone["content"] = safe
            out.append(clone)
        if total > self._cfg.max_history_chars:
            self._violate(
                "message_history",
                "too_long",
                f"history chars {total} exceed cap {self._cfg.max_history_chars}",
            )
        return out

    def validate_json_write(self, name: str, payload: Any) -> None:
        if not self._cfg.enabled:
            return
        as_text = json.dumps(payload, ensure_ascii=False, default=str)
        max_chars = getattr(self._cfg, "max_json_write_chars", self._cfg.max_context_chars)
        self._check_text(
            f"json_write:{name}",
            as_text,
            max_chars,
            check_patterns=False,
        )

    def validate_url(self, url: str, stage: str = "url") -> None:
        if not self._cfg.enabled:
            return
        parsed = urlparse(url or "")
        if parsed.scheme not in set(self._cfg.allowed_url_schemes):
            self._violate(stage, "invalid_scheme", f"scheme={parsed.scheme}")

    def scan_output_for_pii(self, text: str) -> str:
        if not self._cfg.enabled or not self._cfg.pii_redaction.enabled:
            return text
        return self.redact_pii(text)

    def check_prompt_leak(self, output: str, system_prompt: str) -> str:
        if not self._cfg.enabled:
            return output
        no = output.lower().strip()
        np_ = system_prompt.lower().strip()
        if len(np_) > 30 and np_ in no:
            logger.warning("Prompt leak detected in LLM output")
            return "[Response redacted: potential prompt leak detected]"
        return output

    def enforce_user(self, user: Any, operation: str) -> None:
        if not self._cfg.enabled or not self._cfg.require_user:
            return
        if user is None:
            raise PermissionError(
                f"Operation '{operation}' requires an authenticated user."
            )

_default_instance: Optional[Guardrails] = None

def init_guardrails(config: Any) -> Guardrails:
    """Install the process-wide default guardrails instance."""

    global _default_instance
    _default_instance = Guardrails(config)
    return _default_instance

def get_guardrails() -> Guardrails:
    """Return the default guardrails instance, lazily creating one."""

    global _default_instance
    if _default_instance is None:
        from chatbot.settings.app_config import GuardrailsConfig

        _default_instance = Guardrails(GuardrailsConfig())
    return _default_instance

def sanitize_for_log(text: str, max_chars: int = 500) -> str:
    return get_guardrails().sanitize_for_log(text, max_chars)

def validate_user_input(text: str) -> str:
    return get_guardrails().validate_user_input(text)

def validate_llm_output(text: str) -> str:
    return get_guardrails().validate_llm_output(text)

def validate_tool_output(tool_name: str, payload: str) -> str:
    return get_guardrails().validate_tool_output(tool_name, payload)

def validate_message_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return get_guardrails().validate_message_history(history)

def validate_json_write(name: str, payload: Any) -> None:
    return get_guardrails().validate_json_write(name, payload)

def validate_url(url: str, stage: str = "url") -> None:
    return get_guardrails().validate_url(url, stage)
