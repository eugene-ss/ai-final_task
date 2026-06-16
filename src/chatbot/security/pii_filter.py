"""PII detection and redaction utilities (vendored from ai-rag_task)."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b\+?\d[\d\s\-().]{8,}\d\b")
_SSN_RE = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

_PATTERN_MAP: Dict[str, re.Pattern[str]] = {
    "email": _EMAIL_RE,
    "phone": _PHONE_RE,
    "ssn": _SSN_RE,
    "credit_card": _CC_RE,
    "ip_address": _IP_RE,
}

_REDACTION_MAP: Dict[str, str] = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "ssn": "[REDACTED_SSN]",
    "credit_card": "[REDACTED_CC]",
    "ip_address": "[REDACTED_IP]",
}

def detect_pii(text: str, patterns: Optional[List[str]] = None) -> Dict[str, List[str]]:
    if not text:
        return {}
    active = patterns or list(_PATTERN_MAP.keys())
    findings: Dict[str, List[str]] = {}
    for name in active:
        regex = _PATTERN_MAP.get(name)
        if regex:
            matches = regex.findall(text)
            if matches:
                findings[name] = matches
    return findings

def redact_pii(text: str, patterns: Optional[List[str]] = None) -> str:
    if not text:
        return text
    active = patterns or list(_PATTERN_MAP.keys())
    result = text
    for name in active:
        regex = _PATTERN_MAP.get(name)
        replacement = _REDACTION_MAP.get(name, "[REDACTED]")
        if regex:
            result = regex.sub(replacement, result)
    return result

def has_pii(text: str, patterns: Optional[List[str]] = None) -> bool:
    return bool(detect_pii(text, patterns))

def sanitize_for_logging(
    text: str,
    max_len: int = 500,
    patterns: Optional[List[str]] = None,
) -> str:
    redacted = redact_pii(text, patterns)
    ctrl = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    clean = ctrl.sub("", redacted)
    return clean[:max_len]
