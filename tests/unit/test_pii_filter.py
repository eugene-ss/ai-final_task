"""Tests for the standalone PII filter helpers."""
from __future__ import annotations

from chatbot.security.pii_filter import (
    detect_pii,
    has_pii,
    redact_pii,
    sanitize_for_logging,
)

def test_detect_pii_email():
    findings = detect_pii("Email me at alice@example.com please")
    assert findings.get("email") == ["alice@example.com"]

def test_detect_pii_phone():
    findings = detect_pii("call +1 555-123-4567 today")
    assert "phone" in findings

def test_detect_pii_returns_empty_for_clean_text():
    assert detect_pii("just a plain sentence") == {}

def test_redact_pii_replaces_email_and_phone():
    text = "alice@example.com or +1 555-123-4567"
    out = redact_pii(text)
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" in out
    assert "@" not in out

def test_redact_pii_specific_pattern_only():
    text = "alice@example.com and +1 555-123-4567"
    out = redact_pii(text, patterns=["email"])
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" not in out

def test_has_pii():
    assert has_pii("Reach me at me@you.org") is True
    assert has_pii("hello world") is False

def test_sanitize_for_logging_strips_ctrl_chars_and_truncates():
    text = "alice@example.com\x01\x02" + "x" * 1000
    out = sanitize_for_logging(text, max_len=50)
    assert len(out) <= 50
    assert "\x01" not in out
    assert "[REDACTED_EMAIL]" in out

def test_redact_handles_empty_text():
    assert redact_pii("") == ""
    assert redact_pii(None) is None

def test_detect_pii_ssn():
    findings = detect_pii("SSN 123-45-6789", patterns=["ssn"])
    assert findings.get("ssn") == ["123-45-6789"]

def test_redact_credit_card():
    out = redact_pii("Card 4111 1111 1111 1111 expires soon", patterns=["credit_card"])
    assert "[REDACTED_CC]" in out
