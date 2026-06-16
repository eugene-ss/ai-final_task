"""Tests for resume text normalization helpers."""
from __future__ import annotations

from chatbot.rag.ingestion.resume_text import (
    extract_headline,
    extract_skills_line,
    normalize_resume_text,
    tokenize_for_bm25,
)

def test_normalize_resume_text_collapses_whitespace():
    raw = "  Jane\tDoe\n\n\n   Senior   Engineer  \n  \n  Skilled in Python  \n"
    out = normalize_resume_text(raw)
    assert "Jane Doe" in out
    assert "Senior Engineer" in out
    assert "\n\n" not in out

def test_normalize_resume_text_empty():
    assert normalize_resume_text("") == ""

def test_extract_headline_returns_first_non_trivial_line():
    text = "\n  \n  Senior Data Scientist  \nDetails follow"
    assert extract_headline(text) == "Senior Data Scientist"

def test_extract_headline_truncates_long_line():
    text = "X" * 500
    out = extract_headline(text, max_len=160)
    assert out is not None and len(out) == 160

def test_extract_headline_no_input():
    assert extract_headline("") is None

def test_extract_skills_line_finds_skills():
    text = (
        "Summary\nExperienced engineer\n\nSkills\n"
        "Python, Machine Learning, SQL\n\nExperience\nGoogle, 2020-2024"
    )
    out = extract_skills_line(text)
    assert out is not None
    assert "Python" in out
    assert "SQL" in out

def test_extract_skills_line_returns_none_when_missing():
    assert extract_skills_line("Summary only. No relevant section here.") is None

def test_tokenize_for_bm25_lowercases_and_keeps_alphanum():
    tokens = tokenize_for_bm25("Python 3.11 + C++ developer; Node.js!")
    assert "python" in tokens
    assert "c++" in tokens
    assert "node" in tokens

def test_tokenize_for_bm25_empty():
    assert tokenize_for_bm25("") == []
