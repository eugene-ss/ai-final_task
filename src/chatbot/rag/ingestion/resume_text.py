"""Resume text normalization and metadata extraction utilities."""
from __future__ import annotations

import re
from typing import List, Optional

def normalize_resume_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()

def extract_headline(text: str, max_len: int = 160) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 3:
            return line[:max_len]
    return None

def extract_skills_line(text: str, max_skills: int = 40) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    idx = lower.find("skills")
    if idx < 0:
        return None
    window = text[idx : idx + 2500]
    parts = window.split("\n", 1)
    body = parts[1] if len(parts) > 1 else window
    chunk_lines: List[str] = []
    stop_markers = (
        "experience",
        "education",
        "employment",
        "work history",
        "professional experience",
        "summary",
        "objective",
    )
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            if chunk_lines:
                break
            continue
        low = line.lower()
        if any(low.startswith(m) or low.startswith(m + ":") for m in stop_markers):
            break
        chunk_lines.append(line)
        if len(chunk_lines) >= 8:
            break
    if not chunk_lines:
        return None
    blob = " ".join(chunk_lines)
    tokens = re.split(r"[,;|\u2022\n]+", blob)
    skills = [t.strip() for t in tokens if len(t.strip()) > 1][:max_skills]
    if not skills:
        return None
    return ", ".join(skills)

def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_+#]+", (text or "").lower())
