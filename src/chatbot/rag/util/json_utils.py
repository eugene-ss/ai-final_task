from __future__ import annotations

import json
from typing import Any

def strip_markdown_code_fence(raw: str) -> str:
    s = (raw or "").strip()
    if not s.startswith("```"):
        return s
    parts = s.split("\n", 1)
    s = parts[1] if len(parts) > 1 else s
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3].strip()
    return s

def loads_json_stripped(raw: str) -> Any:
    return json.loads(strip_markdown_code_fence(raw))
