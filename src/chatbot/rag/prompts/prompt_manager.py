from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

@dataclass
class PromptBundle:
    system: Optional[str]
    user: str

def _format_template(template: str, kwargs: Dict[str, Any]) -> str:
    if not template:
        return ""
    return template.format(**kwargs)

class PromptManager:
    """Loads ``prompts/*.md`` files with ``## System Message`` / ``## Template`` sections."""

    def __init__(self, prompts_dir: str | Path = "prompts") -> None:
        self.prompts_dir = Path(prompts_dir)
        self._prompts: Dict[str, Dict[str, str]] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        if not self.prompts_dir.is_dir():
            return
        for prompt_file in self.prompts_dir.glob("*.md"):
            name = prompt_file.stem
            with open(prompt_file, encoding="utf-8") as fh:
                content = fh.read()
            system_match = re.search(
                r"## System Message\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL
            )
            template_match = re.search(
                r"## Template\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL
            )
            self._prompts[name] = {
                "system": system_match.group(1).strip() if system_match else "",
                "template": template_match.group(1).strip() if template_match else "",
            }

    def has(self, name: str) -> bool:
        return name in self._prompts

    def list_prompts(self) -> List[str]:
        return sorted(self._prompts.keys())

    def get_system(self, name: str) -> str:
        return self._prompts.get(name, {}).get("system", "")

    def get_template(self, name: str) -> str:
        return self._prompts.get(name, {}).get("template", "")

    def get_bundle(self, name: str, **kwargs: Any) -> PromptBundle:
        p = self._prompts.get(name, {})
        tmpl = p.get("template", "")
        user = _format_template(tmpl, kwargs) if tmpl else ""
        sys_raw = p.get("system", "")
        system: Optional[str] = None
        if sys_raw:
            try:
                system = _format_template(sys_raw, kwargs)
            except KeyError:
                system = sys_raw
        return PromptBundle(system=system, user=user)

    def get_prompt(self, name: str, **kwargs: Any) -> str:
        return self.get_bundle(name, **kwargs).user

    def get_messages(self, name: str, **kwargs: Any) -> List[BaseMessage]:
        bundle = self.get_bundle(name, **kwargs)
        msgs: List[BaseMessage] = []
        if bundle.system:
            msgs.append(SystemMessage(content=bundle.system))
        msgs.append(HumanMessage(content=bundle.user))
        return msgs
