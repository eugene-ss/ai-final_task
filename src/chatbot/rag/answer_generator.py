"""Grounded answer generation using retrieved context.

Prompts now live under ``prompts/rag_answer.md`` (loaded via ``PromptManager``)
instead of being inlined here.
"""
from __future__ import annotations

import json
import logging
from textwrap import dedent
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from chatbot.model.schemas import (
    Permission,
    RAGStructuredAnswer,
    SearchResult,
    User,
)
from chatbot.rag.prompts.prompt_manager import PromptManager
from chatbot.rag.security_facade import AccessControl
from chatbot.rag.util.json_utils import loads_json_stripped
from chatbot.security.guardrails import Guardrails, GuardrailViolation
from chatbot.settings.app_config import ConfigManager

logger = logging.getLogger(__name__)


OUTPUT_SPEC_STRUCTURED = dedent(
    """
    Output format: respond with a JSON object with keys:
    - summary       : a concise narrative answer grounded in the excerpts.
    - candidates    : array of supporting events, each with
                      { doc_id, category, evidence_snippet, relevance_note }.
                      ``doc_id`` is the EM-DAT identifier (e.g. ``1995-0010-JPN``)
                      taken from the excerpt header; ``category`` is the
                      disaster type.
    - confidence    : low | medium | high.
    """
).strip()

OUTPUT_SPEC_PROSE = dedent(
    """
    Provide a detailed, factual response that cites the specific events
    (by EM-DAT id and year) drawn from the excerpts above.
    """
).strip()


class AnswerGenerator:
    def __init__(
        self,
        llm,
        config: ConfigManager,
        prompt_manager: PromptManager,
        guardrails: Optional[Guardrails] = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.prompt_manager = prompt_manager
        self._guardrails = guardrails or Guardrails(config)

    def _resume_answer_messages(
        self,
        query: str,
        documents_text: str,
        output_spec: str,
    ) -> List[BaseMessage]:
        return self.prompt_manager.get_messages(
            "rag_answer",
            query=query,
            documents=documents_text,
            output_spec=output_spec,
        )

    def _system_prompt(self) -> str:
        return self.prompt_manager.get_system("rag_answer")

    @staticmethod
    def build_context_budget(
        retrieved_docs: List[SearchResult],
        max_chars: int,
    ) -> str:
        parts: List[str] = []
        used = 0
        for sr in retrieved_docs:
            meta = sr.document.metadata
            rid = getattr(meta, "id", "")
            cat = getattr(meta, "category", "")
            headline = getattr(meta, "headline", None) or ""
            header = f"--- resume_id={rid} category={cat}"
            if headline:
                header += f" headline={headline[:80]}"
            header += " ---\n"
            allowance = max_chars - used - len(header) - 4
            if allowance < 120:
                break
            body = (sr.document.page_content or "")[:allowance]
            block = header + body
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[SearchResult],
        user: Optional[User],
        access_control: AccessControl,
    ) -> str:
        self._guardrails.enforce_user(user, "generate_answer")

        if not query or not query.strip():
            return "Invalid query provided"

        try:
            validated_query = self._guardrails.validate_user_input(query)
        except GuardrailViolation as exc:
            logger.warning("Query blocked by guardrails: %s", exc)
            return exc.user_message

        if user and not isinstance(user, User):
            return "Invalid user"
        if user and not access_control.check_permission(user, Permission.READ):
            return "Access denied: Insufficient permissions"
        if not retrieved_docs:
            return "No relevant documents found to generate an answer"

        so = self.config.app_settings.structured_output
        documents_text = self.build_context_budget(retrieved_docs, so.max_context_chars)

        if so.enabled:
            messages = self._resume_answer_messages(
                validated_query, documents_text, OUTPUT_SPEC_STRUCTURED
            )
            try:
                structured_llm = self.llm.with_structured_output(RAGStructuredAnswer)
                parsed = structured_llm.invoke(messages)
                if isinstance(parsed, RAGStructuredAnswer):
                    out = parsed.model_dump_json(indent=2)
                else:
                    out = json.dumps(parsed, indent=2)
            except Exception as exc:
                logger.warning("Structured output failed (%s); JSON fallback", exc)
                fb_msgs: List[BaseMessage] = [
                    SystemMessage(content=self._system_prompt()),
                    HumanMessage(
                        content=self.prompt_manager.get_prompt(
                            "rag_answer",
                            query=validated_query,
                            documents=documents_text,
                            output_spec=(
                                OUTPUT_SPEC_STRUCTURED
                                + "\n\nReturn a single JSON object with those keys."
                            ),
                        )
                    ),
                ]
                response = self.llm.invoke(fb_msgs)
                raw = (response.content or "").strip()
                data = loads_json_stripped(raw)
                out = RAGStructuredAnswer.model_validate(data).model_dump_json(indent=2)
        else:
            messages = self._resume_answer_messages(
                validated_query, documents_text, OUTPUT_SPEC_PROSE
            )
            response = self.llm.invoke(messages)
            out = response.content or ""

        out = self._guardrails.scan_output_for_pii(out)
        out = self._guardrails.check_prompt_leak(out, self._system_prompt())

        if user:
            access_control.log_access(
                user,
                "generate_answer",
                self._guardrails.sanitize_for_log(query, 200),
                True,
            )

        return out
