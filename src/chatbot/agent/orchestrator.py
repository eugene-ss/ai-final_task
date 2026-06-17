"""Triage orchestrator: routes user questions to RAG, Disaster or Healthcare specialists.

Each specialist is backed by its own MCP stdio server. The triage agent uses
the OpenAI Agents SDK ``Runner`` loop (implicit ReAct under the hood); the
healthcare specialist prompt enforces an *explicit* ReAct pattern so
reasoning steps are auditable.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

from agents import (
    Agent,
    ModelSettings,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
)
from agents.mcp import MCPServerStdio
from openai import AsyncAzureOpenAI, AsyncOpenAI

from chatbot.rag.prompts.prompt_manager import PromptManager
from chatbot.security import (
    init_guardrails,
    validate_llm_output,
    validate_message_history,
    validate_user_input,
)
from chatbot.settings.app_config import (
    AgentSpec,
    AppConfig,
    PROJECT_ROOT,
    load_config,
)

logger = logging.getLogger(__name__)

_python_exe = sys.executable
_openai_client_configured = False

def _first_nonempty_env(*names: str) -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    raise RuntimeError(
        f"Required environment variable is missing. Set one of: {', '.join(names)}."
    )

def _configure_openai_client(cfg: AppConfig) -> None:
    global _openai_client_configured
    if _openai_client_configured:
        return

    api_key = _first_nonempty_env("OPENAI_API_KEY", "API_KEY", "AZURE_OPENAI_API_KEY")
    timeout = cfg.llm.request_timeout_s
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("ENDPOINT_URL")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or ""
    ).strip()
    api_version = (cfg.llm.api_version or "").strip()

    if api_version:
        if not base_url:
            raise RuntimeError(
                "OPENAI_BASE_URL is required when llm.api_version is set in app_config.yaml"
            )
        deployment = (cfg.llm.deployment_name or cfg.llm.model).strip()
        client = AsyncAzureOpenAI(
            azure_endpoint=base_url.rstrip("/"),
            azure_deployment=deployment,
            api_version=api_version,
            api_key=api_key,
            timeout=timeout,
        )
    else:
        kwargs: dict = {"timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncOpenAI(**kwargs)

    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    _openai_client_configured = True

def _model_settings(cfg: AppConfig) -> ModelSettings:
    return ModelSettings(temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens)

def _build_mcp_server(name: str, script_path: str, cfg: AppConfig) -> MCPServerStdio:
    abs_script = cfg.project_path(script_path)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT / "src"))
    return MCPServerStdio(
        name=name,
        params={"command": _python_exe, "args": [str(abs_script)], "env": env},
        client_session_timeout_seconds=cfg.mcp.startup_timeout_s,
    )

def _create_specialist(
    cfg: AppConfig,
    spec: AgentSpec,
    mcp_server: MCPServerStdio,
    prompt_manager: PromptManager,
) -> Agent:
    instructions = prompt_manager.get_system(spec.prompt_file)
    if not instructions:
        instructions = f"You are the {spec.name}. Use your MCP tools to answer the user."
    return Agent(
        name=spec.name,
        model=cfg.llm.model,
        model_settings=_model_settings(cfg),
        instructions=instructions,
        mcp_servers=[mcp_server],
    )

def _create_triage_agent(
    cfg: AppConfig,
    prompt_manager: PromptManager,
    handoffs: list[Agent],
) -> Agent:
    spec = cfg.agents.triage
    instructions = prompt_manager.get_system(spec.prompt_file)
    if not instructions:
        instructions = (
            "You are a triage router. Hand off to the right specialist agent."
        )
    return Agent(
        name=spec.name,
        model=cfg.llm.model,
        model_settings=_model_settings(cfg),
        instructions=instructions,
        handoffs=handoffs,
    )

class AgentSession:
    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        self.cfg = cfg or load_config()
        init_guardrails(self.cfg)
        _configure_openai_client(self.cfg)
        self._stack: Optional[AsyncExitStack] = None
        self._triage: Optional[Agent] = None
        self._prompts = PromptManager(prompts_dir=str(PROJECT_ROOT / "prompts"))

    async def __aenter__(self) -> "AgentSession":
        self._stack = AsyncExitStack()
        rag_server = await self._stack.enter_async_context(
            _build_mcp_server("RAG MCP Server", self.cfg.mcp.rag_server, self.cfg)
        )
        disaster_server = await self._stack.enter_async_context(
            _build_mcp_server(
                "Disaster MCP Server", self.cfg.mcp.disaster_server, self.cfg
            )
        )
        healthcare_server = await self._stack.enter_async_context(
            _build_mcp_server(
                "Healthcare MCP Server", self.cfg.mcp.healthcare_server, self.cfg
            )
        )

        rag_agent = _create_specialist(
            self.cfg, self.cfg.agents.rag, rag_server, self._prompts
        )
        disaster_agent = _create_specialist(
            self.cfg, self.cfg.agents.disaster, disaster_server, self._prompts
        )
        healthcare_agent = _create_specialist(
            self.cfg, self.cfg.agents.healthcare, healthcare_server, self._prompts
        )
        self._triage = _create_triage_agent(
            self.cfg,
            self._prompts,
            handoffs=[rag_agent, disaster_agent, healthcare_agent],
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._triage = None

    async def run(
        self,
        query: str,
        message_history: Optional[list] = None,
    ) -> tuple[str, list]:
        if self._triage is None:
            raise RuntimeError("AgentSession must be used as an async context manager")
        safe_query = validate_user_input(query)
        input_messages = list(message_history) if message_history else []
        input_messages.append({"role": "user", "content": safe_query})
        result = await Runner.run(
            self._triage,
            input_messages,
            max_turns=self.cfg.agents.max_turns,
        )
        safe_output = validate_llm_output(result.final_output)
        safe_history = validate_message_history(result.to_input_list())
        return safe_output, safe_history

async def run_query(
    query: str,
    message_history: Optional[list] = None,
) -> tuple[str, list]:
    async with AgentSession() as session:
        return await session.run(query, message_history)