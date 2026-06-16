"""Interactive REPL for the ai-final chatbot.

Loads ``.env`` + ``config/app_config.yaml``, installs OpenTelemetry tracing, and
starts the triage agent session. Slash commands:

* ``/help``       - list commands
* ``/quit``       - exit
* ``/clear``      - reset the conversation history
* ``/reindex``    - trigger an MCP reindex via the RAG agent (asks the user
                   to type a natural-language request internally so the agent
                   tool call is still visible in the trace)
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv


def _bootstrap() -> None:
    src_dir = Path(__file__).resolve().parent.parent
    project_root = src_dir.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    load_dotenv(project_root / ".env", override=True)


def _print_banner(model: str, tracing: bool) -> None:
    print("=" * 64)
    print("  AI-Final Chatbot - RAG | Disasters | Healthcare NL")
    print(f"  Model: {model}")
    print("  Slash commands: /help /quit /clear /reindex")
    if tracing:
        print("  (OpenTelemetry tracing enabled - spans printed to console)")
    print("=" * 64)


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  /help     show this message\n"
        "  /quit     exit the chatbot (also: quit, exit, q)\n"
        "  /clear    reset the in-memory conversation history\n"
        "  /reindex  ask the RAG agent to (re)ingest the corpus\n"
        "\nExample questions:\n"
        "  - Find Python developers with cloud experience  (RAG)\n"
        "  - How many earthquakes hit Japan between 1990 and 2010?  (Disasters)\n"
        "  - Extract medications and conditions from this note: ...  (Healthcare)\n"
    )


async def _interactive_loop() -> None:
    from chatbot.agent.orchestrator import AgentSession
    from chatbot.agent.tracing import install_tracing
    from chatbot.security import GuardrailViolation, init_guardrails
    from chatbot.settings.app_config import load_config

    cfg = load_config()
    init_guardrails(cfg)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    install_tracing(cfg.tracing)

    _print_banner(cfg.llm.model, cfg.tracing.enabled)

    message_history: list = []

    async with AgentSession(cfg) as session:
        while True:
            try:
                query = await asyncio.to_thread(input, "\nYou: ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return

            query = query.strip()
            if not query:
                continue
            if query.lower() in ("/quit", "quit", "exit", "q"):
                print("Goodbye!")
                return
            if query.lower() in ("/help", "help", "?"):
                _print_help()
                continue
            if query.lower() == "/clear":
                message_history = []
                print("(conversation history cleared)")
                continue
            if query.lower() == "/reindex":
                query = (
                    "Please ingest the document corpus by calling "
                    "`ingest_corpus(force=true)` and report how many chunks were loaded."
                )

            print("\nAssistant: thinking...", end="", flush=True)
            try:
                response, message_history = await session.run(query, message_history)
                print(f"\rAssistant: {response}")
            except GuardrailViolation as exc:
                logging.warning("guardrail blocked request: %s", exc)
                print(f"\rAssistant: {exc.user_message}")
            except Exception as exc:  # noqa: BLE001 - surface to user
                logging.exception("agent run failed")
                print(f"\rAssistant: Error - {exc}")


def main() -> None:
    _bootstrap()
    try:
        asyncio.run(_interactive_loop())
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":  # pragma: no cover
    main()
