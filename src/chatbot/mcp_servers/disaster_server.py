"""Disaster Knowledge MCP server (Pandas-backed).

Spawned as a subprocess by the orchestrator. Exposes filtering, aggregation,
and top-N tools over the EM-DAT natural disaster CSV files configured in
``config/app_config.yaml``.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from chatbot.disasters.repository import DisasterRepository  # noqa: E402
from chatbot.security.guardrails import validate_tool_output  # noqa: E402
from chatbot.settings.app_config import ConfigManager  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("disaster_server")

_cfg = ConfigManager()
_dcfg = _cfg.app_settings.disasters
_repository = DisasterRepository(
    data_dir=_cfg.data_dir,
    csv_files=_dcfg.csv_files,
    max_limit=_dcfg.max_query_limit,
)

mcp = FastMCP("disasters")

def _pack(tool_name: str, payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str, indent=2)
    return validate_tool_output(f"disasters.{tool_name}", text)

def _error_payload(code: str, message: str, retryable: bool = False) -> dict:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }

@mcp.tool()
async def query_disasters(
    disaster_type: Optional[str] = None,
    country: Optional[str] = None,
    region: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 20,
) -> str:
    """List natural disaster events matching the given filters.

    Args:
        disaster_type: e.g. "Flood", "Earthquake", "Drought". Optional.
        country: country name or ISO-3 code. Optional.
        region: e.g. "South America". Optional.
        year_from: inclusive lower bound on event year. Optional.
        year_to: inclusive upper bound on event year. Optional.
        limit: max number of events to return (capped by config; default 20).
    """

    try:
        result = _repository.query(
            disaster_type=disaster_type,
            country=country,
            region=region,
            year_from=year_from,
            year_to=year_to,
            limit=limit or _dcfg.default_query_limit,
        )
    except Exception as exc:
        logger.exception("query_disasters failed")
        return _pack(
            "query_disasters",
            _error_payload("INTERNAL_ERROR", "Disaster query failed."),
        )
    return _pack("query_disasters", result)

@mcp.tool()
async def disaster_stats(
    group_by: str,
    metric: str,
    disaster_type: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    top_n: int = 10,
) -> str:
    """Aggregate disaster events.

    Args:
        group_by: one of "country", "year", "region", "disaster_type", "continent".
        metric: one of "events", "total_deaths", "total_affected", "total_damages_usd".
        disaster_type: optional filter (e.g. "Flood").
        year_from: optional inclusive lower bound on year.
        year_to: optional inclusive upper bound on year.
        top_n: how many top groups to return (default 10).
    """

    try:
        result = _repository.stats(
            group_by=group_by,
            metric=metric,
            disaster_type=disaster_type,
            year_from=year_from,
            year_to=year_to,
            top_n=top_n,
        )
    except ValueError as exc:
        return _pack("disaster_stats", _error_payload("INVALID_ARGUMENT", str(exc)))
    except Exception:
        logger.exception("disaster_stats failed")
        return _pack(
            "disaster_stats",
            _error_payload("INTERNAL_ERROR", "Disaster statistics query failed."),
        )
    return _pack("disaster_stats", result)

@mcp.tool()
async def top_disasters_by_impact(
    metric: str,
    n: int = 10,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> str:
    """Return the n single events with the largest values for the chosen metric.

    Args:
        metric: one of "total_deaths", "total_affected", "total_damages_usd".
        n: how many events to return (default 10).
        year_from: optional inclusive lower bound on year.
        year_to: optional inclusive upper bound on year.
    """

    try:
        result = _repository.top_by_impact(
            metric=metric,
            n=n,
            year_from=year_from,
            year_to=year_to,
        )
    except ValueError as exc:
        return _pack(
            "top_disasters_by_impact",
            _error_payload("INVALID_ARGUMENT", str(exc)),
        )
    except Exception:  # noqa: BLE001
        logger.exception("top_disasters_by_impact failed")
        return _pack(
            "top_disasters_by_impact",
            _error_payload("INTERNAL_ERROR", "Top-impact query failed."),
        )
    return _pack("top_disasters_by_impact", result)

@mcp.tool()
async def list_disaster_types() -> str:
    """List all distinct disaster types present in the dataset."""

    return _pack("list_disaster_types", {"disaster_types": _repository.list_disaster_types()})

@mcp.tool()
async def list_countries() -> str:
    """List all distinct countries present in the dataset."""

    return _pack("list_countries", {"countries": _repository.list_countries()})

if __name__ == "__main__":
    mcp.run()
