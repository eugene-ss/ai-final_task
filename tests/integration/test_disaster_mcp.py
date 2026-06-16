"""Integration test: spawn the Disaster MCP server as a subprocess and call its tools.

This validates that the FastMCP plumbing + Pandas repository really work together
in the same form the orchestrator uses.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = PROJECT_ROOT / "src" / "chatbot" / "mcp_servers" / "disaster_server.py"


async def _call_tool(name: str, arguments: dict) -> dict:
    """Spawn the disaster server, call a tool, return the parsed JSON payload."""

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    params = StdioServerParameters(command=sys.executable, args=[str(SCRIPT)], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            assert result.content, f"Tool {name} returned no content"
            payload = result.content[0].text
            return json.loads(payload)

async def test_query_disasters_returns_japan_earthquakes():
    payload = await _call_tool(
        "query_disasters",
        {"disaster_type": "Earthquake", "country": "Japan", "limit": 3},
    )
    assert payload["total_matches"] > 0
    assert payload["returned"] <= 3
    assert all(e["disaster_type"] == "Earthquake" for e in payload["events"])
    assert all(e["country"] == "Japan" for e in payload["events"])

async def test_disaster_stats_by_country_total_deaths():
    payload = await _call_tool(
        "disaster_stats",
        {
            "group_by": "country",
            "metric": "total_deaths",
            "year_from": 2000,
            "year_to": 2010,
            "top_n": 5,
        },
    )
    assert payload["metric"] == "total_deaths"
    assert payload["items"]
    # Haiti 2010 should rank highly given the dataset.
    keys = [i["key"] for i in payload["items"]]
    assert "Haiti" in keys


async def test_list_disaster_types_includes_known_categories():
    payload = await _call_tool("list_disaster_types", {})
    types = payload["disaster_types"]
    assert "Earthquake" in types
    assert "Flood" in types

async def test_disaster_stats_invalid_metric_returns_error():
    payload = await _call_tool(
        "disaster_stats",
        {"group_by": "country", "metric": "bogus"},
    )
    assert "error" in payload

async def test_top_disasters_by_impact_returns_events():
    payload = await _call_tool(
        "top_disasters_by_impact",
        {"metric": "total_deaths", "n": 3},
    )
    assert payload["metric"] == "total_deaths"
    assert len(payload["events"]) <= 3
