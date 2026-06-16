"""Turn EM-DAT disaster rows into RAG-friendly narrative Documents.

Each disaster row becomes a single LangChain ``Document`` containing:

* a short narrative paragraph describing what happened, where, and when;
* a markdown table summarising the impact metrics (demonstrating the
  table-in-document multimodal pattern from focus area D);
* metadata pointing back to the EM-DAT row (``id``, ``category``, ``year``, ...)
  so the hybrid retriever and ACL can filter by disaster type / country / year.

The builder selects the most recent ``max_rows`` events by default so the demo
indexes a sane number of vectors (vs the ~14 600 rows in the full dataset). A
top-by-impact selector is also provided.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Iterable, List, Optional

import pandas as pd
from langchain_core.documents import Document

from chatbot.disasters.repository import DisasterRepository
from chatbot.model.schemas import DocumentMetadata

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROWS = 500

def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{int(value):,}"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)

def _fmt_usd(value: Any) -> str:
    """Format a Total Damages ('000 US$) value as USD."""

    if value is None:
        return "n/a"
    try:
        amount = float(value) * 1000.0
        if math.isnan(amount):
            return "n/a"
        return f"${amount:,.0f} USD"
    except (TypeError, ValueError):
        return str(value)

def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    return s

def _format_date(year: Any, month: Any, day: Any) -> str:
    y, m, d = _safe_str(year), _safe_str(month), _safe_str(day)
    if not y:
        return ""
    try:
        y_int = int(float(y))
    except (TypeError, ValueError):
        return y
    parts = [str(y_int)]
    if m:
        try:
            m_int = int(float(m))
            parts.append(f"{m_int:02d}")
            if d:
                try:
                    d_int = int(float(d))
                    parts.append(f"{d_int:02d}")
                except (TypeError, ValueError):
                    pass
        except (TypeError, ValueError):
            pass
    return "-".join(parts)

class DisasterDocumentBuilder:
    """Convert disaster rows into narrative LangChain Documents."""

    def __init__(
        self,
        repository: DisasterRepository,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        self.repository = repository
        self.max_rows = int(max_rows)

    def build_documents(
        self,
        max_rows: Optional[int] = None,
        strategy: str = "recent",
    ) -> List[Document]:
        """Return a list of Documents, one per selected disaster row.

        Args:
            max_rows: cap on the number of rows turned into documents. ``None``
                uses the builder's default (``self.max_rows``).
            strategy: ``"recent"`` (default - latest year first) or
                ``"impact"`` (highest total_deaths first).
        """

        n = int(max_rows) if max_rows is not None else self.max_rows
        if n <= 0:
            return []
        df = self.repository.df
        if df.empty:
            return []
        df = self._select_rows(df, n, strategy)
        documents: List[Document] = []
        for _, row in df.iterrows():
            content = self._render(row)
            metadata = self._metadata(row)
            documents.append(Document(page_content=content, metadata=metadata))
        logger.info(
            "Built %d disaster narrative documents (strategy=%s)",
            len(documents),
            strategy,
        )
        return documents

    @staticmethod
    def _select_rows(df: pd.DataFrame, n: int, strategy: str) -> pd.DataFrame:
        strategy_norm = (strategy or "recent").lower()
        if strategy_norm == "impact":
            if "total_deaths" in df.columns:
                return df.sort_values(
                    by="total_deaths", ascending=False, na_position="last"
                ).head(n)
            return df.head(n)
        # "recent" - by year, then disaster type / country for tie-breaking
        if "year" in df.columns:
            return df.sort_values(by="year", ascending=False, na_position="last").head(n)
        return df.head(n)

    def _render(self, row: pd.Series) -> str:
        dis_no = _safe_str(row.get("dis_no"))
        year = _safe_str(row.get("year"))
        d_type = _safe_str(row.get("disaster_type")) or "Disaster"
        d_subtype = _safe_str(row.get("disaster_subtype"))
        country = _safe_str(row.get("country")) or "Unknown country"
        iso = _safe_str(row.get("iso"))
        region = _safe_str(row.get("region"))
        continent = _safe_str(row.get("continent"))
        location = _safe_str(row.get("location"))
        event_name = _safe_str(row.get("event_name"))
        origin = _safe_str(row.get("origin"))
        mag_value = _safe_str(row.get("dis_mag_value"))
        mag_scale = _safe_str(row.get("dis_mag_scale"))
        start = _format_date(
            row.get("start_year"), row.get("start_month"), row.get("start_day")
        )
        end = _format_date(row.get("end_year"), row.get("end_month"), row.get("end_day"))

        title_parts = [d_type]
        if d_subtype and d_subtype.lower() != d_type.lower():
            title_parts.append(f"({d_subtype})")
        title_parts.append("in")
        title_parts.append(country)
        if year:
            title_parts.append(f"({year})")
        title = " ".join(title_parts)

        narrative_lines: List[str] = []
        narrative_lines.append(f"# EM-DAT Event {dis_no or '(unknown)'}: {title}")
        narrative_lines.append("")
        if event_name:
            narrative_lines.append(f"**Event name:** {event_name}")
        narrative_lines.append(
            f"**Type:** {d_type}"
            + (f" / {d_subtype}" if d_subtype and d_subtype.lower() != d_type.lower() else "")
        )
        narrative_lines.append(
            f"**Where:** {location}, {country}" + (f" ({iso})" if iso else "")
            + (f" - {region}, {continent}" if region or continent else "")
        )
        date_line = "**When:** "
        if start and end and start != end:
            date_line += f"{start} to {end}"
        elif start:
            date_line += start
        else:
            date_line += year or "unknown date"
        narrative_lines.append(date_line)
        if origin:
            narrative_lines.append(f"**Origin / cause:** {origin}")
        if mag_value or mag_scale:
            narrative_lines.append(
                f"**Magnitude:** {mag_value or 'n/a'} {mag_scale}".strip()
            )

        narrative_lines.append("")
        narrative_lines.append("## Impact")
        narrative_lines.append("")
        narrative_lines.append("| Metric | Value |")
        narrative_lines.append("| --- | --- |")
        narrative_lines.append(f"| Total deaths | {_fmt_int(row.get('total_deaths'))} |")
        narrative_lines.append(f"| Injured | {_fmt_int(row.get('no_injured'))} |")
        narrative_lines.append(f"| Affected | {_fmt_int(row.get('no_affected'))} |")
        narrative_lines.append(f"| Homeless | {_fmt_int(row.get('no_homeless'))} |")
        narrative_lines.append(
            f"| Total affected | {_fmt_int(row.get('total_affected'))} |"
        )
        narrative_lines.append(
            f"| Total damages | {_fmt_usd(row.get('total_damages_usd_000'))} |"
        )

        return "\n".join(narrative_lines)

    def _metadata(self, row: pd.Series) -> dict:
        dis_no = _safe_str(row.get("dis_no")) or _safe_str(row.get("Glide")) or "unknown"
        d_type = _safe_str(row.get("disaster_type")) or "Unknown"
        country = _safe_str(row.get("country"))
        year = _safe_str(row.get("year"))
        event_name = _safe_str(row.get("event_name"))
        headline = " - ".join(
            p for p in [d_type, country, year, event_name] if p
        )[:160] or d_type

        meta = DocumentMetadata(
            id=dis_no,
            category=d_type,
            source="emdat",
            headline=headline or None,
            source_type="text",
        )
        extra = {
            "country": country,
            "iso": _safe_str(row.get("iso")),
            "region": _safe_str(row.get("region")),
            "year": year,
        }
        return {**meta.model_dump(exclude_none=True), **{k: v for k, v in extra.items() if v}}

    @staticmethod
    def available_strategies() -> List[str]:
        return ["recent", "impact"]

    @classmethod
    def from_repository(
        cls,
        repository: DisasterRepository,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> "DisasterDocumentBuilder":
        return cls(repository=repository, max_rows=max_rows)
