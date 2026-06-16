"""Pandas wrapper over EM-DAT natural disaster CSVs.

The CSV columns in the source data use Title Case with spaces and parentheses,
e.g. ``Total Deaths`` and ``Total Damages ('000 US$)``. This module normalises
them to snake_case once at load time, dedupes overlapping years between the two
files on ``Dis No``, and exposes a small filter / aggregate API.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMN_RENAME: Dict[str, str] = {
    "Dis No": "dis_no",
    "Year": "year",
    "Seq": "seq",
    "Glide": "glide",
    "Disaster Group": "disaster_group",
    "Disaster Subgroup": "disaster_subgroup",
    "Disaster Type": "disaster_type",
    "Disaster Subtype": "disaster_subtype",
    "Disaster Subsubtype": "disaster_subsubtype",
    "Event Name": "event_name",
    "Country": "country",
    "ISO": "iso",
    "Region": "region",
    "Continent": "continent",
    "Location": "location",
    "Origin": "origin",
    "Associated Dis": "associated_dis",
    "Associated Dis2": "associated_dis2",
    "OFDA Response": "ofda_response",
    "Appeal": "appeal",
    "Declaration": "declaration",
    "Aid Contribution": "aid_contribution",
    "Dis Mag Value": "dis_mag_value",
    "Dis Mag Scale": "dis_mag_scale",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Local Time": "local_time",
    "River Basin": "river_basin",
    "Start Year": "start_year",
    "Start Month": "start_month",
    "Start Day": "start_day",
    "End Year": "end_year",
    "End Month": "end_month",
    "End Day": "end_day",
    "Total Deaths": "total_deaths",
    "No Injured": "no_injured",
    "No Affected": "no_affected",
    "No Homeless": "no_homeless",
    "Total Affected": "total_affected",
    "Reconstruction Costs ('000 US$)": "reconstruction_costs_usd_000",
    "Insured Damages ('000 US$)": "insured_damages_usd_000",
    "Total Damages ('000 US$)": "total_damages_usd_000",
    "CPI": "cpi",
    "Adm Level": "adm_level",
    "Admin1 Code": "admin1_code",
    "Admin2 Code": "admin2_code",
    "Geo Locations": "geo_locations",
}

_METRIC_COLUMNS: Dict[str, Dict[str, Any]] = {
    "events": {"column": None, "kind": "count"},
    "total_deaths": {"column": "total_deaths", "kind": "sum"},
    "total_affected": {"column": "total_affected", "kind": "sum"},
    "total_damages_usd": {"column": "total_damages_usd_000", "kind": "sum_usd"},
}

_GROUP_COLUMNS = {
    "country": "country",
    "year": "year",
    "region": "region",
    "disaster_type": "disaster_type",
    "continent": "continent",
}

def _clean_value(value: Any) -> Any:
    """Convert NaN / pd.NA to None, leave everything else alone."""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA:
        return None
    return value

def _to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        out.append({k: _clean_value(v) for k, v in record.items()})
    return out

class DisasterRepository:
    """Lightweight Pandas wrapper around EM-DAT CSV exports."""

    DEFAULT_COLUMNS: List[str] = [
        "dis_no",
        "year",
        "disaster_type",
        "country",
        "iso",
        "region",
        "continent",
        "location",
        "total_deaths",
        "total_affected",
        "total_damages_usd_000",
    ]

    def __init__(
        self,
        data_dir: str | Path,
        csv_files: Iterable[str],
        max_limit: int = 200,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.csv_files = list(csv_files)
        self.max_limit = int(max_limit)
        self._df: Optional[pd.DataFrame] = None
        self._lock = threading.Lock()

    def _load(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for filename in self.csv_files:
            path = self.data_dir / filename
            if not path.is_file():
                logger.warning("Disaster CSV not found: %s (skipping)", path)
                continue
            logger.info("Loading EM-DAT CSV: %s", path)
            df = pd.read_csv(path, low_memory=False)
            df = df.rename(columns=_COLUMN_RENAME)
            frames.append(df)
        if not frames:
            raise FileNotFoundError(
                f"No disaster CSV files loaded from {self.data_dir}; "
                f"expected one of {self.csv_files}"
            )
        combined = pd.concat(frames, ignore_index=True, sort=False)
        # Drop duplicate events (the two source files overlap on 1970-2021).
        if "dis_no" in combined.columns:
            before = len(combined)
            combined = combined.drop_duplicates(subset=["dis_no"], keep="first")
            logger.info(
                "Deduplicated disasters by dis_no: %d -> %d rows",
                before,
                len(combined),
            )
        for col in (
            "year",
            "start_year",
            "end_year",
            "total_deaths",
            "no_injured",
            "no_affected",
            "no_homeless",
            "total_affected",
            "reconstruction_costs_usd_000",
            "insured_damages_usd_000",
            "total_damages_usd_000",
        ):
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")
        return combined

    @property
    def df(self) -> pd.DataFrame:
        """Lazy, thread-safe DataFrame accessor."""

        if self._df is None:
            with self._lock:
                if self._df is None:
                    self._df = self._load()
        return self._df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, max_limit: int = 200) -> "DisasterRepository":
        """Build a repository directly from an in-memory DataFrame (for tests)."""

        repo = cls(data_dir=".", csv_files=[], max_limit=max_limit)
        repo._df = df.copy()
        return repo

    def _normalise_str(self, value: Optional[str]) -> Optional[str]:
        return value.strip().lower() if isinstance(value, str) and value.strip() else None

    def _filter(
        self,
        disaster_type: Optional[str] = None,
        country: Optional[str] = None,
        region: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self.df
        if disaster_type:
            target = self._normalise_str(disaster_type)
            if target:
                df = df[df["disaster_type"].astype(str).str.lower() == target]
        if country:
            target = self._normalise_str(country)
            if target:
                country_col = df["country"].astype(str).str.lower()
                iso_col = df["iso"].astype(str).str.lower()
                df = df[(country_col == target) | (iso_col == target)]
        if region:
            target = self._normalise_str(region)
            if target:
                df = df[df["region"].astype(str).str.lower() == target]
        if year_from is not None:
            df = df[df["year"] >= int(year_from)]
        if year_to is not None:
            df = df[df["year"] <= int(year_to)]
        return df

    def query(
        self,
        disaster_type: Optional[str] = None,
        country: Optional[str] = None,
        region: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        limit: int = 20,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return up to ``limit`` matching rows plus the unfiltered count."""

        filtered = self._filter(
            disaster_type=disaster_type,
            country=country,
            region=region,
            year_from=year_from,
            year_to=year_to,
        )
        total = int(len(filtered))
        capped_limit = max(1, min(int(limit), self.max_limit))
        cols = columns or self.DEFAULT_COLUMNS
        cols = [c for c in cols if c in filtered.columns]
        sorted_df = (
            filtered.sort_values(by="year", ascending=False, na_position="last")
            if "year" in filtered.columns
            else filtered
        )
        sliced = sorted_df.head(capped_limit)[cols] if cols else sorted_df.head(capped_limit)
        return {
            "total_matches": total,
            "returned": int(len(sliced)),
            "filters": {
                "disaster_type": disaster_type,
                "country": country,
                "region": region,
                "year_from": year_from,
                "year_to": year_to,
            },
            "events": _to_records(sliced),
        }

    def stats(
        self,
        group_by: str,
        metric: str,
        disaster_type: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Aggregate events by ``group_by`` according to ``metric``."""

        group_col = _GROUP_COLUMNS.get(group_by.lower())
        if not group_col:
            raise ValueError(
                f"Unsupported group_by={group_by!r}; pick one of {sorted(_GROUP_COLUMNS)}"
            )
        metric_spec = _METRIC_COLUMNS.get(metric.lower())
        if not metric_spec:
            raise ValueError(
                f"Unsupported metric={metric!r}; pick one of {sorted(_METRIC_COLUMNS)}"
            )

        filtered = self._filter(
            disaster_type=disaster_type,
            year_from=year_from,
            year_to=year_to,
        )
        if filtered.empty or group_col not in filtered.columns:
            return {
                "group_by": group_by,
                "metric": metric,
                "items": [],
                "filters": {
                    "disaster_type": disaster_type,
                    "year_from": year_from,
                    "year_to": year_to,
                },
            }

        if metric_spec["kind"] == "count":
            counts = filtered.groupby(group_col).size().reset_index(name="value")
        else:
            col = metric_spec["column"]
            if col not in filtered.columns:
                return {
                    "group_by": group_by,
                    "metric": metric,
                    "items": [],
                    "filters": {
                        "disaster_type": disaster_type,
                        "year_from": year_from,
                        "year_to": year_to,
                    },
                }
            counts = (
                filtered.groupby(group_col)[col].sum(min_count=1).reset_index(name="value")
            )
            counts["value"] = counts["value"].fillna(0)
            if metric_spec["kind"] == "sum_usd":
                counts["value"] = counts["value"].astype(float) * 1000.0

        counts = counts.sort_values(by="value", ascending=False).head(max(1, int(top_n)))
        return {
            "group_by": group_by,
            "metric": metric,
            "filters": {
                "disaster_type": disaster_type,
                "year_from": year_from,
                "year_to": year_to,
            },
            "items": [
                {"key": _clean_value(row[group_col]), "value": _clean_value(row["value"])}
                for _, row in counts.iterrows()
            ],
        }

    def top_by_impact(
        self,
        metric: str,
        n: int = 10,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Top ``n`` single events ranked by ``metric``."""

        metric_spec = _METRIC_COLUMNS.get(metric.lower())
        if not metric_spec or metric_spec["kind"] == "count":
            raise ValueError(
                "metric must be one of total_deaths, total_affected, total_damages_usd"
            )
        col = metric_spec["column"]
        if col is None:
            raise ValueError(
                "metric must be one of total_deaths, total_affected, total_damages_usd"
            )

        df = self._filter(year_from=year_from, year_to=year_to)
        if df.empty or col not in df.columns:
            return {"metric": metric, "events": []}

        df = df.sort_values(by=col, ascending=False, na_position="last").head(max(1, int(n)))
        cols = [c for c in self.DEFAULT_COLUMNS if c in df.columns]
        return {"metric": metric, "events": _to_records(df[cols])}

    def list_disaster_types(self) -> List[str]:
        if "disaster_type" not in self.df.columns:
            return []
        return sorted(
            {
                str(x)
                for x in self.df["disaster_type"].dropna().unique()
                if str(x).strip()
            }
        )

    def list_countries(self) -> List[str]:
        if "country" not in self.df.columns:
            return []
        return sorted(
            {
                str(x)
                for x in self.df["country"].dropna().unique()
                if str(x).strip()
            }
        )

    @staticmethod
    def available_metrics() -> List[str]:
        return list(_METRIC_COLUMNS.keys())

    @staticmethod
    def available_group_keys() -> List[str]:
        return list(_GROUP_COLUMNS.keys())

    @staticmethod
    def normalise_column_name(raw: str) -> str:
        """Public helper used by tests / debugging."""

        if raw in _COLUMN_RENAME:
            return _COLUMN_RENAME[raw]
        return re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower())
