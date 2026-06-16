"""Tests for the Pandas-backed disaster repository."""
from __future__ import annotations

import pandas as pd
import pytest

from chatbot.disasters.repository import DisasterRepository


def test_loads_fixture_csv(small_disasters_repo: DisasterRepository):
    df = small_disasters_repo.df
    # 8 rows in the fixture.
    assert len(df) == 8
    # Column rename applied.
    assert "disaster_type" in df.columns
    assert "total_deaths" in df.columns
    assert "Disaster Type" not in df.columns


def test_query_filters_by_country_and_type(small_disasters_repo: DisasterRepository):
    out = small_disasters_repo.query(disaster_type="Earthquake", country="Japan", limit=5)
    assert out["total_matches"] == 2
    assert all(e["country"] == "Japan" and e["disaster_type"] == "Earthquake" for e in out["events"])


def test_query_filter_by_iso_code(small_disasters_repo: DisasterRepository):
    out = small_disasters_repo.query(country="USA", limit=5)
    assert out["total_matches"] == 2
    isos = {e["iso"] for e in out["events"]}
    assert isos == {"USA"}


def test_query_year_range(small_disasters_repo: DisasterRepository):
    out = small_disasters_repo.query(year_from=2010, year_to=2018, limit=10)
    years = [e["year"] for e in out["events"]]
    assert years and all(2010 <= y <= 2018 for y in years)


def test_query_limit_capped_by_max(small_disasters_repo: DisasterRepository):
    # max_limit=100 in the fixture; even huge `limit` cannot exceed real rows.
    out = small_disasters_repo.query(limit=10_000)
    assert out["returned"] <= 100


def test_query_returns_none_for_missing_values(small_disasters_repo: DisasterRepository):
    out = small_disasters_repo.query(country="Argentina", limit=5)
    assert out["events"][0]["total_deaths"] == 0
    # Reconstruction costs column may be all-NaN -> filtered to None / not in default cols.
    assert "reconstruction_costs_usd_000" not in out["events"][0]


def test_stats_by_country_events(small_disasters_repo: DisasterRepository):
    stats = small_disasters_repo.stats(group_by="country", metric="events", top_n=10)
    items = {item["key"]: item["value"] for item in stats["items"]}
    assert items.get("Japan") == 2
    assert items.get("United States of America (the)") == 2


def test_stats_by_country_total_deaths(small_disasters_repo: DisasterRepository):
    stats = small_disasters_repo.stats(group_by="country", metric="total_deaths", top_n=5)
    # Haiti 2010 earthquake dominates.
    assert stats["items"][0]["key"] == "Haiti"


def test_stats_by_year(small_disasters_repo: DisasterRepository):
    stats = small_disasters_repo.stats(group_by="year", metric="events", top_n=10)
    keys = [item["key"] for item in stats["items"]]
    assert 1990 in keys and 2010 in keys


def test_stats_metric_total_damages_usd_converts_thousands(small_disasters_repo: DisasterRepository):
    stats = small_disasters_repo.stats(group_by="country", metric="total_damages_usd", top_n=2)
    top = stats["items"][0]
    # Top damage row is the Kobe earthquake at 100_000_000 (in thousands of USD) = 1e11 USD.
    assert top["value"] >= 1e11 - 1


def test_stats_invalid_group_by(small_disasters_repo: DisasterRepository):
    with pytest.raises(ValueError):
        small_disasters_repo.stats(group_by="something", metric="events")


def test_stats_invalid_metric(small_disasters_repo: DisasterRepository):
    with pytest.raises(ValueError):
        small_disasters_repo.stats(group_by="country", metric="bogus")


def test_top_by_impact_total_deaths(small_disasters_repo: DisasterRepository):
    out = small_disasters_repo.top_by_impact(metric="total_deaths", n=3)
    countries = [e["country"] for e in out["events"]]
    assert countries[0] == "Haiti"


def test_top_by_impact_rejects_events_metric(small_disasters_repo: DisasterRepository):
    with pytest.raises(ValueError):
        small_disasters_repo.top_by_impact(metric="events", n=5)


def test_list_disaster_types(small_disasters_repo: DisasterRepository):
    types = small_disasters_repo.list_disaster_types()
    assert "Earthquake" in types
    assert "Flood" in types


def test_list_countries(small_disasters_repo: DisasterRepository):
    countries = small_disasters_repo.list_countries()
    assert "Japan" in countries
    assert "Haiti" in countries


def test_from_dataframe_skips_csv_loading():
    df = pd.DataFrame(
        {"dis_no": ["1"], "year": [2000], "disaster_type": ["Flood"], "country": ["X"],
         "total_deaths": [1.0]}
    )
    repo = DisasterRepository.from_dataframe(df)
    assert len(repo.df) == 1
    out = repo.query(disaster_type="Flood", limit=5)
    assert out["total_matches"] == 1


def test_repository_missing_files_raises(tmp_path):
    repo = DisasterRepository(data_dir=tmp_path, csv_files=["nonexistent.csv"])
    with pytest.raises(FileNotFoundError):
        _ = repo.df


def test_available_metric_and_group_helpers():
    assert "events" in DisasterRepository.available_metrics()
    assert "country" in DisasterRepository.available_group_keys()


def test_normalise_column_name_known_and_unknown():
    assert DisasterRepository.normalise_column_name("Total Deaths") == "total_deaths"
    assert DisasterRepository.normalise_column_name("Something New") == "something_new"
