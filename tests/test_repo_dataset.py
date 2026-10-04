"""Repository-only checks; real meter records intentionally do not ship on PyPI."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecam_mv import model_records
from ecam_mv.workflows import load_records

DATA = Path(__file__).resolve().parents[1] / "datasets"
CSV = DATA / "school_hourly_weather.csv"
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="School data is repository-only")


@pytest.fixture(scope="module")
def school():
    return load_records(CSV)


def test_school_processed_energy_rows_and_quality(school):
    assert len(school) == 35029
    assert school.energy.sum() == pytest.approx(2753981.6)
    assert school.energy.isna().sum() == 72
    assert ((school.coverage > 0) & (school.coverage < 1)).sum() == 1
    assert np.isfinite(school.temperature_2m).all()
    assert not school.index.has_duplicates
    assert str(school.index.tz) == school.attrs["timezone"] == "America/Chicago"


def test_school_exclusions_are_explicit_and_conserve_known_energy(school):
    with pytest.raises(ValueError, match="73 unusable"):
        model_records(school)
    clean = model_records(school, missing="drop")
    audit = clean.attrs["record_filter"]
    assert len(clean) == 34956
    assert audit["excluded_records"] == 73
    assert audit["excluded_known_energy"] == pytest.approx(8.7)
    assert clean.energy.sum() == pytest.approx(2753981.6 - 8.7)


def test_school_both_dst_folds_and_spring_gaps_are_preserved(school):
    for date in ("2013-11-03", "2014-11-02", "2015-11-01", "2016-11-06"):
        day = school.loc[date]
        assert len(day) == 25
        assert (day.index.hour == 1).sum() == 2
    for date in ("2014-03-09", "2015-03-08", "2016-03-13", "2017-03-12"):
        day = school.loc[date]
        assert len(day) == 23
        assert not (day.index.hour == 2).any()


def test_published_school_data_is_processed_only_with_portable_provenance():
    assert sorted(p.name for p in DATA.glob("*.csv")) == ["school_hourly_weather.csv"]
    audit = json.loads((DATA / "school_weather_audit.json").read_text(encoding="utf-8"))
    assert audit["meter"]["row_count"] == 139825
    assert audit["meter"]["energy_total"] == pytest.approx(2753981.6)
    assert audit["hourly"]["hourly_row_count"] == 35029
    assert len(audit["meter"]["source_sha256"]) == 4
    assert all(Path(name).name == name for name in audit["meter"]["files"])
    assert all(Path(name).name == name for name in audit["meter"]["source_sha256"])
