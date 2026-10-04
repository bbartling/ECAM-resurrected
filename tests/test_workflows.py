import json

import numpy as np
import pandas as pd
import pytest

from ecam_mv import fit_baseline, model_records
from ecam_mv.workflows import load_records, prepare_weather_dataset


class FakeClient:
    def geocode(self, name, *, country, region):
        return {
            "name": "Lake Geneva",
            "admin1": "Wisconsin",
            "latitude": 42.59168,
            "longitude": -88.43343,
            "timezone": "America/Chicago",
        }

    def fetch_hourly(self, latitude, longitude, start, end):
        index = pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "temperature_2m": np.arange(len(index), dtype=float),
                "relative_humidity_2m": [np.nan] * len(index),
            },
            index=index,
        )
        frame.attrs = {
            "weather_units": {"temperature_2m": "C", "relative_humidity_2m": "%"},
            "timezone": "UTC",
            "weather_source": "test fake, not actual weather",
        }
        return frame


def inputs(tmp_path):
    a = pd.DataFrame({"Date": pd.date_range("2024-01-01 00:15", periods=8, freq="15min"), "kW": 4})
    b = pd.DataFrame({"Date": pd.date_range("2024-01-01 03:15", periods=4, freq="15min"), "kW": 6})
    folder = tmp_path / "input"
    folder.mkdir()
    a.to_csv(folder / "a.csv", index=False)
    b.to_csv(folder / "b.csv", index=False)
    return folder


def test_full_workflow_conserves_energy_retains_gaps_and_reference_nulls(tmp_path):
    folder = inputs(tmp_path)
    before = {p.name: p.read_bytes() for p in folder.glob("*.csv")}
    out = tmp_path / "out"
    report = prepare_weather_dataset(
        [folder], out, location="Lake Geneva", region="Wisconsin", client=FakeClient()
    )
    assert {p.name: p.read_bytes() for p in folder.glob("*.csv")} == before
    intervals = pd.read_csv(out / "intervals.csv")
    hourly = load_records(out / "hourly.csv")
    assert len(intervals) == 12 and len(hourly) == 4
    assert intervals.energy.sum() == hourly.energy.sum() == 14
    assert str(hourly.index.tz) == hourly.attrs["timezone"] == "America/Chicago"
    assert hourly.energy.isna().sum() == 1
    assert hourly.duration.tolist() == [1, 1, 0, 1]
    assert hourly.relative_humidity_2m.isna().all()
    with pytest.raises(ValueError, match="unusable"):
        model_records(hourly)
    clean = model_records(hourly, missing="drop")
    assert len(clean) == 3 and clean.energy.sum() == 14
    assert clean.attrs["record_filter"]["excluded_records"] == 1
    # Tiny demonstration fit does not select a larger change-point model.
    fit = fit_baseline(clean, model="1p")
    assert fit.metadata["record_filter"]["excluded_records"] == 1
    assert json.loads((out / "audit.json").read_text()) == report


def test_existing_deliverables_are_never_overwritten(tmp_path):
    folder = inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    target = out / "hourly.csv"
    target.write_text("keep this", encoding="utf-8")
    with pytest.raises(ValueError, match="already exist"):
        prepare_weather_dataset([folder], out, location="Lake Geneva", client=FakeClient())
    assert target.read_text() == "keep this"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"latitude": 42}, "both latitude and longitude"),
        ({"latitude": 42, "longitude": -88, "location": "Lake Geneva"}, "not both"),
        ({}, "Supply a location"),
        ({"latitude": 42, "longitude": -88}, "meter timezone"),
    ],
)
def test_coordinate_and_location_errors(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        prepare_weather_dataset([inputs(tmp_path)], tmp_path / "out", client=FakeClient(), **kwargs)
