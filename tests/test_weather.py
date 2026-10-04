"""Offline tests for Open-Meteo handling and UTC-safe weather joins."""

from __future__ import annotations

import json
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

import ecam_mv.weather as weather_module
from ecam_mv.weather import OpenMeteoClient, join_weather


def hourly_payload(start, end, **values):
    requested = pd.date_range(start, end, freq="h", tz="UTC")
    index = pd.date_range(
        requested[0].normalize(),
        requested[-1].normalize() + pd.Timedelta(hours=23),
        freq="h",
        tz="UTC",
    )
    offset = int((requested[0] - index[0]) / pd.Timedelta(hours=1))
    padded = {}
    for key, data in values.items():
        padded[key] = [None] * offset + list(data) + [None] * (len(index) - offset - len(data))
    return {
        "latitude": 42.6,
        "longitude": -88.4,
        "hourly_units": {key: "°C" for key in padded},
        "hourly": {"time": (index.as_unit("ns").asi8 // 1_000_000_000).tolist(), **padded},
    }


def test_fetch_validates_epoch_grid_units_and_cache(tmp_path, monkeypatch):
    calls = []

    def fake_request(self, url):
        calls.append(url)
        return hourly_payload("2024-01-01", "2024-01-01 02:00", temperature_2m=[-4, None, 2])

    monkeypatch.setattr(OpenMeteoClient, "_request_json", fake_request)
    client = OpenMeteoClient(tmp_path)
    start, end = (
        pd.Timestamp("2024-01-01 00:15", tz="UTC"),
        pd.Timestamp("2024-01-01 02:59", tz="UTC"),
    )
    data = client.fetch_hourly(42.59168, -88.43343, start, end, variables=["temperature_2m"])
    assert data.index.equals(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    assert data.temperature_2m.tolist()[0] == -4
    assert np.isnan(data.temperature_2m.iloc[1])
    assert data.attrs["weather_units"]["temperature_2m"] == "°C"
    assert data.attrs["attribution"] == "Open-Meteo; Copernicus ERA5"
    assert data.attrs["license"] == "CC BY 4.0"
    assert len(data) == 3  # Request bounds are partial days; response covers whole UTC dates.
    assert len(calls) == 1
    again = client.fetch_hourly(42.59168, -88.43343, start, end, variables=["temperature_2m"])
    assert again.equals(data)
    assert len(calls) == 1


def test_corrupt_cache_refetches_and_request_parameters_are_keyed(tmp_path, monkeypatch):
    calls = []

    def fake_request(self, url):
        calls.append(url)
        return hourly_payload("2024-01-01", "2024-01-01", temperature_2m=[1])

    monkeypatch.setattr(OpenMeteoClient, "_request_json", fake_request)
    client = OpenMeteoClient(tmp_path)
    args = (1, 2, pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
    client.fetch_hourly(*args, variables=["temperature_2m"])
    cache_file = next(tmp_path.glob("*.json"))
    cache_file.write_text("{broken", encoding="utf-8")
    client.fetch_hourly(*args, variables=["temperature_2m"])
    client.fetch_hourly(*args, variables=["temperature_2m"], model="era5_land")
    assert len(calls) == 3
    assert json.loads(cache_file.read_text(encoding="utf-8"))["hourly"]["temperature_2m"][0] == 1


@pytest.mark.parametrize(
    "start,end",
    [
        ("2024-01-01", "2024-01-02T00:00Z"),
        ("2024-01-02T00:00Z", "2024-01-01T00:00Z"),
    ],
)
def test_fetch_requires_aware_ordered_bounds(start, end):
    with pytest.raises(ValueError):
        OpenMeteoClient(retries=0).fetch_hourly(0, 0, start, end)


@pytest.mark.parametrize(
    "payload",
    [
        {"hourly_units": {"temperature_2m": "°C"}, "hourly": {"temperature_2m": [1]}},
        {"hourly_units": {"temperature_2m": "°C"}, "hourly": {"time": [0], "temperature_2m": [1]}},
    ],
)
def test_fetch_rejects_missing_or_wrong_timestamps(payload, monkeypatch):
    monkeypatch.setattr(OpenMeteoClient, "_request_json", lambda self, url: payload)
    client = OpenMeteoClient(retries=0)
    with pytest.raises(ValueError, match="timestamps"):
        client.fetch_hourly(
            0,
            0,
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
            variables=["temperature_2m"],
        )


def test_fetch_rejects_reference_length_mismatch(monkeypatch):
    index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    payload = {
        "hourly_units": {"temperature_2m": "°C"},
        "hourly": {
            "time": (index.as_unit("ns").asi8 // 1_000_000_000).tolist(),
            "temperature_2m": [1.0],
        },
    }
    monkeypatch.setattr(OpenMeteoClient, "_request_json", lambda self, url: payload)
    with pytest.raises(ValueError, match="invalid length"):
        OpenMeteoClient(retries=0).fetch_hourly(
            0,
            0,
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
            variables=["temperature_2m"],
        )


@pytest.mark.parametrize("bad", ["bad", float("inf")])
def test_fetch_rejects_bad_reference_values(bad, monkeypatch):
    monkeypatch.setattr(
        OpenMeteoClient,
        "_request_json",
        lambda self, url: hourly_payload("2024-01-01", "2024-01-01", temperature_2m=[bad]),
    )
    with pytest.raises(ValueError, match="temperature_2m"):
        OpenMeteoClient(retries=0).fetch_hourly(
            0,
            0,
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
            variables=["temperature_2m"],
        )


def test_year_chunking(monkeypatch):
    requests = []

    def fake(self, params):
        requests.append(params)
        lo, hi = params["start_date"], params["end_date"]
        index = pd.date_range(lo, hi + " 23:00", freq="h", tz="UTC")
        epoch_seconds = index.as_unit("ns").asi8 // 1_000_000_000
        return {
            "hourly_units": {"temperature_2m": "°C"},
            "hourly": {"time": epoch_seconds.tolist(), "temperature_2m": [1] * len(index)},
        }

    monkeypatch.setattr(OpenMeteoClient, "_cached_request", fake)
    result = OpenMeteoClient(retries=0).fetch_hourly(
        0,
        0,
        pd.Timestamp("2023-12-31 23:00", tz="UTC"),
        pd.Timestamp("2024-01-01 01:00", tz="UTC"),
        variables=["temperature_2m"],
    )
    assert len(requests) == 2
    assert [r["start_date"] for r in requests] == ["2023-12-31", "2024-01-01"]
    assert len(result) == 3


def test_retry_transient_and_no_retry_bad_request(monkeypatch):
    calls = []

    def transient(url, *, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise HTTPError(url, 503, "busy", {}, None)
        return {"ok": True}

    monkeypatch.setattr(weather_module, "_json_response", transient)
    monkeypatch.setattr(weather_module.time, "sleep", lambda delay: None)
    assert OpenMeteoClient(retries=1)._request_json("url") == {"ok": True}
    assert len(calls) == 2

    calls.clear()

    def bad_request(url, *, timeout):
        calls.append(url)
        raise HTTPError(url, 400, "bad", {}, None)

    monkeypatch.setattr(weather_module, "_json_response", bad_request)
    with pytest.raises(OSError, match="HTTP 400"):
        OpenMeteoClient(retries=3)._request_json("url")
    assert len(calls) == 1


def test_retry_timeout_and_reject_malformed_json(monkeypatch):
    calls = []
    json_response = weather_module._json_response

    def timeout_once(url, *, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise TimeoutError("timed out")
        return {"ok": True}

    monkeypatch.setattr(weather_module, "_json_response", timeout_once)
    monkeypatch.setattr(weather_module.time, "sleep", lambda delay: None)
    assert OpenMeteoClient(retries=1)._request_json("url") == {"ok": True}
    assert len(calls) == 2

    class BadResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{malformed"

    monkeypatch.setattr(weather_module, "urlopen", lambda request, timeout: BadResponse())
    with pytest.raises(ValueError, match="malformed JSON"):
        json_response("https://example.invalid", timeout=1)


def test_geocode_exact_and_ambiguous_region(monkeypatch):
    places = [
        {
            "name": "Lake Geneva",
            "country_code": "US",
            "admin1": "Wisconsin",
            "latitude": 42.59168,
            "longitude": -88.43343,
        },
        {
            "name": "Lake Geneva",
            "country_code": "US",
            "admin1": "Florida",
            "latitude": 28,
            "longitude": -81,
        },
        {
            "name": "Lake Geneva Dam",
            "country_code": "US",
            "admin1": "Wisconsin",
            "latitude": 42,
            "longitude": -88,
        },
    ]
    monkeypatch.setattr(OpenMeteoClient, "_request_json", lambda self, url: {"results": places})
    client = OpenMeteoClient(retries=0)
    with pytest.raises(ValueError, match="Ambiguous"):
        client.geocode("Lake Geneva")
    match = client.geocode("lake geneva", region="Wisconsin")
    assert match["latitude"] == pytest.approx(42.59168)
    assert client.geocode("Lake Geneva Dam", region="Wisconsin")["name"] == "Lake Geneva Dam"
    with pytest.raises(ValueError, match="No exact"):
        client.geocode("Lake Geneva", country="CA")


def test_geocode_bad_coordinates(monkeypatch):
    monkeypatch.setattr(
        OpenMeteoClient,
        "_request_json",
        lambda self, url: {
            "results": [{"name": "X", "country_code": "US", "latitude": 91, "longitude": 0}]
        },
    )
    with pytest.raises(ValueError, match="coordinates"):
        OpenMeteoClient(retries=0).geocode("X")


def test_join_spring_and_fall_dst_folds_and_interval_hour_boundaries():
    meter_index = pd.DatetimeIndex(
        [
            "2024-11-03 06:45:00Z",
            "2024-11-03 07:00:00Z",
            "2024-11-03 07:15:00Z",
        ]
    ).tz_convert("America/Chicago")
    records = pd.DataFrame({"energy": [1.0, 2.0, 3.0]}, index=meter_index)
    records.attrs["timezone"] = "America/Chicago"
    wx_index = pd.date_range("2024-11-03 06:00", "2024-11-03 08:00", freq="h", tz="UTC")
    wx = pd.DataFrame(
        {
            "temperature_2m": [10.0, 11.0, 12.0],
            "precipitation": [0, 1, 2],
            "cloud_cover": ["20", "30", "40"],
        },
        index=wx_index,
    )
    batch = join_weather(records, wx)
    assert batch.records.temperature_2m.tolist() == [10, 11, 11]
    assert batch.records.energy.sum() == records.energy.sum()
    assert batch.records.index.equals(records.index)
    assert batch.report["matched_rows"] == 3
    assert batch.records.attrs["predictor"] == "temperature"
    assert batch.records.attrs["timezone"] == "America/Chicago"
    assert batch.records.temperature.tolist() == [10, 11, 11]
    assert batch.records.cloud_cover.tolist() == [20.0, 30.0, 30.0]
    assert wx.cloud_cover.tolist() == ["20", "30", "40"]
    assert "not allocated" in batch.report["precipitation_semantics"]


def test_join_hour_boundary_and_flag_missing_drybulb_and_reference_nan():
    idx = pd.date_range("2024-03-10 01:45", periods=2, freq="15min", tz="America/Chicago")
    records = pd.DataFrame({"energy": [1, 1]}, index=idx)
    wxidx = pd.DatetimeIndex(["2024-03-10 07:00Z", "2024-03-10 08:00Z"])
    wx = pd.DataFrame({"temperature_2m": [5.0, np.nan], "cloud_cover": [np.nan, 80]}, index=wxidx)
    batch = join_weather(records, wx, missing="flag")
    assert batch.records.weather_missing.tolist() == [False, True]
    assert batch.records.index.equals(records.index)
    assert batch.report["reference_missing_counts"] == {"temperature_2m": 1, "cloud_cover": 1}
    assert batch.report["unmatched_rows"] == 0
    assert batch.report["bad_drybulb_rows"] == 1
    with pytest.raises(ValueError, match="Missing drybulb"):
        join_weather(records, wx)


@pytest.mark.parametrize("problem", ["naive", "duplicate", "offhour", "collision", "nonnumeric"])
def test_join_rejects_invalid_weather(problem):
    idx = pd.DatetimeIndex(["2024-01-01T00:00Z"])
    records = pd.DataFrame({"energy": [1.0]}, index=idx)
    wxidx = idx
    cols = {"temperature_2m": [2.0]}
    if problem == "naive":
        wxidx = pd.DatetimeIndex(["2024-01-01"])
    elif problem == "duplicate":
        wxidx = pd.DatetimeIndex(["2024-01-01T00:00Z", "2024-01-01T00:00Z"])
        cols = {"temperature_2m": [2.0, 3.0]}
    elif problem == "offhour":
        wxidx = pd.DatetimeIndex(["2024-01-01T00:30Z"])
    elif problem == "collision":
        records["temperature_2m"] = [99]
    elif problem == "nonnumeric":
        cols = {"temperature_2m": ["hot"]}
    wx = pd.DataFrame(cols, index=wxidx)
    with pytest.raises(ValueError):
        join_weather(records, wx, missing="flag")


def test_join_reports_missing_weather_hours_and_bad_input_meter_order():
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    records = pd.DataFrame({"energy": [1, 2]}, index=idx)
    wx = pd.DataFrame({"temperature_2m": [3]}, index=idx[:1])
    joined = join_weather(records, wx, missing="flag")
    assert joined.report["unmatched_rows"] == 1
    unordered = records.iloc[::-1]
    with pytest.raises(ValueError, match="chronological"):
        join_weather(unordered, wx, missing="flag")
