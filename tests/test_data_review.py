import numpy as np
import pandas as pd
import pytest

from ecam_mv import (
    MeterBatch,
    OpenMeteoClient,
    concat_meter_csvs,
    hourly_meter_records,
    join_weather,
)


def test_reserved_quality_columns_cannot_overwrite_actual_coverage(tmp_path):
    path = tmp_path / "meter.csv"
    pd.DataFrame({"Date": ["2024-01-01 00:15"], "kW": [4], "coverage": [1]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="reserved"):
        concat_meter_csvs([path], timezone="UTC")


@pytest.mark.parametrize("value", [np.nan, np.inf, "bad"])
def test_hourly_public_api_rejects_invalid_interval_energy(value):
    frame = pd.DataFrame(
        {"energy": [value]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC", freq="15min")
    )
    with pytest.raises(ValueError, match="energy"):
        hourly_meter_records(MeterBatch(frame, {"interval_minutes": 15}))


@pytest.mark.parametrize("interval", [True, 15.5, 17])
def test_hourly_public_api_does_not_truncate_bad_interval_metadata(interval):
    frame = pd.DataFrame(
        {"energy": [1]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC", freq="15min")
    )
    with pytest.raises(ValueError, match="interval_minutes"):
        hourly_meter_records(MeterBatch(frame, {"interval_minutes": interval}))


def test_join_preserves_declared_drybulb_units_and_rejects_bad_units():
    index = pd.date_range("2024-01-01", periods=1, tz="UTC", freq="h")
    meter, weather = (
        pd.DataFrame({"energy": [2]}, index=index),
        pd.DataFrame({"temperature_2m": [32]}, index=index),
    )
    weather.attrs["weather_units"] = {"temperature_2m": "°F"}
    joined = join_weather(meter, weather)
    assert joined.records.temperature.iloc[0] == 32
    assert joined.records.attrs["temperature_unit"] == "F"
    weather.attrs["weather_units"]["temperature_2m"] = "K"
    with pytest.raises(ValueError, match="unit"):
        join_weather(meter, weather)


def test_reversed_weather_bounds_in_same_hour_are_rejected():
    with pytest.raises(ValueError, match="start must"):
        OpenMeteoClient().fetch_hourly(42, -88, "2024-01-01T00:45Z", "2024-01-01T00:15Z")
