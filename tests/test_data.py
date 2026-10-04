import numpy as np
import pandas as pd
import pytest

from ecam_mv.data import calendar_features, prepare_billing, prepare_hourly


def quarter_hour_frame():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 00:15", periods=8, freq="15min"),
            "energy": [4.0] * 8,
            "temperature": np.arange(8.0),
        }
    )


def test_average_power_integration_and_interval_end_semantics():
    hourly = prepare_hourly(
        quarter_hour_frame(), kind="power", interval_minutes=15, timestamp_position="end"
    )
    assert hourly.index.tolist() == list(pd.date_range("2024-01-01", periods=2, freq="h"))
    assert hourly.energy.tolist() == [4.0, 4.0]
    assert hourly.temperature.tolist() == [1.5, 5.5]
    assert hourly.duration.tolist() == [1.0, 1.0]


def test_interval_energy_is_summed_not_integrated_again():
    hourly = prepare_hourly(quarter_hour_frame(), interval_minutes=15, timestamp_position="end")
    assert hourly.energy.tolist() == [16.0, 16.0]


@pytest.mark.parametrize("bad", ["duplicate", "missing", "nan", "negative", "off_grid"])
def test_meter_quality_errors_are_explicit(bad):
    frame = quarter_hour_frame()
    if bad == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif bad == "missing":
        frame = frame.drop(index=2)
    elif bad == "nan":
        frame.loc[1, "temperature"] = np.nan
    elif bad == "negative":
        frame.loc[1, "energy"] = -2
    else:
        frame.loc[1, "timestamp"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        prepare_hourly(frame, interval_minutes=15, timestamp_position="end")


def test_explicit_partial_hour_exposure():
    frame = quarter_hour_frame().drop(index=2)
    hourly = prepare_hourly(
        frame, kind="power", interval_minutes=15, timestamp_position="end", min_coverage=0.75
    )
    assert hourly.energy.tolist() == [3.0, 4.0]
    assert hourly.duration.tolist() == [0.75, 1.0]


def test_dst_fall_back_offsets_preserve_both_hours():
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-11-03T01:00:00-05:00", "2024-11-03T01:00:00-06:00"],
            "energy": [4, 5],
            "temperature": [10, 11],
        }
    )
    hourly = prepare_hourly(frame, timezone="America/Chicago")
    assert len(hourly) == 2
    assert hourly.index[0].hour == hourly.index[1].hour == 1
    assert hourly.energy.sum() == 9


def test_naive_dst_ambiguity_rejected():
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-11-03 01:00"],
            "energy": [5],
            "temperature": [10],
        }
    )
    with pytest.raises(ValueError, match="DST"):
        prepare_hourly(frame, timezone="America/Chicago")


def test_billing_calendar_days_and_overlap():
    frame = pd.DataFrame(
        {
            "start": ["2024-01-01", "2024-02-01"],
            "end": ["2024-02-01", "2024-03-01"],
            "energy": [310, 290],
            "temperature": [5, 6],
        }
    )
    bills = prepare_billing(frame)
    assert bills.duration.tolist() == [31, 29]
    assert (bills.energy / bills.duration).tolist() == [10, 10]
    frame.loc[1, "start"] = "2024-01-31"
    with pytest.raises(ValueError, match="overlap"):
        prepare_billing(frame)


def test_calendar_holidays_and_overnight_schedule():
    frame = prepare_hourly(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01 22:00", periods=10, freq="h"),
                "energy": 4,
                "temperature": 5,
            }
        )
    )
    calendar = calendar_features(frame, occupancy={0: (22, 6)})
    assert calendar.occupied.tolist() == [True] * 8 + [False] * 2
    holiday = calendar_features(frame, holidays=["2024-01-02"], occupancy={0: (22, 6)})
    assert holiday.occupied.tolist() == [True, True] + [False] * 8
    assert holiday.daytype.iloc[2] == "holiday"
    assert holiday.attrs == frame.attrs
