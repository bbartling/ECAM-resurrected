"""Tests for bounded meter CSV ingestion and hourly aggregation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ecam_mv.ingestion import MeterBatch, concat_meter_csvs, hourly_meter_records


def write_csv(path, frame):
    frame.to_csv(path, index=False)
    return path


def test_sorted_files_keep_within_file_order_and_audit(tmp_path):
    late = write_csv(
        tmp_path / "b.csv", pd.DataFrame({"Date": ["1/1/2024 00:30"], "kW": [8], "ref": [None]})
    )
    early = write_csv(
        tmp_path / "a file.csv", pd.DataFrame({"Date": ["1/1/2024 00:00"], "kW": [4], "ref": [1]})
    )
    batch = concat_meter_csvs([late, early], timezone="UTC", timestamp_position="start")
    assert batch.records.index.is_monotonic_increasing
    assert batch.records["source_file"].iloc[0] == str(early)
    assert batch.records["source_row"].tolist() == [2, 2]
    assert batch.report["file_count"] == 2
    assert batch.report["reference_missing_counts"] == {"ref": 1}
    json.dumps(batch.report)


def test_timestamp_end_shift_and_power_energy_semantics(tmp_path):
    path = write_csv(
        tmp_path / "meter.csv",
        pd.DataFrame({"Date": ["1/1/2024 00:15", "1/1/2024 00:30"], "kW": [4, 8]}),
    )
    power = concat_meter_csvs([path], timezone="UTC")
    assert power.records.index.tolist() == list(
        pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")
    )
    assert power.records.energy.tolist() == [1, 2]
    direct = concat_meter_csvs([path], timezone="UTC", kind="energy")
    assert direct.records.energy.tolist() == [4, 8]
    custom_unit = concat_meter_csvs([path], timezone="UTC", kind="energy", energy_unit="MWh")
    assert custom_unit.records.attrs["energy_unit"] == "MWh"
    assert hourly_meter_records(custom_unit).records.attrs["energy_unit"] == "MWh"


def test_aware_timestamps_convert_to_requested_timezone(tmp_path):
    path = write_csv(
        tmp_path / "aware.csv", pd.DataFrame({"Date": ["2024-01-01T06:00:00+00:00"], "kW": [1]})
    )
    batch = concat_meter_csvs([path], timezone="America/Chicago", timestamp_position="start")
    assert str(batch.records.index.tz) == "America/Chicago"
    assert batch.records.index[0].tz_convert("UTC") == pd.Timestamp("2024-01-01T06:00Z")


def test_bad_timezone_is_reported_clearly(tmp_path):
    path = write_csv(tmp_path / "meter.csv", pd.DataFrame({"Date": ["1/1/2024"], "kW": [1]}))
    with pytest.raises(ValueError, match="cannot localize timestamps"):
        concat_meter_csvs([path], timezone="Not/A_Timezone", timestamp_position="start")


def test_dst_fall_back_strict_and_infer(tmp_path):
    path = write_csv(
        tmp_path / "fold.csv",
        pd.DataFrame(
            {
                "Date": [
                    "11/3/2024 00:45",
                    "11/3/2024 01:00",
                    "11/3/2024 01:15",
                    "11/3/2024 01:30",
                    "11/3/2024 01:45",
                    "11/3/2024 01:00",
                    "11/3/2024 01:15",
                    "11/3/2024 01:30",
                    "11/3/2024 01:45",
                    "11/3/2024 02:00",
                ],
                "kW": [1] * 10,
            }
        ),
    )
    with pytest.raises(ValueError, match="DST policy='raise'"):
        concat_meter_csvs([path], timezone="America/Chicago", timestamp_position="start")
    batch = concat_meter_csvs(
        [path], timezone="America/Chicago", timestamp_position="start", dst="infer"
    )
    assert batch.records.index.tz_convert("UTC").is_monotonic_increasing
    assert batch.report["inferred_dst_row_count"] == 8


def test_dst_nonexistent_rejected_even_in_infer_mode(tmp_path):
    path = write_csv(
        tmp_path / "spring.csv", pd.DataFrame({"Date": ["3/10/2024 02:15"], "kW": [1]})
    )
    with pytest.raises(ValueError, match="nonexistent"):
        concat_meter_csvs(
            [path], timezone="America/Chicago", timestamp_position="start", dst="infer"
        )


def test_overlap_and_true_utc_duplicate_rejected(tmp_path):
    first = write_csv(tmp_path / "a.csv", pd.DataFrame({"Date": ["1/1/2024 00:00"], "kW": [1]}))
    second = write_csv(tmp_path / "b.csv", pd.DataFrame({"Date": ["1/1/2024 00:00"], "kW": [2]}))
    with pytest.raises(ValueError, match="duplicate/overlapping UTC"):
        concat_meter_csvs([first, second], timezone="UTC", timestamp_position="start")


def test_schema_and_reserved_name_rejected(tmp_path):
    a = write_csv(tmp_path / "a.csv", pd.DataFrame({"Date": ["1/1/2024"], "kW": [1]}))
    b = write_csv(tmp_path / "b.csv", pd.DataFrame({"Date": ["1/1/2024"], "kW": [1], "other": [3]}))
    with pytest.raises(ValueError, match="schema differs"):
        concat_meter_csvs([a, b], timezone="UTC", timestamp_position="start")
    collision = write_csv(
        tmp_path / "collision.csv", pd.DataFrame({"Date": ["1/1/2024"], "kW": [1], "energy": [1]})
    )
    with pytest.raises(ValueError, match="reserved"):
        concat_meter_csvs([collision], timezone="UTC", timestamp_position="start")
    duplicate_header = tmp_path / "duplicate.csv"
    duplicate_header.write_text("Date,kW,kW\n1/1/2024,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate CSV header"):
        concat_meter_csvs([duplicate_header], timezone="UTC", timestamp_position="start")


@pytest.mark.parametrize("value", [None, "bad", "inf", "-inf", "nan"])
def test_invalid_usage_rejected_with_row(value, tmp_path):
    path = write_csv(tmp_path / "bad.csv", pd.DataFrame({"Date": ["1/1/2024"], "kW": [value]}))
    with pytest.raises(ValueError, match=r"CSV rows \[2\]"):
        concat_meter_csvs([path], timezone="UTC", timestamp_position="start")


def test_invalid_timestamp_and_negative_rejected(tmp_path):
    path = write_csv(tmp_path / "bad.csv", pd.DataFrame({"Date": ["not a date"], "kW": [1]}))
    with pytest.raises(ValueError, match="invalid/missing Date"):
        concat_meter_csvs([path], timezone="UTC", timestamp_position="start")
    path = write_csv(tmp_path / "negative.csv", pd.DataFrame({"Date": ["1/1/2024"], "kW": [-1]}))
    with pytest.raises(ValueError, match="negative value"):
        concat_meter_csvs([path], timezone="UTC", timestamp_position="start")
    allowed = concat_meter_csvs(
        [path], timezone="UTC", timestamp_position="start", allow_negative=True
    )
    assert allowed.records.energy.iloc[0] < 0


def test_off_grid_rejected(tmp_path):
    path = write_csv(
        tmp_path / "offgrid.csv", pd.DataFrame({"Date": ["1/1/2024 00:07"], "kW": [1]})
    )
    with pytest.raises(ValueError, match="align"):
        concat_meter_csvs([path], timezone="UTC", timestamp_position="start")


def test_hourly_exact_sums_partial_and_missing_hours(tmp_path):
    path = write_csv(
        tmp_path / "gap.csv",
        pd.DataFrame(
            {
                "Date": ["1/1/2024 00:00", "1/1/2024 00:15", "1/1/2024 02:00", "1/1/2024 02:15"],
                "kW": [4, 8, 12, 16],
                "reference": [1, 3, None, 5],
            }
        ),
    )
    batch = concat_meter_csvs([path], timezone="UTC", timestamp_position="start")
    hourly = hourly_meter_records(batch)
    assert hourly.records.index.tolist() == list(
        pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    )
    assert hourly.records.energy.iloc[0] == 3
    assert pd.isna(hourly.records.energy.iloc[1])
    assert hourly.records.energy.iloc[2] == 7
    assert hourly.records.duration.tolist() == [0.5, 0, 0.5]
    assert hourly.records.coverage.tolist() == [0.5, 0, 0.5]
    assert hourly.records.reference.iloc[0] == 2
    assert hourly.report["missing_hour_count"] == 1
    assert hourly.report["partial_coverage_hour_count"] == 2
    assert hourly.report["hourly_energy_total"] == batch.report["energy_total"]
    assert hourly.records.attrs["frequency"] == "hourly"


def test_nonnumeric_reference_dropped_and_gap_report(tmp_path):
    path = write_csv(
        tmp_path / "gap.csv",
        pd.DataFrame(
            {"Date": ["1/1/2024 00:00", "1/1/2024 00:30"], "kW": [4, 4], "note": ["a", "b"]}
        ),
    )
    batch = concat_meter_csvs([path], timezone="UTC", timestamp_position="start")
    assert batch.report["missing_interval_count"] == 1
    assert batch.report["gap_ranges"][0]["missing_intervals"] == 1
    hourly = hourly_meter_records(batch)
    assert "note" not in hourly.records
    assert hourly.report["dropped_nonnumeric_reference_columns"] == ["note"]


def test_reject_bad_hourly_input():
    with pytest.raises(ValueError, match="timezone-aware"):
        hourly_meter_records(MeterBatch(pd.DataFrame({"energy": [1]}, index=[0]), {}))
