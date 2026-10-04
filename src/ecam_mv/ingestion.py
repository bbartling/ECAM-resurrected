"""Bounded, audited ingestion and hourly aggregation for meter CSV files."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MeterBatch:
    """Meter records together with a JSON-serializable ingestion/QA report."""

    records: pd.DataFrame
    report: dict[str, Any]


def _validated_interval(interval_minutes: int) -> None:
    if (
        isinstance(interval_minutes, bool)
        or not isinstance(interval_minutes, int)
        or interval_minutes <= 0
        or 60 % interval_minutes
    ):
        raise ValueError("interval_minutes must be a positive integer divisor of 60")


def _parse_file_timestamps(
    values: pd.Series, path: Path, timezone: str, dst: str
) -> pd.DatetimeIndex:
    parsed: list[pd.Timestamp | pd.NaT] = []
    bad: list[int] = []
    for row, value in enumerate(values, start=2):
        try:
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                raise ValueError("missing")
            parsed.append(stamp)
        except (TypeError, ValueError, OverflowError):
            parsed.append(pd.NaT)
            bad.append(row)
    if bad:
        raise ValueError(
            f"{path}: invalid/missing {values.name} at {len(bad)} row(s), CSV rows {bad[:8]}"
        )
    aware = [stamp.tzinfo is not None for stamp in parsed]
    if any(aware) and not all(aware):
        raise ValueError(f"{path}: timestamp column mixes timezone-aware and naive values")
    try:
        if all(aware):
            return pd.DatetimeIndex(pd.to_datetime(parsed, utc=True)).tz_convert(timezone)
        naive = pd.DatetimeIndex(parsed)
        return naive.tz_localize(
            timezone, ambiguous="infer" if dst == "infer" else "raise", nonexistent="raise"
        )
    except Exception as exc:
        raise ValueError(
            f"{path}: cannot localize timestamps in {timezone} (DST policy={dst!r}): {exc}"
        ) from exc


def _json_timestamp(value: pd.Timestamp) -> str:
    return value.isoformat()


def _gap_ranges(index: pd.DatetimeIndex, step: pd.Timedelta) -> list[dict[str, Any]]:
    utc = index.tz_convert("UTC")
    gaps: list[dict[str, Any]] = []
    for i in range(1, len(utc)):
        difference = utc[i] - utc[i - 1]
        if difference > step:
            missing_count = int(difference // step) - 1
            gaps.append(
                {
                    "after": _json_timestamp(utc[i - 1]),
                    "before": _json_timestamp(utc[i]),
                    "missing_intervals": missing_count,
                }
            )
    return gaps


def concat_meter_csvs(
    paths: Iterable[str | Path],
    *,
    timestamp: str = "Date",
    usage: str = "kW",
    timezone: str,
    interval_minutes: int = 15,
    timestamp_position: str = "end",
    kind: str = "power",
    energy_unit: str = "kWh",
    dst: str = "raise",
    allow_negative: bool = False,
) -> MeterBatch:
    """Read CSVs deterministically, validate them, and retain all interval rows."""
    _validated_interval(interval_minutes)
    if not isinstance(timezone, str) or not timezone.strip():
        raise ValueError("timezone must be an explicit nonempty timezone name")
    if timestamp_position not in {"start", "end"}:
        raise ValueError("timestamp_position must be 'start' or 'end'")
    if kind not in {"power", "energy"}:
        raise ValueError("kind must be 'power' or 'energy'")
    if not isinstance(energy_unit, str) or not energy_unit.strip():
        raise ValueError("energy_unit must be a nonempty string")
    if dst not in {"raise", "infer"}:
        raise ValueError("dst must be 'raise' or 'infer'")
    ordered_paths = sorted((Path(path) for path in paths), key=lambda path: str(path))
    if not ordered_paths:
        raise ValueError("At least one CSV path is required")
    frames: list[pd.DataFrame] = []
    schema: list[str] | None = None
    null_counts: dict[str, int] = {}
    inferred_dst_rows = 0
    file_checksums = {}
    for path in ordered_paths:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                header = next(csv.reader(csv_file), None)
            if header is None:
                raise ValueError(f"{path}: CSV is empty")
            if len(header) != len(set(header)):
                raise ValueError(f"{path}: duplicate CSV header names are not allowed")
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
            raise ValueError(f"Could not read CSV {path}: {exc}") from exc
        if frame.columns.duplicated().any():
            raise ValueError(f"{path}: duplicate column names are not allowed")
        columns = list(frame.columns)
        if schema is None:
            schema = columns
        elif columns != schema:
            raise ValueError(f"{path}: CSV schema differs; expected {schema}, found {columns}")
        missing = {timestamp, usage} - set(columns)
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        reserved = {"energy", "duration", "coverage", "source_file", "source_row", "timestamp_end"}
        collision = reserved.intersection(columns) - ({usage} if usage == "energy" else set())
        if collision:
            raise ValueError(f"{path}: reserved generated column name(s): {sorted(collision)}")
        if frame.empty:
            raise ValueError(f"{path}: contains no data rows")
        index = _parse_file_timestamps(frame[timestamp], path, timezone, dst)
        file_checksums[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Count repeated local wall-clock labels resolved by infer, including each fold's rows.
        if dst == "infer" and index.tz is not None:
            local_naive = index.tz_localize(None)
            inferred_dst_rows += int(local_naive.duplicated(keep=False).sum())
        raw_usage = frame[usage]
        numeric = pd.to_numeric(raw_usage, errors="coerce")
        invalid = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        if invalid.any():
            bad_rows = (np.flatnonzero(invalid.to_numpy()) + 2).tolist()
            raise ValueError(
                f"{path}: {usage} has {len(bad_rows)} missing/nonnumeric/nonfinite value(s), "
                f"CSV rows {bad_rows[:8]}"
            )
        if not allow_negative and (numeric < 0).any():
            rows = (np.flatnonzero((numeric < 0).to_numpy()) + 2).tolist()
            raise ValueError(
                f"{path}: {usage} has {len(rows)} negative value(s), CSV rows {rows[:8]}"
            )
        data = frame.copy()
        data[usage] = numeric.to_numpy(dtype=float)
        data["source_file"] = str(path)
        data["source_row"] = np.arange(2, len(data) + 2, dtype=int)
        if timestamp_position == "end":
            data["timestamp_end"] = index
            index = index - pd.Timedelta(minutes=interval_minutes)
        data.index = index
        frames.append(data)
        for col, count in frame.isna().sum().items():
            if col not in {timestamp, usage}:
                null_counts[col] = null_counts.get(col, 0) + int(count)
    combined = pd.concat(frames, axis=0)
    utc_index = combined.index.tz_convert("UTC")
    if utc_index.has_duplicates:
        duplicates = utc_index[utc_index.duplicated(keep=False)]
        raise ValueError(
            f"Meter files contain {len(duplicates)} duplicate/overlapping UTC timestamps, "
            f"e.g. {duplicates[:5].tolist()}"
        )
    step = pd.Timedelta(minutes=interval_minutes)
    misaligned = (
        (utc_index.minute % interval_minutes != 0)
        | (utc_index.second != 0)
        | (utc_index.microsecond != 0)
        | (utc_index.nanosecond != 0)
    )
    if misaligned.any():
        rows = combined.loc[misaligned, ["source_file", "source_row"]].head(8)
        raise ValueError(
            f"Timestamps must align to {interval_minutes}-minute grid; "
            f"examples: {rows.to_dict('records')}"
        )
    combined = combined.sort_index(kind="stable")
    raw = combined[usage].to_numpy(dtype=float)
    combined["energy"] = raw * (interval_minutes / 60) if kind == "power" else raw
    combined.index.name = "timestamp"
    gaps = _gap_ranges(combined.index, step)
    required_missing = {timestamp: 0, usage: 0}
    report: dict[str, Any] = {
        "file_count": len(ordered_paths),
        "row_count": len(combined),
        "files": [str(path) for path in ordered_paths],
        "source_sha256": file_checksums,
        "start": _json_timestamp(combined.index[0]),
        "end": _json_timestamp(combined.index[-1]),
        "energy_total": float(combined["energy"].sum()),
        "timezone": timezone,
        "timestamp_position": timestamp_position,
        "interval_minutes": interval_minutes,
        "energy_unit": energy_unit,
        "input_kind": kind,
        "dst_policy": dst,
        "inferred_dst_row_count": inferred_dst_rows,
        "missing_interval_count": sum(gap["missing_intervals"] for gap in gaps),
        "gap_ranges": gaps,
        "partial_hour_count": int(
            (combined.tz_convert("UTC").resample("h").size() % (60 // interval_minutes) != 0).sum()
        ),
        "required_missing_counts": required_missing,
        "reference_missing_counts": null_counts,
        "timestamp_column": timestamp,
    }
    combined.attrs = {
        "frequency": f"{interval_minutes}min",
        "duration_unit": "hour",
        "energy_unit": energy_unit,
        "temperature_unit": "C",
        "timezone": timezone,
        "timestamp_position": "start",
        "input_kind": kind,
        "interval_minutes": interval_minutes,
    }
    return MeterBatch(combined, report)


def hourly_meter_records(batch: MeterBatch) -> MeterBatch:
    """Aggregate intervals in UTC-safe hours, retaining wholly missing hours."""
    frame = batch.records
    if frame.empty:
        raise ValueError("Meter records are empty")
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.tz is None
        or frame.index.isna().any()
    ):
        raise ValueError("Meter records require a valid timezone-aware DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Meter records must have a unique chronological index")
    if "energy" not in frame:
        raise ValueError("Meter records must contain energy")
    interval_minutes = frame.attrs.get("interval_minutes", batch.report.get("interval_minutes", 15))
    _validated_interval(interval_minutes)
    utc = frame.copy()
    try:
        values = pd.to_numeric(utc.energy, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Meter interval energy must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Meter interval energy contains missing or nonfinite values")
    utc["energy"] = values
    utc.index = utc.index.tz_convert("UTC")
    if (utc.index != utc.index.floor(f"{interval_minutes}min")).any():
        raise ValueError("Meter intervals must align to the declared UTC interval grid")
    utc.index.name = "timestamp"
    expected = 60 // interval_minutes
    grouped = utc.resample("h")
    counts = grouped["energy"].count()
    energy = grouped["energy"].sum(min_count=1)
    start = utc.index.min().floor("h")
    end = utc.index.max().floor("h")
    full_index = pd.date_range(start, end, freq="h", tz="UTC", name="timestamp")
    energy = energy.reindex(full_index)
    counts = counts.reindex(full_index, fill_value=0)
    result = pd.DataFrame(
        {
            "energy": energy,
            "duration": counts * interval_minutes / 60,
            "coverage": counts / expected,
        },
        index=full_index,
    )
    dropped_nonnumeric: list[str] = []
    for column in utc.columns:
        if column in {
            "energy",
            "source_file",
            "source_row",
            "timestamp_end",
            batch.report.get("timestamp_column", "Date"),
        }:
            continue
        if pd.api.types.is_numeric_dtype(utc[column].dtype):
            result[column] = grouped[column].mean().reindex(full_index)
        else:
            dropped_nonnumeric.append(str(column))
    result.index = result.index.tz_convert(frame.index.tz)
    result.index.name = "timestamp"
    # Compute energy conservation with tolerance suitable for floating-point interval conversion.
    source_total = float(utc["energy"].sum())
    hourly_total = float(result["energy"].sum(min_count=1))
    if not np.isclose(source_total, hourly_total, rtol=1e-12, atol=1e-10):
        raise ArithmeticError("Hourly aggregation changed total meter energy")
    missing_hours = int(result["energy"].isna().sum())
    partial_hours = int(((result["coverage"] > 0) & (result["coverage"] < 1)).sum())
    report = dict(batch.report)
    report.update(
        {
            "hourly_row_count": len(result),
            "hourly_energy_total": hourly_total,
            "missing_hour_count": missing_hours,
            "partial_coverage_hour_count": partial_hours,
            "dropped_nonnumeric_reference_columns": dropped_nonnumeric,
        }
    )
    attrs = batch.records.attrs.copy()
    attrs.update(
        {
            "frequency": "hourly",
            "duration_unit": "hour",
            "energy_unit": batch.report.get("energy_unit", attrs.get("energy_unit", "kWh")),
            "temperature_unit": "C",
            "timezone": str(result.index.tz),
            "timestamp_position": "start",
            "input_kind": batch.report.get("input_kind", attrs.get("input_kind")),
        }
    )
    result.attrs = attrs
    return MeterBatch(result, report)
