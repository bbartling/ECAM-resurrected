"""Auditable meter/weather preparation without modifying source CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import _timestamps


def save_records(records: pd.DataFrame, path: str | Path) -> None:
    """Write CSV and a matching metadata sidecar; JSON never contains NaN."""
    path = Path(path)
    metadata = json.dumps(records.attrs, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(path, index_label="timestamp")
    path.with_suffix(path.suffix + ".metadata.json").write_text(metadata, encoding="utf-8")


def load_records(path: str | Path) -> pd.DataFrame:
    """Load prepared records, preserving units and local-time calendar semantics."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if not sidecar.is_file():
        raise ValueError(f"Prepared CSV requires metadata sidecar: {sidecar.name}")
    try:
        attrs = json.loads(sidecar.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid prepared-record metadata JSON") from exc
    if not isinstance(attrs, dict) or attrs.get("frequency") not in {"hourly", "billing"}:
        raise ValueError("Invalid prepared-record frequency metadata")
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise ValueError("Prepared CSV requires a timestamp column")
    frame.index = _timestamps(frame.pop("timestamp"), attrs.get("timezone"))
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Prepared records must be chronological")
    frame.index.name = "timestamp"
    frame.attrs = attrs
    return frame


def prepare_weather_dataset(
    paths,
    destination: str | Path,
    *,
    location: str | None = None,
    region: str | None = None,
    country: str = "US",
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
    timestamp: str = "Date",
    usage: str = "kW",
    energy_unit: str = "kWh",
    kind: str = "power",
    interval_minutes: int = 15,
    timestamp_position: str = "end",
    dst: str = "raise",
    allow_negative: bool = False,
    missing_weather: str = "raise",
    client=None,
) -> dict:
    """Concat intervals, join reference weather, and write hourly M&V inputs.

    Never overwrites source inputs or existing deliverables. Coverage gaps remain
    explicit in the hourly CSV; analysis must opt in to excluding those records.
    Weather is fetched once for the exact UTC-hour span and reused for both joins.
    """
    from .ingestion import concat_meter_csvs, hourly_meter_records
    from .weather import OpenMeteoClient, join_weather

    files = []
    for value in paths:
        path = Path(value)
        files.extend(sorted(path.glob("*.csv")) if path.is_dir() else [path])
    files = sorted({p.resolve() for p in files})
    if not files:
        raise ValueError("No input CSV files found")
    destination = Path(destination)
    names = ["intervals.csv", "hourly.csv", "weather.csv", "audit.json"]
    targets = [destination / name for name in names]
    targets += [p.with_suffix(p.suffix + ".metadata.json") for p in targets[:3]]
    if any(p.resolve() in files for p in targets):
        raise ValueError("Output paths must not overwrite input meter files")
    existing = [p.name for p in targets if p.exists()]
    if existing:
        raise ValueError(f"Output files already exist: {existing}; choose a new output directory")
    if (latitude is None) != (longitude is None):
        raise ValueError("Supply both latitude and longitude")
    if location is not None and latitude is not None:
        raise ValueError("Choose a location name OR explicit coordinates, not both")
    if location is None and latitude is None:
        raise ValueError("Supply a location name or latitude/longitude")
    client = client or OpenMeteoClient(cache_dir=destination / ".weather-cache")
    site = None
    if location is not None:
        site = client.geocode(location, country=country, region=region)
        latitude, longitude = site["latitude"], site["longitude"]
        timezone = timezone or site["timezone"]
    if not timezone:
        raise ValueError("Supply the meter timezone when using explicit coordinates")
    intervals = concat_meter_csvs(
        files,
        timestamp=timestamp,
        usage=usage,
        energy_unit=energy_unit,
        timezone=timezone,
        interval_minutes=interval_minutes,
        timestamp_position=timestamp_position,
        kind=kind,
        dst=dst,
        allow_negative=allow_negative,
    )
    hourly = hourly_meter_records(intervals)
    weather = client.fetch_hourly(
        latitude,
        longitude,
        intervals.records.index[0].tz_convert("UTC"),
        intervals.records.index[-1].tz_convert("UTC"),
    )
    joined_intervals = join_weather(intervals.records, weather, missing=missing_weather)
    joined_hourly = join_weather(hourly.records, weather, missing=missing_weather)
    report = {
        "site": site or {"latitude": latitude, "longitude": longitude},
        "meter": intervals.report,
        "hourly": hourly.report,
        "interval_weather_join": joined_intervals.report,
        "hourly_weather_join": joined_hourly.report,
        "weather": weather.attrs,
        "regression_policy": (
            "Dry bulb by default; gaps retained, not imputed; explicit exclusions required"
        ),
    }
    # Validate serializability before writing any user-facing outputs.
    document = json.dumps(report, indent=2, allow_nan=False)
    for data in (joined_intervals.records, joined_hourly.records, weather):
        json.dumps(data.attrs, allow_nan=False)
    destination.mkdir(parents=True, exist_ok=True)
    save_records(joined_intervals.records, destination / "intervals.csv")
    save_records(joined_hourly.records, destination / "hourly.csv")
    save_records(weather, destination / "weather.csv")
    (destination / "audit.json").write_text(document, encoding="utf-8")
    return report
