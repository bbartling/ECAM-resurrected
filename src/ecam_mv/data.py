"""Explicit meter and bill semantics for energy measurement and verification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date

import numpy as np
import pandas as pd


def _numeric(series: pd.Series, name: str) -> np.ndarray:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing or nonfinite values")
    return values


def _timestamps(values: pd.Series, timezone: str | None = None) -> pd.DatetimeIndex:
    try:
        result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise", format="mixed"))
    except (TypeError, ValueError) as exc:
        if timezone is None:
            raise ValueError("Invalid or mixed-offset timestamps; supply a timezone") from exc
        try:
            result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise", utc=True))
        except (TypeError, ValueError) as inner:
            raise ValueError("Invalid timestamps") from inner
    if result.isna().any():
        raise ValueError("Timestamps contain missing values")
    if timezone is not None:
        try:
            result = (
                result.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
                if result.tz is None
                else result.tz_convert(timezone)
            )
        except Exception as exc:
            raise ValueError("Timezone or DST ambiguity: supply explicit UTC offsets") from exc
    if result.has_duplicates:
        raise ValueError("Duplicate timestamps")
    return result


def prepare_hourly(
    frame: pd.DataFrame,
    *,
    timestamp: str = "timestamp",
    usage: str = "energy",
    temperature: str | None = "temperature",
    kind: str = "energy",
    interval_minutes: int = 60,
    timestamp_position: str = "start",
    timezone: str | None = None,
    energy_unit: str = "kWh",
    temperature_unit: str = "C",
    min_coverage: float = 1.0,
    allow_negative: bool = False,
) -> pd.DataFrame:
    """Aggregate fixed intervals to hours; power is an interval-average rate.

    ``kind='power'`` means kW when energy_unit is kWh (or the corresponding
    energy-unit/hour rate). Interval-end timestamps are shifted to their start.
    No interpolation is performed. Partial hours require explicit min_coverage;
    their duration is measured exposure, so energy and predictions match.
    """
    if frame.empty:
        raise ValueError("Meter data is empty")
    if kind not in {"energy", "power"}:
        raise ValueError("kind must be 'energy' or 'power'")
    if timestamp_position not in {"start", "end"}:
        raise ValueError("timestamp_position must be 'start' or 'end'")
    if (
        isinstance(interval_minutes, bool)
        or not isinstance(interval_minutes, int)
        or interval_minutes <= 0
        or 60 % interval_minutes
    ):
        raise ValueError("interval_minutes must be a positive integer divisor of 60")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    required = [timestamp, usage] + ([temperature] if temperature is not None else [])
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(set(required) - set(frame.columns))}")
    index = _timestamps(frame[timestamp], timezone)
    delta = pd.Timedelta(minutes=interval_minutes)
    if timestamp_position == "end":
        index = index - delta
    if (
        (index.minute % interval_minutes != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
        or (index.nanosecond != 0).any()
    ):
        raise ValueError("Timestamps must align to the declared interval grid")
    values = _numeric(frame[usage], usage)
    if not allow_negative and (values < 0).any():
        raise ValueError("Negative meter values; set allow_negative=True for net export")
    temps = _numeric(frame[temperature], temperature) if temperature else np.zeros(len(frame))
    dt_hours = interval_minutes / 60
    records = pd.DataFrame(
        {"energy": values * dt_hours if kind == "power" else values, "temperature": temps},
        index=index,
    ).sort_index()
    grouped = records.resample("h")
    counts = grouped.size()
    coverage = counts / (60 // interval_minutes)
    if (coverage < min_coverage).any():
        first = coverage[coverage < min_coverage].index[0]
        raise ValueError(f"Insufficient meter coverage at {first}; no automatic gap filling")
    result = pd.DataFrame(
        {
            "energy": grouped.energy.sum(min_count=1),
            "temperature": grouped.temperature.mean(),
            "duration": counts * dt_hours,
            "coverage": coverage,
        }
    )
    result.index.name = "timestamp"
    result.attrs = {
        "frequency": "hourly",
        "duration_unit": "hour",
        "energy_unit": energy_unit,
        "temperature_unit": temperature_unit,
        "predictor": temperature if temperature is not None else "none",
        "predictor_unit": temperature_unit,
        "timezone": str(result.index.tz) if result.index.tz else None,
        "timestamp_position": "start",
        "input_kind": kind,
        "interval_minutes": interval_minutes,
    }
    return result


def prepare_billing(
    frame: pd.DataFrame,
    *,
    start: str = "start",
    end: str = "end",
    usage: str = "energy",
    temperature: str | None = "temperature",
    energy_unit: str = "kWh",
    temperature_unit: str = "C",
    allow_negative: bool = False,
) -> pd.DataFrame:
    """Validate nonoverlapping bills on half-open [start, end) calendar dates.

    Temperature is the mean over the exact billing period, supplied by the user.
    End is exclusive, independent of any utility's printed date convention.
    """
    if frame.empty:
        raise ValueError("Billing data is empty")
    required = [start, end, usage] + ([temperature] if temperature is not None else [])
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(set(required) - set(frame.columns))}")
    starts, ends = _timestamps(frame[start]), _timestamps(frame[end])
    if starts.tz is not None or ends.tz is not None:
        raise ValueError("Bills require timezone-free calendar dates")
    if (starts != starts.normalize()).any() or (ends != ends.normalize()).any():
        raise ValueError("Bills require midnight calendar dates")
    days = (ends - starts).total_seconds().to_numpy() / 86400
    if (days <= 0).any():
        raise ValueError("Each billing end must be after its start")
    energy = _numeric(frame[usage], usage)
    if not allow_negative and (energy < 0).any():
        raise ValueError("Negative billing energy; set allow_negative=True for net export")
    temps = _numeric(frame[temperature], temperature) if temperature else np.zeros(len(frame))
    result = pd.DataFrame(
        {"end": ends, "energy": energy, "temperature": temps, "duration": days}, index=starts
    ).sort_index()
    if len(result) > 1 and (result.index[1:] < pd.DatetimeIndex(result.end.iloc[:-1])).any():
        raise ValueError("Billing periods overlap")
    result.index.name = "timestamp"
    result.attrs = {
        "frequency": "billing",
        "duration_unit": "day",
        "energy_unit": energy_unit,
        "temperature_unit": temperature_unit,
        "predictor": temperature if temperature is not None else "none",
        "predictor_unit": temperature_unit,
        "timezone": None,
        "timestamp_position": "start",
    }
    return result


def model_records(
    records: pd.DataFrame,
    *,
    predictor: str | None = None,
    predictor_unit: str | None = None,
    missing: str = "raise",
    min_coverage: float = 1.0,
) -> pd.DataFrame:
    """Select one numeric regression column and audit any explicit exclusions.

    Reference-column nulls are retained. Missing meter energy, a missing selected
    predictor, and incomplete hours fail by default. ``missing='drop'`` is an
    explicit opt-in; the exclusion counts and known excluded energy remain in
    ``attrs['record_filter']``. No interpolation or zero-filling is performed.
    The historical internal ``temperature`` field stores the selected predictor;
    the actual name and units are recorded to prevent accidental model mismatch.
    """
    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be 'raise' or 'drop'")
    if not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if (
        not isinstance(records.index, pd.DatetimeIndex)
        or records.index.isna().any()
        or records.index.has_duplicates
        or not records.index.is_monotonic_increasing
    ):
        raise ValueError("Records require unique, valid, chronological timestamps")
    if records.attrs.get("frequency") not in {"hourly", "billing"}:
        raise ValueError("Prepare hourly or billing records first")
    column = predictor or records.attrs.get("predictor", "temperature")
    if column == "none":
        column = "temperature"
    required = {"energy", "duration", column}
    if not required.issubset(records.columns):
        raise ValueError(f"Missing model columns: {sorted(required - set(records.columns))}")
    arrays = {
        name: pd.to_numeric(records[name], errors="coerce").to_numpy(dtype=float)
        for name in required
    }
    reasons = {
        "invalid_energy": ~np.isfinite(arrays["energy"]),
        "invalid_duration": ~np.isfinite(arrays["duration"]) | (arrays["duration"] <= 0),
        "invalid_predictor": ~np.isfinite(arrays[column]),
    }
    if "coverage" in records and records.attrs["frequency"] == "hourly":
        coverage = pd.to_numeric(records.coverage, errors="coerce").to_numpy(dtype=float)
        reasons["incomplete_meter_coverage"] = ~np.isfinite(coverage) | (coverage < min_coverage)
        if (coverage > 1).any():
            raise ValueError("Meter coverage exceeds 1; check duplicated intervals")
    bad = np.logical_or.reduce(list(reasons.values()))
    audit = {
        "input_records": len(records),
        "excluded_records": int(bad.sum()),
        "retained_records": int((~bad).sum()),
        "predictor": column,
        "policy": missing,
        "min_coverage": min_coverage,
        "reason_counts": {name: int(mask.sum()) for name, mask in reasons.items()},
        "excluded_known_energy": float(
            np.sum(arrays["energy"][bad & np.isfinite(arrays["energy"])])
        ),
        "example_excluded_timestamps": [t.isoformat() for t in records.index[bad][:5]],
    }
    if bad.any() and missing == "raise":
        raise ValueError(
            f"{bad.sum()} unusable model records for predictor {column!r}: "
            f"{audit['reason_counts']}; first at {records.index[bad][0]}. "
            "Review the gaps or explicitly use missing='drop' / --exclude-incomplete"
        )
    if records.empty or bad.all():
        raise ValueError("No usable model records remain")
    result = records.loc[~bad].copy()
    result["energy"] = arrays["energy"][~bad]
    result["duration"] = arrays["duration"][~bad]
    result["temperature"] = arrays[column][~bad]
    source_units = records.attrs.get("weather_units", {})
    unit = predictor_unit or source_units.get(column)
    if unit is None:
        previous = records.attrs.get("predictor", "temperature")
        if column != previous and column != "temperature":
            raise ValueError("Supply predictor_unit for a custom regression column")
        unit = records.attrs.get("predictor_unit", records.attrs.get("temperature_unit", "C"))
    unit = {"°C": "C", "°F": "F"}.get(unit, unit)
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("predictor_unit must be a nonempty string")
    result.attrs = records.attrs.copy()
    result.attrs.update(
        predictor=records.attrs.get("predictor", column) if predictor is None else predictor,
        predictor_unit=unit,
        temperature_unit=unit,
        record_filter=audit,
    )
    return result


def calendar_features(
    frame: pd.DataFrame,
    *,
    holidays: Iterable[date | str] = (),
    occupancy: Mapping[int, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Add local calendar fields; weekdays use Monday=0, Sunday=6.

    Occupancy ranges are half-open hours, e.g. {0: (8, 18)} for Mondays.
    An overnight range such as (22, 6) belongs to its starting weekday.
    Holidays suppress occupancy and take precedence over weekend labels.
    """
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.isna().any():
        raise ValueError("A valid DatetimeIndex is required")
    schedule = occupancy if occupancy is not None else {i: (8, 18) for i in range(5)}
    for weekday, hours in schedule.items():
        if weekday not in range(7) or len(hours) != 2:
            raise ValueError("Schedule keys must be weekdays 0..6 with (start, end) hours")
        if any(isinstance(h, bool) or not isinstance(h, int) for h in hours):
            raise ValueError("Schedule hours must be integers")
        if not 0 <= hours[0] < 24 or not 0 <= hours[1] <= 24 or hours[0] == hours[1]:
            raise ValueError("Schedule hours must define a nonempty range in 0..24")
    holiday_dates = {pd.Timestamp(d).date() for d in holidays}
    result = frame.copy()
    index = result.index
    weekday, hour = index.dayofweek.to_numpy(), index.hour.to_numpy()
    holiday = np.array([d in holiday_dates for d in index.date])
    occupied = np.zeros(len(index), dtype=bool)
    for day, (begin, finish) in schedule.items():
        if begin < finish:
            occupied |= (weekday == day) & (hour >= begin) & (hour < finish)
        else:
            occupied |= ((weekday == day) & (hour >= begin)) | (
                (weekday == (day + 1) % 7) & (hour < finish)
            )
    result["year"] = index.year
    result["month"] = index.month
    result["weekday"] = weekday
    result["hour"] = hour
    result["holiday"] = holiday
    result["daytype"] = np.where(holiday, "holiday", np.where(weekday >= 5, "weekend", "weekday"))
    result["occupied"] = occupied & ~holiday
    result.attrs = frame.attrs.copy()
    return result
