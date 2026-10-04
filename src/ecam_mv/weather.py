"""Open-Meteo archive/geocoding client and UTC-safe meter weather joins."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .ingestion import MeterBatch

WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "shortwave_radiation",
)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


def _aware_utc(value: Any, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _json_response(url: str, *, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "ecam-mv/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError:
        raise
    except (URLError, TimeoutError, OSError) as exc:
        raise OSError(f"Open-Meteo request failed: {exc}") from exc
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Open-Meteo returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Open-Meteo response must be a JSON object")
    if payload.get("error"):
        raise ValueError(f"Open-Meteo API error: {payload.get('reason', 'unknown API error')}")
    return payload


class OpenMeteoClient:
    """Small standard-library client with bounded retries and a JSON disk cache."""

    def __init__(self, cache_dir: str | Path | None = None, timeout: float = 30, retries: int = 2):
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be a positive finite number")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("retries must be a nonnegative integer")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.timeout = float(timeout)
        self.retries = retries

    def _request_json(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return _json_response(url, timeout=self.timeout)
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise OSError(f"Open-Meteo HTTP {exc.code}: {exc.reason}") from exc
            except OSError as exc:
                last_error = exc
                if attempt == self.retries:
                    raise
            time.sleep(min(0.2 * (2**attempt), 1.0))
        raise OSError(f"Open-Meteo request failed: {last_error}")

    def _cached_request(self, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{ARCHIVE_URL}?{urlencode(params, doseq=True)}"
        cache_path: Path | None = None
        if self.cache_dir is not None:
            digest = hashlib.sha256(
                json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            cache_path = self.cache_dir / f"{digest}.json"
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                hourly = cached.get("hourly") if isinstance(cached, dict) else None
                times = hourly.get("time") if isinstance(hourly, dict) else None
                vars_list = params["hourly"].split(",")
                units = cached.get("hourly_units", {}) if isinstance(cached, dict) else {}
                try:
                    cache_index = pd.DatetimeIndex(
                        pd.to_datetime(times, unit="s", utc=True, errors="raise")
                    ).as_unit("ns")
                    expected_index = pd.date_range(
                        pd.Timestamp(params["start_date"], tz="UTC"),
                        pd.Timestamp(params["end_date"], tz="UTC") + pd.Timedelta(hours=23),
                        freq="h",
                        tz="UTC",
                    )
                    grid_valid = pd.DatetimeIndex(cache_index).equals(expected_index)
                except (TypeError, ValueError, OverflowError):
                    grid_valid = False
                valid = (
                    isinstance(times, list)
                    and grid_valid
                    and all(
                        isinstance(units, dict)
                        and isinstance(units.get(variable), str)
                        and bool(units[variable])
                        and isinstance(hourly.get(variable), list)
                        and len(hourly[variable]) == len(times)
                        and all(
                            value is None
                            or (isinstance(value, (int, float)) and math.isfinite(value))
                            for value in hourly[variable]
                        )
                        for variable in vars_list
                    )
                )
                if valid:
                    return cached
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                pass
        payload = self._request_json(url)
        if cache_path is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return payload

    def geocode(
        self, name: str, *, country: str = "US", region: str | None = None
    ) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty place name")
        if not isinstance(country, str) or len(country.strip()) != 2:
            raise ValueError("country must be a two-letter country code")
        if region is not None and (not isinstance(region, str) or not region.strip()):
            raise ValueError("region must be a nonempty region name")
        params = {"name": name.strip(), "count": 100, "language": "en", "format": "json"}
        payload = self._request_json(f"{GEOCODING_URL}?{urlencode(params)}")
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Open-Meteo geocoding response has invalid results")
        wanted_name, wanted_country = name.strip().casefold(), country.strip().casefold()
        matches = [
            item
            for item in results
            if isinstance(item, dict)
            and str(item.get("name", "")).casefold() == wanted_name
            and str(item.get("country_code", "")).casefold() == wanted_country
        ]
        if region is not None:
            wanted_region = region.strip().casefold()
            matches = [
                item for item in matches if str(item.get("admin1", "")).casefold() == wanted_region
            ]
        if len(matches) != 1:
            qualifier = f" in {country}" + (f", {region}" if region else "")
            if not matches:
                raise ValueError(f"No exact geocoding match for {name!r}{qualifier}")
            candidates = [
                f"{m.get('name')}, {m.get('admin1')}, {m.get('country_code')}" for m in matches
            ]
            raise ValueError(
                f"Ambiguous geocoding result for {name!r}{qualifier}: {candidates}; specify region"
            )
        result = matches[0].copy()
        try:
            latitude, longitude = float(result["latitude"]), float(result["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Geocoding result has invalid coordinates") from exc
        if not (
            math.isfinite(latitude)
            and -90 <= latitude <= 90
            and math.isfinite(longitude)
            and -180 <= longitude <= 180
        ):
            raise ValueError("Geocoding result has invalid coordinates")
        result["latitude"], result["longitude"] = latitude, longitude
        return result

    def fetch_hourly(
        self,
        latitude: float,
        longitude: float,
        start: Any,
        end: Any,
        *,
        variables: Iterable[str] = WEATHER_VARIABLES,
        model: str = "era5",
    ) -> pd.DataFrame:
        """Fetch hourly archive coverage on the UTC grid spanning inclusive bounds."""
        try:
            latitude, longitude = float(latitude), float(longitude)
        except (TypeError, ValueError) as exc:
            raise ValueError("latitude and longitude must be numeric") from exc
        if not (
            math.isfinite(latitude)
            and -90 <= latitude <= 90
            and math.isfinite(longitude)
            and -180 <= longitude <= 180
        ):
            raise ValueError("latitude/longitude are outside valid ranges")
        begin, finish = _aware_utc(start, "start"), _aware_utc(end, "end")
        if begin > finish:
            raise ValueError("start must be at or before end")
        begin, finish = begin.floor("h"), finish.floor("h")
        selected = tuple(variables)
        if (
            not selected
            or any(not isinstance(v, str) or not v for v in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("variables must be a nonempty sequence of unique names")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        chunks: list[pd.DataFrame] = []
        locations: list[dict[str, Any]] = []
        response_units: dict[str, str] | None = None
        cursor = begin
        last_payload: dict[str, Any] = {}
        while cursor <= finish:
            year_end = pd.Timestamp(datetime(cursor.year, 12, 31, 23, tzinfo=UTC))
            chunk_end = min(finish, year_end, cursor + pd.Timedelta(days=365, hours=23))
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": cursor.date().isoformat(),
                "end_date": chunk_end.date().isoformat(),
                "hourly": ",".join(selected),
                "timezone": "GMT",
                "timeformat": "unixtime",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "precipitation_unit": "mm",
                "models": model,
            }
            payload = last_payload = self._cached_request(params)
            try:
                current_units = payload["hourly_units"]
                current_units = {name: current_units[name] for name in selected}
            except (KeyError, TypeError) as exc:
                raise ValueError("Open-Meteo response is missing hourly units") from exc
            if any(not isinstance(unit, str) or not unit for unit in current_units.values()):
                raise ValueError("Open-Meteo returned invalid hourly units")
            if response_units is None:
                response_units = current_units
            elif current_units != response_units:
                raise ValueError("Open-Meteo hourly units changed between archive chunks")
            locations.append(
                {
                    key: payload.get(key)
                    for key in (
                        "latitude",
                        "longitude",
                        "elevation",
                        "timezone",
                        "timezone_abbreviation",
                        "utc_offset_seconds",
                    )
                    if key in payload
                }
            )
            hourly = payload.get("hourly")
            if not isinstance(hourly, dict) or "time" not in hourly:
                raise ValueError("Open-Meteo response is missing hourly timestamps")
            times = hourly["time"]
            if not isinstance(times, list):
                raise ValueError("Open-Meteo hourly timestamps must be a list")
            try:
                index = pd.DatetimeIndex(
                    pd.to_datetime(times, unit="s", utc=True, errors="raise")
                ).as_unit("ns")
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Open-Meteo returned invalid epoch timestamps") from exc
            api_start = cursor.normalize()
            api_end = chunk_end.normalize() + pd.Timedelta(hours=23)
            expected = pd.date_range(api_start, api_end, freq="h", tz="UTC")
            if not index.equals(expected):
                raise ValueError(
                    "Open-Meteo timestamps are missing, duplicated, unordered, or "
                    "outside the requested hourly grid"
                )
            columns: dict[str, list[float]] = {}
            for variable in selected:
                values = hourly.get(variable)
                if not isinstance(values, list) or len(values) != len(index):
                    raise ValueError(f"Open-Meteo response has invalid length for {variable}")
                converted = []
                for value in values:
                    if value is None:
                        converted.append(np.nan)
                        continue
                    try:
                        number = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Open-Meteo returned a nonnumeric {variable} value"
                        ) from exc
                    if not math.isfinite(number):
                        raise ValueError(f"Open-Meteo returned a nonfinite {variable} value")
                    converted.append(number)
                columns[variable] = converted
            chunk = pd.DataFrame(columns, index=index)
            chunks.append(chunk.loc[cursor:chunk_end])
            cursor = chunk_end + pd.Timedelta(hours=1)
        result = pd.concat(chunks)
        result.index.name = "timestamp"
        units = response_units or {}
        result.attrs = {
            "weather_source": "Open-Meteo Archive API",
            "attribution": "Open-Meteo; Copernicus ERA5",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "documentation_url": "https://open-meteo.com/en/docs/historical-weather-api",
            "model": model,
            "latitude": latitude,
            "longitude": longitude,
            "returned_latitude": last_payload.get("latitude"),
            "returned_longitude": last_payload.get("longitude"),
            "api_url": ARCHIVE_URL,
            "returned_locations": locations,
            "weather_timezone": "UTC",
            "timeformat": "unixtime",
            "weather_units": units,
            "hourly_value_semantics": (
                "temperature_2m is an instantaneous hourly value; precipitation is the "
                "preceding-hour sum in mm; shortwave_radiation is the preceding-hour "
                "mean in W/m². Joining repeats each source-hour value on matching "
                "meter intervals."
            ),
        }
        return result


def join_weather(
    records: pd.DataFrame, weather: pd.DataFrame, *, missing: str = "raise"
) -> MeterBatch:
    """Join hourly weather at meter interval starts in UTC, retaining all rows."""
    if missing not in {"raise", "flag"}:
        raise ValueError("missing must be 'raise' or 'flag'")
    for frame, label in ((records, "meter records"), (weather, "weather")):
        if (
            not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.tz is None
            or frame.index.isna().any()
        ):
            raise ValueError(f"{label} require a valid timezone-aware DatetimeIndex")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{label} require a unique chronological DatetimeIndex")
    if weather.columns.duplicated().any():
        raise ValueError("weather columns must be unique")
    if records.columns.duplicated().any():
        raise ValueError("meter columns must be unique")
    if "_weather_hour" in weather.columns:
        raise ValueError("_weather_hour is reserved for the weather join")
    collisions = set(records.columns).intersection(weather.columns)
    collisions.update(
        set(records.columns).intersection({"temperature", "drybulb", "weather_missing"})
    )
    if collisions:
        raise ValueError(f"Weather columns collide with meter columns: {sorted(collisions)}")
    if "temperature_2m" not in weather:
        raise ValueError("weather must contain temperature_2m")
    drybulb_unit = weather.attrs.get("weather_units", {}).get("temperature_2m", "C")
    drybulb_unit = {"°C": "C", "°F": "F"}.get(drybulb_unit, drybulb_unit)
    if drybulb_unit not in {"C", "F"}:
        raise ValueError("Dry-bulb weather unit must be C or F")
    weather_numeric = weather.copy()
    for column in weather_numeric.columns:
        try:
            values = pd.to_numeric(weather_numeric[column], errors="raise").to_numpy(
                dtype=float, na_value=np.nan
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weather reference {column} must be numeric or missing") from exc
        if np.isinf(values).any():
            raise ValueError(f"weather reference {column} contains nonfinite values")
        weather_numeric[column] = values
    weather_utc = weather_numeric
    weather_utc.index = weather.index.tz_convert("UTC")
    if (
        (weather_utc.index.minute != 0)
        | (weather_utc.index.second != 0)
        | (weather_utc.index.microsecond != 0)
        | (weather_utc.index.nanosecond != 0)
    ).any():
        raise ValueError("weather timestamps must lie on the hourly UTC grid")
    floored = records.index.tz_convert("UTC").floor("h")
    lookup = weather_utc.copy()
    lookup.index.name = "_weather_hour"
    left = pd.DataFrame({"_weather_hour": floored})
    merged = left.merge(
        lookup.reset_index(), how="left", on="_weather_hour", validate="many_to_one", sort=False
    )
    hour_matched = floored.isin(weather_utc.index)
    matched = merged["temperature_2m"].notna()
    result = records.copy()
    for column in weather.columns:
        result[column] = merged[column].to_numpy()
    result["drybulb"] = result["temperature_2m"]
    result["temperature"] = result["temperature_2m"]
    result["weather_missing"] = ~matched.to_numpy()
    missing_count = int((~hour_matched).sum())
    bad_drybulb_count = int((hour_matched & ~matched.to_numpy()).sum())
    if (missing_count or bad_drybulb_count) and missing == "raise":
        stamps = records.index[~matched.to_numpy()]
        raise ValueError(
            f"Missing drybulb weather for {missing_count + bad_drybulb_count} meter row(s) "
            f"({missing_count} absent hours, {bad_drybulb_count} null temperatures), "
            f"e.g. {stamps[:8].tolist()}"
        )
    report = {
        "row_count": len(result),
        "matched_rows": int(hour_matched.sum()),
        "unmatched_rows": missing_count,
        "bad_drybulb_rows": bad_drybulb_count,
        "reference_missing_counts": {
            str(col): int(result[col].isna().sum()) for col in weather.columns
        },
        "weather_join": (
            "interval-start UTC floor-hour; temperature is the instantaneous hourly value"
        ),
        "precipitation_semantics": (
            "hourly precipitation is the preceding-hour sum and is repeated for "
            "joined meter intervals; it is not allocated as an interval total"
        ),
    }
    attrs = records.attrs.copy()
    meter_timezone = attrs.get("timezone", str(records.index.tz))
    attrs.update(weather.attrs)
    attrs.update(
        {
            "timezone": meter_timezone,
            "weather_timezone": "UTC",
            "predictor": "temperature",
            "predictor_unit": drybulb_unit,
            "temperature_unit": drybulb_unit,
            "weather_join": "UTC floor-hour of meter interval start",
        }
    )
    result.attrs = attrs
    return MeterBatch(result, report)
