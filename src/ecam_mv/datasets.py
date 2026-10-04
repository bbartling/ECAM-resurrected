"""Small, reproducible synthetic practice data (never real savings evidence)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import calendar_features, prepare_hourly


def practice_dataset(*, seed: int = 14) -> pd.DataFrame:
    """Return one synthetic hourly year with reference weather and known model.

    No network or private meter records are needed. This models a fictional
    building, not Lake Geneva measurements. Energy is kWh; dry bulb is Celsius.
    An integer seed recreates exactly the same data on supported NumPy versions.
    """
    index = pd.date_range(
        "2023-01-01", "2024-01-01", freq="h", inclusive="left", tz="America/Chicago"
    )
    rng = np.random.default_rng(seed)
    seasonal = 10 + 15 * np.sin(2 * np.pi * (index.dayofyear.to_numpy() - 105) / 365)
    temperature = seasonal + 4 * np.sin(2 * np.pi * (index.hour.to_numpy() - 9) / 24)
    energy = 30 + 2 * np.maximum(8 - temperature, 0) + 3 * np.maximum(temperature - 20, 0)
    raw = pd.DataFrame(
        {
            "timestamp": index,
            "energy": energy + rng.normal(0, 1, len(index)),
            "temperature": temperature,
        }
    )
    records = calendar_features(prepare_hourly(raw, timezone="America/Chicago"))
    records["temperature_2m"] = temperature
    records["relative_humidity_2m"] = np.clip(
        65 - temperature + rng.normal(0, 3, len(index)), 10, 95
    )
    records["dew_point_2m"] = temperature - 7
    records["wind_speed_10m"] = rng.uniform(0, 25, len(index))
    records.attrs.update(
        synthetic=True,
        description="Fictional annual meter/weather practice data; not measured school energy",
        generator_seed=seed,
        known_model={
            "model": "5p",
            "intercept": 30,
            "heating_slope": 2,
            "cooling_slope": 3,
            "breakpoints_C": [8, 20],
        },
        weather_units={
            "temperature_2m": "C",
            "relative_humidity_2m": "%",
            "dew_point_2m": "C",
            "wind_speed_10m": "km/h",
        },
    )
    return records
