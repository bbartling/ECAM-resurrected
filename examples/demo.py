"""Synthetic M&V demonstration; outputs are not measured building savings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecam_mv import avoided_energy, calendar_features, fit_baseline, prepare_billing, prepare_hourly
from ecam_mv.plots import report_plots


def main(directory: str | Path):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(42)
    n = 24 * 120
    temperature = 16 + 12 * np.sin(np.arange(n) / 210) + 4 * np.sin(np.arange(n) / 4)
    energy = 20 + 1.8 * np.maximum(temperature - 18, 0) + random.normal(0, 1.2, n)
    training = calendar_features(
        prepare_hourly(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
                    "energy": energy,
                    "temperature": temperature,
                }
            )
        )
    )
    baseline = fit_baseline(training, model="3pC")
    baseline.save(destination / "hourly_baseline.json")
    reporting = calendar_features(
        prepare_hourly(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
                    "energy": energy - 3 + random.normal(0, 0.2, n),
                    "temperature": temperature,
                }
            )
        )
    )
    result = avoided_energy(baseline, reporting, confidence=0.9, uncertainty_method="ecam")
    result.save(destination / "hourly_savings.json", destination / "hourly_savings.csv")
    report_plots(baseline, training, directory=destination / "hourly_baseline_charts")
    report_plots(baseline, reporting, result=result, directory=destination / "hourly_charts")

    dates = pd.date_range("2022-01-01", periods=25, freq="MS")
    temperatures = 12 + 13 * np.sin(np.arange(24) * 2 * np.pi / 12)
    duration = np.diff(dates).astype("timedelta64[D]").astype(float)
    rates = 90 + 5 * np.maximum(temperatures - 18, 0) + random.normal(0, 2, 24)
    bills = prepare_billing(
        pd.DataFrame(
            {
                "start": dates[:-1],
                "end": dates[1:],
                "temperature": temperatures,
                "energy": rates * duration,
            }
        )
    )
    monthly_baseline = fit_baseline(bills, model="3pC")
    monthly_baseline.save(destination / "monthly_baseline.json")
    monthly_reporting = bills.copy()
    monthly_reporting.energy -= 10 * duration
    monthly_result = avoided_energy(monthly_baseline, monthly_reporting, uncertainty_method="ecam")
    monthly_result.save(destination / "monthly_savings.json", destination / "monthly_savings.csv")
    report_plots(
        monthly_baseline,
        monthly_reporting,
        result=monthly_result,
        directory=destination / "monthly_charts",
    )
    print(f"Synthetic demonstrations saved in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="work/demo")
    main(parser.parse_args().out)
