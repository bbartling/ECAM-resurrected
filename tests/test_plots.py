import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from ecam_mv.analysis import avoided_energy, fit_baseline
from ecam_mv.data import calendar_features, prepare_hourly
from ecam_mv.plots import report_plots


def test_plot_bundle_has_history_profiles_models_precision_and_savings(tmp_path):
    x = np.tile(np.arange(24), 3)
    records = calendar_features(
        prepare_hourly(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=len(x), freq="h"),
                    "energy": 10 + 0.5 * x + 0.1 * np.sin(x),
                    "temperature": x,
                }
            )
        )
    )
    baseline = fit_baseline(records, model="2p")
    reporting = records.copy()
    reporting.energy -= 2
    result = avoided_energy(baseline, reporting)
    paths = report_plots(baseline, reporting, result=result, directory=tmp_path)
    assert {p.name for p in paths} == {
        "history.png",
        "load_profiles.png",
        "model_all.png",
        "precision.png",
        "savings.png",
    }
    assert all(p.stat().st_size > 1000 for p in paths)
