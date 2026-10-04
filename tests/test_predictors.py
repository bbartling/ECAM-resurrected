import json

import numpy as np
import pandas as pd
import pytest

from ecam_mv import fit_baseline, model_records, prepare_hourly
from ecam_mv.cli import main
from ecam_mv.plots import report_plots
from ecam_mv.workflows import load_records, save_records


def records():
    x = np.tile(np.arange(24), 3)
    data = prepare_hourly(
        pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=72, freq="h", tz="America/Chicago"
                ),
                "energy": 10 + 0.5 * x,
                "temperature": x,
            }
        )
    )
    data["relative_humidity_2m"] = x
    data["reference_only"] = np.nan
    data.attrs["weather_units"] = {"relative_humidity_2m": "%"}
    return data


def test_arbitrary_numeric_predictor_fit_and_round_trip(tmp_path):
    chosen = model_records(records(), predictor="relative_humidity_2m")
    assert chosen.attrs["predictor_unit"] == "%"
    assert chosen.reference_only.isna().all()
    fit = fit_baseline(chosen, model="2p")
    assert fit.models["all"].coefficients == pytest.approx([10, 0.5])
    save_records(chosen, tmp_path / "records.csv")
    restored = load_records(tmp_path / "records.csv")
    assert restored.index.equals(chosen.index)
    assert fit.predict(restored).to_numpy() == pytest.approx(chosen.energy)


def test_predictor_identity_cannot_be_silently_switched():
    data = records()
    fit = fit_baseline(model_records(data, predictor="relative_humidity_2m"), model="2p")
    changed = model_records(data, predictor="temperature", predictor_unit="%")
    with pytest.raises(ValueError, match="predictor mismatch"):
        fit.predict(changed)


@pytest.mark.parametrize("value", [np.nan, np.inf, "invalid"])
def test_selected_missing_bad_values_require_opt_in(value):
    data = records()
    data["relative_humidity_2m"] = data.relative_humidity_2m.astype(object)
    data.loc[data.index[3], "relative_humidity_2m"] = value
    with pytest.raises(ValueError, match="unusable model records"):
        model_records(data, predictor="relative_humidity_2m")
    chosen = model_records(data, predictor="relative_humidity_2m", missing="drop")
    assert len(chosen) == 71
    assert chosen.attrs["record_filter"]["excluded_records"] == 1
    assert chosen.attrs["record_filter"]["excluded_known_energy"] == pytest.approx(11.5)


def test_incomplete_hour_not_converted_to_zero_or_complete():
    data = records()
    data.loc[data.index[0], ["energy", "duration", "coverage"]] = [np.nan, 0, 0]
    data.loc[data.index[1], ["energy", "duration", "coverage"]] = [2, 0.25, 0.25]
    with pytest.raises(ValueError, match="2 unusable"):
        model_records(data)
    clean = model_records(data, missing="drop")
    assert len(clean) == 70
    assert clean.attrs["record_filter"]["excluded_known_energy"] == 2
    assert np.isnan(data.energy.iloc[0])


def test_empty_all_bad_and_missing_predictor_errors():
    data = records()
    with pytest.raises(ValueError, match="Missing model columns"):
        model_records(data, predictor="unknown")
    data["temperature"] = np.nan
    with pytest.raises(ValueError, match="No usable"):
        model_records(data, missing="drop")


def test_custom_column_units_are_required():
    data = records()
    data["production"] = 1.0
    with pytest.raises(ValueError, match="predictor_unit"):
        model_records(data, predictor="production")
    assert (
        model_records(data, predictor="production", predictor_unit="widgets").attrs[
            "predictor_unit"
        ]
        == "widgets"
    )


def test_prepared_cli_filters_explicitly_and_reports(tmp_path, capsys):
    data = records()
    data.loc[data.index[3], "relative_humidity_2m"] = np.nan
    csv, baseline = tmp_path / "records.csv", tmp_path / "baseline.json"
    save_records(data, csv)
    args = [
        "fit",
        str(csv),
        "--prepared",
        "--predictor",
        "relative_humidity_2m",
        "--model",
        "2p",
        "--out",
        str(baseline),
    ]
    assert main(args) == 2
    assert not baseline.exists()
    assert main([*args, "--exclude-incomplete"]) == 0
    doc = json.loads(baseline.read_text())
    assert doc["metadata"]["training_records"] == 71
    assert doc["metadata"]["record_filter"]["excluded_records"] == 1
    assert "excluded_records" in capsys.readouterr().err


def test_prepared_missing_sidecar_error(tmp_path):
    csv = tmp_path / "records.csv"
    records().to_csv(csv)
    with pytest.raises(ValueError, match="metadata sidecar"):
        load_records(csv)


def test_non_temperature_chart_is_labelled_correctly(tmp_path, monkeypatch):
    import ecam_mv.plots as plots

    data = model_records(records(), predictor="relative_humidity_2m")
    fit = fit_baseline(data, model="2p")
    seen = []
    actual = plots.model_diagnostics

    def capture(*args, **kwargs):
        fig = actual(*args, **kwargs)
        seen.append(fig.axes[0].get_xlabel())
        return fig

    monkeypatch.setattr(plots, "model_diagnostics", capture)
    report_plots(fit, data, directory=tmp_path)
    assert seen == ["relative_humidity_2m (%)"]
