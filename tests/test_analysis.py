import numpy as np
import pandas as pd
import pytest

from ecam_mv.analysis import Baseline, avoided_energy, fit_baseline, normalized_savings
from ecam_mv.data import calendar_features, prepare_billing, prepare_hourly


def hourly(start="2024-01-01", offset=0.0):
    x = np.tile(np.arange(24), 14).astype(float)
    return calendar_features(
        prepare_hourly(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(start, periods=len(x), freq="h"),
                    "energy": 10 + 0.5 * x + offset,
                    "temperature": x,
                }
            )
        )
    )


def test_reporting_savings_on_matched_conditions_and_adjustments():
    baseline = fit_baseline(hourly(), model="2p")
    reporting = hourly("2024-02-01", -2)
    result = avoided_energy(baseline, reporting, adjustment=0.5)
    assert result.summary["avoided_energy"] == pytest.approx(2.5 * len(reporting))
    assert result.records.cumulative_savings.iloc[-1] == pytest.approx(2.5 * len(reporting))
    assert result.summary["uncertainty_half_width"] == pytest.approx(0, abs=1e-9)
    assert result.summary["precision"]["passed"]
    assert result.summary["ecam_ashrae_fsu"] is not None


def test_monthly_zero_total_bias_and_actual_bill_lengths():
    dates = pd.date_range("2023-01-01", periods=13, freq="MS")
    lengths = np.diff(dates).astype("timedelta64[D]").astype(float)
    x = np.array([2, 4, 6, 10, 15, 20, 24, 23, 18, 11, 6, 3])
    rate = 10 + 0.5 * x + np.sin(x)
    bills = prepare_billing(
        pd.DataFrame(
            {
                "start": dates[:-1],
                "end": dates[1:],
                "energy": rate * lengths,
                "temperature": x,
            }
        )
    )
    baseline = fit_baseline(bills, model="2p")
    assert baseline.predict(bills).sum() == pytest.approx(bills.energy.sum(), abs=1e-7)
    assert baseline.metadata["ecam_net_determination_bias"] == pytest.approx(0, abs=1e-10)
    result = avoided_energy(baseline, bills)
    assert result.summary["avoided_energy"] == pytest.approx(0, abs=1e-7)
    assert result.summary["uncertainty_half_width"] > 0


def test_saved_baseline_reuse_and_explicit_unit_compatibility(tmp_path):
    before = hourly()
    baseline = fit_baseline(before, model="2p", group_by="daytype")
    path = tmp_path / "baseline.json"
    baseline.save(path)
    restored = Baseline.load(path)
    assert restored.predict(before).to_numpy() == pytest.approx(baseline.predict(before).to_numpy())
    before.attrs["temperature_unit"] = "F"
    with pytest.raises(ValueError, match="temperature_unit mismatch"):
        restored.predict(before)


def test_unknown_reporting_daytype_cannot_be_silently_dropped():
    baseline = fit_baseline(hourly(), model="2p", group_by="daytype")
    reporting = hourly("2024-02-01")
    reporting.loc[reporting.index[0], "daytype"] = "holiday"
    with pytest.raises(ValueError, match="absent"):
        avoided_energy(baseline, reporting)


def test_common_weather_normalization_compares_two_models():
    before, after = fit_baseline(hourly(), model="2p"), fit_baseline(hourly(offset=-3), model="2p")
    conditions = hourly("2024-03-01", offset=500)
    result = normalized_savings(before, after, conditions)
    assert result.summary["normalized_savings"] == pytest.approx(3 * len(conditions))


def test_ecam_normalized_precision_uses_independent_fit_quadrature():
    from ecam_mv.validation import ecam_total_uncertainty

    training = hourly()
    training.energy += np.sin(np.arange(len(training)))
    after_training = training.copy()
    after_training.energy -= 3
    before = fit_baseline(training, model="2p")
    after = fit_baseline(after_training, model="2p")
    conditions = hourly("2024-03-01")
    result = normalized_savings(before, after, conditions, uncertainty_method="ecam")
    components = []
    for fitted in (before, after):
        stats = fitted.metadata["ecam_statistics"]
        components.append(
            ecam_total_uncertainty(
                n=stats["n"],
                p=stats["p"],
                reporting_count=len(conditions),
                rmse=stats["rmse"],
                baseline_temperature_mean=stats["baseline_temperature_mean"],
                baseline_temperature_variance=stats["baseline_temperature_variance"],
                reporting_temperature_mean=conditions.temperature.mean(),
                rho=stats["rho"],
            )["uncertainty_half_width"]
        )
    assert result.summary["uncertainty_half_width"] == pytest.approx(np.hypot(*components))


def test_adjustment_requires_exact_index_and_undefined_zero_savings():
    records = hourly()
    baseline = fit_baseline(records, model="2p")
    with pytest.raises(ValueError, match="exactly match"):
        avoided_energy(baseline, records, adjustment=pd.Series([1]))
    result = avoided_energy(baseline, records, uncertainty_method="ecam")
    assert result.summary["ecam_uncertainty"]["method"] == "ECAM v6 StdErrorCalcsAll"


def test_unsorted_and_invalid_baseline_json_rejected():
    records = hourly().iloc[::-1]
    with pytest.raises(ValueError, match="chronological"):
        fit_baseline(records, model="2p")
    with pytest.raises(ValueError):
        Baseline.from_dict({"schema_version": 100})
