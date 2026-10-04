import numpy as np
import pytest
from scipy.stats import t

from ecam_mv.metrics import regression_metrics
from ecam_mv.validation import (
    baseline_precision_plan,
    ecam_fractional_savings_uncertainty,
    ecam_prediction_interval,
    ecam_regression_details,
    ecam_total_uncertainty,
    guideline14_check,
    precision_check,
    residual_flags,
)


def test_guideline14_screen_hourly_vs_monthly_and_absolute_bias():
    observed = [100.0] * 12
    predicted = [92.0] * 12
    metrics = regression_metrics(observed, predicted, 1)
    assert guideline14_check(metrics, "hourly").passed
    assert not guideline14_check(metrics, "monthly").passed
    overpredict = regression_metrics(observed, [108.0] * 12, 1)
    assert not guideline14_check(overpredict, "monthly").bias_pass


def test_monthly_legacy_fsu_matches_transcribed_equation():
    result = ecam_fractional_savings_uncertainty(
        n=12,
        p=2,
        reporting_count=12,
        cvrmse=0.1,
        savings_fraction=0.2,
        months=12,
        rho=0.8,
        frequency="billing",
        confidence=0.9,
    )
    # Explicit source coefficients; monthly rho is zero regardless of input.
    expected = t.ppf(0.95, 10) / 0.2 * 1.30558 * 0.1 * np.sqrt((1 + 2 / 12) / 12)
    assert result["fractional_savings_uncertainty"] == pytest.approx(expected)
    assert result["effective_n"] == 12


def test_hourly_legacy_absolute_rho_and_effective_degrees_of_freedom():
    arguments = dict(
        n=100,
        p=3,
        reporting_count=50,
        cvrmse=0.1,
        savings_fraction=0.2,
        months=1,
        confidence=0.9,
    )
    positive = ecam_fractional_savings_uncertainty(**arguments, rho=0.5)
    negative = ecam_fractional_savings_uncertainty(**arguments, rho=-0.5)
    assert positive == negative
    assert positive["effective_n"] == pytest.approx(100 / 3)


def test_legacy_total_se_includes_mean_leverage_and_prediction_noise():
    result = ecam_total_uncertainty(
        n=12,
        p=2,
        reporting_count=6,
        rmse=2,
        baseline_temperature_mean=10,
        baseline_temperature_variance=4,
        reporting_temperature_mean=12,
        frequency="billing",
        confidence=0.9,
    )
    mean_se = (365 / 12) * 2 * 6 / np.sqrt(12) * np.sqrt(2)
    noise_se = (365 / 12) * 2 * np.sqrt(6)
    expected = t.ppf(0.95, 10) * np.hypot(mean_se, noise_se)
    assert result["uncertainty_half_width"] == pytest.approx(expected)


def test_precision_default_and_zero_savings_are_explicit():
    assert precision_check(5, 100, confidence=0.9)["passed"]
    assert not precision_check(20, 100, confidence=0.9)["passed"]
    assert precision_check(0, 0)["relative_precision"] is None


def test_fixed_baseline_precision_floor_cannot_be_fixed_by_more_reporting_points():
    plan = baseline_precision_plan(
        n=12,
        p=2,
        rmse=5,
        baseline_mean_rate=100,
        expected_savings_fraction=0.01,
        frequency="billing",
        maximum_points=24,
    )
    assert plan["reporting_points_needed"] is None
    assert plan["model_uncertainty_floor"] > plan["required_precision"]


def test_undefined_gl14_ratios_fail_screen_and_outliers_are_flags_only():
    metrics = regression_metrics([0, 0, 0], [0, 1, 0], 1)
    assert not guideline14_check(metrics).passed
    values = np.array([0, 0, 0, 10])
    assert residual_flags(values, scale=1).tolist() == [False, False, False, True]
    assert values.tolist() == [0, 0, 0, 10]


@pytest.mark.parametrize("confidence", [0, 1, np.nan])
def test_invalid_precision_confidence(confidence):
    with pytest.raises(ValueError):
        precision_check(10, 100, confidence=confidence)


def test_ecam_steyx_prediction_interval_matches_simple_regression_formula():
    from ecam_mv.models import fit_model

    x = np.arange(10.0)
    y = 10 + x + np.array([1, -1, 0, 1, -1, 0, 1, -1, 0, 0])
    fit = fit_model(x, y, "2p")
    details = ecam_regression_details(fit, x, y, frequency="billing")
    lower, upper = ecam_prediction_interval(fit, [5], details, confidence=0.9)
    independent_beta = np.linalg.lstsq(np.column_stack([np.ones(10), x]), y, rcond=None)[0]
    se = np.sqrt(np.sum((y - independent_beta[0] - independent_beta[1] * x) ** 2) / 8)
    expected = (
        t.ppf(0.95, 8)
        * se
        * np.sqrt(1 + 1 / 10 + (5 - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    )
    assert (upper[0] - lower[0]) / 2 == pytest.approx(expected)
    assert details["segments"][0]["rho"] == 0
