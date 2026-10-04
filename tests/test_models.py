import json

import numpy as np
import pytest
from scipy.stats import t as student_t

from ecam_mv.models import FitResult, fit_model, select_model


def test_2p_exact_fit_and_prediction():
    x = np.arange(8.0)
    y = 4.0 - 1.5 * x
    fit = fit_model(x, y, "2p")
    assert fit.n_parameters == 2
    assert fit.coefficients == pytest.approx([4.0, -1.5])
    assert fit.predict([2, 5]) == pytest.approx([1.0, -3.5])
    assert fit.metrics.rmse == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("1p", 1),
        ("2p", 2),
        ("3pH", 3),
        ("3pC", 3),
        ("4p", 4),
        ("5p", 5),
        ("6p", 6),
        ("5pH", 5),
        ("5pC", 5),
        ("3pHzero", 2),
        ("5pHzero", 4),
    ],
)
def test_recovers_each_continuous_model_shape(model, expected):
    x = np.linspace(-12.0, 24.0, 121)
    if model == "1p":
        y = np.full_like(x, 15.0) + 0.04 * np.sin(x)
    elif model == "2p":
        y = 15.0 + 0.35 * x + 0.04 * np.sin(x)
    elif model == "3pH":
        y = 12.0 + 0.9 * np.maximum(4.0 - x, 0) + 0.04 * np.sin(x)
    elif model == "3pC":
        y = 12.0 + 0.9 * np.maximum(x - 4.0, 0) + 0.04 * np.sin(x)
    elif model == "3pHzero":
        y = 0.9 * np.maximum(4.0 - x, 0) + 0.04 * np.sin(x)
    elif model == "4p":
        dx = x - 4.0
        y = 12.0 - 0.25 * np.minimum(dx, 0) + 0.8 * np.maximum(dx, 0) + 0.04 * np.sin(x)
    elif model == "5p":
        y = 12.0 + 0.6 * np.maximum(3.0 - x, 0) + 0.7 * np.maximum(x - 15.0, 0) + 0.04 * np.sin(x)
    elif model == "6p":
        y = (
            12.0
            - 0.4 * np.minimum(x - 0.0, 0)
            + 0.2 * np.maximum(x - 0.0, 0)
            + 0.65 * np.maximum(x - 13.0, 0)
            + 0.04 * np.sin(x)
        )
    elif model == "5pC":
        y = 12.0 + 0.2 * np.maximum(x - 0.0, 0) + 0.45 * np.maximum(x - 13.0, 0) + 0.04 * np.sin(x)
    elif model == "5pH":
        y = (
            12.0
            - 0.4 * np.minimum(x - 0.0, 0)
            + 0.2 * (np.maximum(x, 0) - np.maximum(x - 13.0, 0))
            + 0.04 * np.sin(x)
        )
    else:  # 5pHzero: left and middle slopes, with the right segment fixed at zero.
        y = (
            -0.4 * np.minimum(x - 0.0, 0)
            + 0.2 * (np.minimum(x - 13.0, 0) - np.minimum(x, 0))
            + 0.04 * np.sin(x)
        )
    fit = fit_model(x, y, model)
    assert fit.n_parameters == expected
    assert fit.metrics.p == expected
    assert fit.metrics.dof == len(x) - expected
    assert np.sqrt(np.mean(fit.residuals**2)) < 0.15
    assert len(fit.breakpoints) == expected - len(fit.coefficients)


def test_bias_weights_impose_exact_energy_total():
    x = np.arange(12.0)
    usage = 8.0 + 0.4 * x + np.sin(x) * 0.7
    exposure = np.linspace(1.0, 2.0, x.size)
    fit = fit_model(x, usage, "2p", bias_weights=exposure)
    assert exposure @ fit.predict(x) == pytest.approx(exposure @ usage, abs=1e-10)


def test_bias_constraint_covariance_matches_ols_when_ols_already_satisfies_it():
    x = np.arange(12.0)
    y = (
        4.0
        + 0.6 * x
        + np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15, -0.05, 0.25, -0.2, 0.1, -0.15])
    )
    unconstrained = fit_model(x, y, "2p")
    constrained = fit_model(x, y, "2p", bias_weights=np.ones_like(x))
    assert constrained.coefficients == pytest.approx(unconstrained.coefficients)
    assert constrained.noise_variance == pytest.approx(unconstrained.noise_variance)
    assert constrained.covariance == pytest.approx(unconstrained.covariance, rel=1e-10, abs=1e-12)


def test_variable_exposure_aggregate_uncertainty_propagates_noise_and_shared_fit():
    x = np.arange(10.0)
    days = np.array([28, 31, 30, 31, 30, 31, 31, 28, 30, 31], dtype=float)
    y = 12 + 0.3 * x + np.array([0.2, -0.4, 0.1, -0.2, 0.5, -0.1, 0.3, -0.3, 0.2, -0.1])
    fit = fit_model(x, y, "2p", bias_weights=days)
    zsum = days @ fit.design_matrix(x)
    variance = zsum @ fit.covariance @ zsum + fit.noise_variance * (days @ days)
    expected = student_t.ppf(0.95, fit.metrics.dof) * np.sqrt(variance)
    assert fit.aggregate_uncertainty(x, days) == pytest.approx(expected)


def test_wls_changes_fit_to_emphasize_high_weight_record():
    x = np.arange(6.0)
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 30.0])
    plain = fit_model(x, y, "2p")
    emphasized = fit_model(x, y, "2p", weights=[1, 1, 1, 1, 1, 100])
    assert abs(emphasized.predict([5])[0] - y[-1]) < abs(plain.predict([5])[0] - y[-1])


def test_wls_weight_scale_does_not_change_noise_or_covariance():
    x = np.arange(8.0)
    y = np.array([1.2, 2.0, 1.8, 3.4, 3.1, 4.9, 4.5, 6.1])
    weights = np.array([1, 2, 1, 4, 2, 3, 1, 5], dtype=float)
    fit = fit_model(x, y, "2p", weights=weights)
    scaled = fit_model(x, y, "2p", weights=50 * weights)
    assert scaled.coefficients == pytest.approx(fit.coefficients)
    assert scaled.noise_variance == pytest.approx(fit.noise_variance)
    assert scaled.covariance == pytest.approx(fit.covariance)


def test_aggregate_uncertainty_keeps_shared_parameter_covariance():
    x = np.arange(10.0)
    y = 3.0 + 0.5 * x + np.array([0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2, -0.25, 0.15, -0.1])
    fit = fit_model(x, y, "2p")
    exposure = np.ones(10)
    aggregate = fit.aggregate_uncertainty(x, exposure, include_noise=False)
    zsum = exposure @ fit.design_matrix(x)
    expected = student_t.ppf(0.95, fit.metrics.dof) * np.sqrt(zsum @ fit.covariance @ zsum)
    assert aggregate == pytest.approx(expected)
    individual = (
        sum(fit.aggregate_uncertainty([value], [1], include_noise=False) ** 2 for value in x) ** 0.5
    )
    assert aggregate > individual


def test_serialization_roundtrip_and_validation():
    x = np.linspace(0, 20, 50)
    y = 5 + 0.7 * np.maximum(x - 8, 0) + np.sin(x) * 0.1
    fit = fit_model(x, y, "3pC")
    encoded = json.dumps(fit.to_dict(), allow_nan=False)
    restored = FitResult.from_dict(json.loads(encoded))
    assert restored.model == fit.model
    assert restored.coefficients == pytest.approx(fit.coefficients)
    assert restored.predict(x) == pytest.approx(fit.predict(x))
    assert restored.noise_variance == pytest.approx(fit.noise_variance)
    invalid = fit.to_dict()
    invalid["n_parameters"] = 999
    with pytest.raises(ValueError):
        FitResult.from_dict(invalid)


def test_constant_one_parameter_fit_roundtrips_with_constant_temperature():
    x = np.zeros(8)
    y = np.full(8, 10.0)
    fit = fit_model(x, y, "1p")
    assert fit.predict(x) == pytest.approx(y)
    restored = FitResult.from_dict(json.loads(json.dumps(fit.to_dict(), allow_nan=False)))
    assert restored.x_min == restored.x_max == 0
    assert restored.predict([0.0]) == pytest.approx([10.0])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["metrics"].update(rmse=float("nan")),
        lambda value: value["metrics"].update(dof=1),
        lambda value: value["covariance"].__setitem__(0, [-1e6, 0]),
        lambda value: value.update(selection={"score": float("inf")}),
    ],
)
def test_from_dict_rejects_invalid_metrics_covariance_and_selection(mutate):
    x = np.arange(8.0)
    fit = fit_model(x, np.array([1, 2, 1, 3, 2, 4, 3, 5]), "2p")
    document = fit.to_dict()
    mutate(document)
    with pytest.raises(ValueError):
        FitResult.from_dict(document)


def test_from_dict_rejects_unordered_breakpoints():
    x = np.linspace(0, 20, 80)
    y = 5 + 0.2 * np.maximum(x - 5, 0) + 0.4 * np.maximum(x - 14, 0)
    document = fit_model(x, y, "5pC").to_dict()
    document["breakpoints"] = list(reversed(document["breakpoints"]))
    with pytest.raises(ValueError):
        FitResult.from_dict(document)


def test_5phzero_right_segment_is_exactly_zero():
    x = np.linspace(-10.0, 25.0, 141)
    cp1, cp2 = 0.0, 13.0
    y = -0.4 * np.minimum(x - cp1, 0) + 0.2 * (np.minimum(x - cp2, 0) - np.minimum(x - cp1, 0))
    fit = fit_model(x, y, "5pHzero")
    warm = np.linspace(cp2, 25.0, 20)
    assert fit.predict(warm) == pytest.approx(np.zeros_like(warm), abs=1e-10)
    assert fit.predict(x) == pytest.approx(y, abs=1e-6)


def test_model_selection_and_rejection():
    x = np.linspace(0, 20, 60)
    y = 7 + 0.8 * np.maximum(x - 9, 0) + 0.03 * np.sin(x)
    fit = select_model(x, y, candidates=["2p", "3pC"])
    assert fit.model == "3pC"
    assert fit.selection["criterion"] == "AICc"
    assert set(fit.selection["scores"]) == {"2p", "3pC"}


@pytest.mark.parametrize(
    "x,y,model,kwargs",
    [
        ([1, 2, np.nan], [1, 2, 3], "2p", {}),
        ([1, 2], [1], "2p", {}),
        ([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], "3pC", {}),
        ([1, 2, 3], [2, 3, 4], "unknown", {}),
        ([1, 2, 3, 4], [2, 3, 4, 5], "2p", {"weights": [1, 0, 1, 1]}),
    ],
)
def test_rejects_invalid_or_undersampled_data(x, y, model, kwargs):
    with pytest.raises(ValueError):
        fit_model(x, y, model, **kwargs)


def test_prediction_uncertainty_validates_confidence():
    fit = fit_model(np.arange(8.0), np.array([1, 2, 1, 3, 2, 4, 3, 5]), "2p")
    with pytest.raises(ValueError):
        fit.prediction_interval([1.0], confidence=1.0)
    with pytest.raises(ValueError):
        fit.predict([])
    with pytest.raises(ValueError):
        fit.aggregate_uncertainty([1.0], [0.0])
