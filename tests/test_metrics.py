from dataclasses import fields

import numpy as np
import pytest

from ecam_mv.metrics import regression_metrics


def test_analytical_regression_statistics_and_residual_sign():
    observed = np.array([2.0, 4.0, 6.0, 8.0])
    predicted = np.array([1.0, 3.0, 5.0, 7.0])
    result = regression_metrics(observed, predicted, n_parameters=2)

    assert result.n == 4
    assert result.p == 2
    assert result.dof == 2
    assert result.net_bias == pytest.approx(0.2)
    assert result.nmbe == pytest.approx(0.4)
    assert result.rmse == pytest.approx(np.sqrt(2.0))
    assert result.cvrmse == pytest.approx(np.sqrt(2.0) / 5.0)
    assert result.r_squared == pytest.approx(0.8)
    assert result.rho is None  # constant residual sequence
    assert result.to_dict()["net_bias"] == pytest.approx(0.2)
    assert {field.name for field in fields(result)} >= {"nmbe", "cvrmse", "rmse", "n", "p", "rho"}


def test_signed_lag_one_correlation():
    result = regression_metrics([1, 1, 1, 1, 1], [0, 1, 0, 1, 0], n_parameters=1)
    assert result.rho < 0


@pytest.mark.parametrize(
    "observed,predicted,p",
    [([1, 2], [1], 1), ([1, np.nan], [1, 2], 1), ([1, 2], [1, 2], 2)],
)
def test_rejects_invalid_metrics_inputs(observed, predicted, p):
    with pytest.raises(ValueError):
        regression_metrics(observed, predicted, p)


def test_constant_response_has_undefined_r_squared_and_mean_ratios():
    result = regression_metrics([0, 0, 0], [0, 1, 0], n_parameters=1)
    assert result.r_squared is None
    assert result.cvrmse is None
    assert result.nmbe is None


@pytest.mark.parametrize("observed", [[0, 0, 0], [-1, -2, -3]])
def test_nonpositive_mean_makes_normalized_metrics_undefined(observed):
    result = regression_metrics(observed, [0, 0, 0], n_parameters=1)
    assert result.net_bias is None
    assert result.nmbe is None
    assert result.cvrmse is None
