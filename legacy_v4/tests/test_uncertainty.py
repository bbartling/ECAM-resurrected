import numpy as np
import pytest
from ecam_resurrected.models import fit_model
from ecam_resurrected.uncertainty import critical_t, fractional_savings_uncertainty, projected_total_uncertainty


def test_critical_t_95():
    assert critical_t(0.95, 1000) == pytest.approx(1.9623, rel=2e-3)


def test_fractional_savings_uncertainty_positive():
    u = fractional_savings_uncertainty(cvrmse=0.1, n_points=100, n_prime=80, post_points=50, fractional_savings=0.2, confidence=0.8, n_params=3)
    assert u > 0
    assert np.isfinite(u)


def test_projected_total_uncertainty_zero_noise_near_zero():
    x = np.linspace(30, 90, 80)
    y = 5 + 2*x
    r = fit_model(x, y, "2p")
    u = projected_total_uncertainty(r, x, 0.8)
    assert u < 1e-8
