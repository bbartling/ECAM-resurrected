import numpy as np
import pytest
from ecam_resurrected.metrics import compute_metrics, effective_sample_size, lag1_rho_ecam


def test_metrics_basic():
    y = np.array([1., 2., 3., 4., 5.])
    yhat = np.array([1.1, 1.9, 3.0, 4.1, 4.9])
    m = compute_metrics(y, yhat, 2)
    assert m.n == 5
    assert m.sse == pytest.approx(0.04)
    assert 0.99 < m.r2 <= 1.0


def test_effective_sample_size_decreases_with_rho():
    assert effective_sample_size(100, 0) == pytest.approx(100)
    assert effective_sample_size(100, 0.5) == pytest.approx(100 / 3)


def test_ecam_rho_is_magnitude():
    r = np.array([1., -1., 1., -1., 1., -1.])
    assert lag1_rho_ecam(r) == pytest.approx(1.0)
