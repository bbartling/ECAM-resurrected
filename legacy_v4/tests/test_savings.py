import numpy as np
import pytest
from ecam_resurrected.models import fit_model
from ecam_resurrected.savings import avoided_energy_savings, normalized_savings


def test_avoided_energy_exact_savings():
    x = np.linspace(40, 80, 100)
    y_base = 100 + 2*x
    baseline = fit_model(x, y_base, "2p")
    y_post = 0.9 * y_base
    s = avoided_energy_savings(baseline, x, y_post)
    assert s.savings == pytest.approx(0.1 * np.sum(y_base), rel=1e-10)
    assert s.savings_fraction == pytest.approx(0.1, rel=1e-10)


def test_normalized_savings():
    x = np.linspace(35, 85, 100)
    b = fit_model(x, 100 + x, "2p")
    p = fit_model(x, 90 + x, "2p")
    s = normalized_savings(b, p, x)
    assert s.savings == pytest.approx(10 * len(x), abs=1e-8)
