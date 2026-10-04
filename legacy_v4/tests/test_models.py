import numpy as np
import pytest

from ecam_resurrected.models import fit_model, fit_best_model


def test_2p_exact():
    x = np.linspace(20, 90, 100)
    y = 12.0 + 1.7 * x
    r = fit_model(x, y, "2p")
    assert r.metrics.r2 == pytest.approx(1.0, abs=1e-12)
    assert r.params["intercept"] == pytest.approx(12.0, abs=1e-9)
    assert r.params["slope"] == pytest.approx(1.7, abs=1e-9)


def test_3p_heating_recovers_change_point():
    x = np.linspace(30, 90, 241)
    cp = 62.0
    y = 100.0 - 2.5 * np.minimum(x - cp, 0.0)
    r = fit_model(x, y, "3p_heat", grid_size=41)
    assert r.change_points[0] == pytest.approx(cp, abs=0.05)
    assert r.params["base"] == pytest.approx(100.0, abs=0.1)
    assert r.params["heating_slope"] == pytest.approx(-2.5, abs=0.01)
    assert r.metrics.r2 > 0.999999


def test_3p_cooling_recovers_change_point():
    x = np.linspace(30, 100, 281)
    cp = 68.0
    y = 75.0 + 3.2 * np.maximum(x - cp, 0.0)
    r = fit_model(x, y, "3p_cool")
    assert r.change_points[0] == pytest.approx(cp, abs=0.08)
    assert r.params["cooling_slope"] == pytest.approx(3.2, abs=0.02)


def test_4p_recovers_piecewise_slopes():
    x = np.linspace(20, 100, 321)
    cp = 61.0
    y = 120 + (-1.8) * np.minimum(x - cp, 0) + 2.2 * np.maximum(x - cp, 0)
    r = fit_model(x, y, "4p")
    assert r.change_points[0] == pytest.approx(cp, abs=0.1)
    assert r.params["low_slope"] == pytest.approx(-1.8, abs=0.02)
    assert r.params["high_slope"] == pytest.approx(2.2, abs=0.02)


def test_5p_recovers_two_change_points():
    x = np.linspace(20, 100, 321)
    c1, c2 = 55.0, 72.0
    y = 90 + (-2.0) * np.minimum(x - c1, 0) + 2.8 * np.maximum(x - c2, 0)
    r = fit_model(x, y, "5p", grid_size=35)
    assert r.change_points[0] == pytest.approx(c1, abs=0.2)
    assert r.change_points[1] == pytest.approx(c2, abs=0.2)
    assert r.params["low_slope"] == pytest.approx(-2.0, abs=0.03)
    assert r.params["high_slope"] == pytest.approx(2.8, abs=0.03)


def test_6p_recovers_three_slopes():
    x = np.linspace(20, 100, 321)
    c1, c2 = 50.0, 74.0
    low = np.minimum(x - c1, 0)
    mid = np.clip(x - c1, 0, c2 - c1)
    high = np.maximum(x - c2, 0)
    y = 80 + (-1.5) * low + 0.35 * mid + 2.1 * high
    r = fit_model(x, y, "6p", grid_size=35)
    assert r.change_points[0] == pytest.approx(c1, abs=0.3)
    assert r.change_points[1] == pytest.approx(c2, abs=0.3)
    assert r.params["low_slope"] == pytest.approx(-1.5, abs=0.05)
    assert r.params["middle_slope"] == pytest.approx(0.35, abs=0.05)
    assert r.params["high_slope"] == pytest.approx(2.1, abs=0.05)


def test_auto_selector_prefers_linear_for_linear_data():
    x = np.linspace(30, 90, 100)
    y = 10 + 2 * x
    r = fit_best_model(x, y, models=["2p", "3p_heat", "3p_cool", "4p"])
    assert r.model_type == "2p"
