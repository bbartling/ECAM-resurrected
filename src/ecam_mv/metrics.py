"""Regression diagnostics for energy models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    """Diagnostics; ``net_bias`` is normalized by total observed usage."""

    n: int
    p: int
    rmse: float | None
    cvrmse: float | None
    nmbe: float | None
    net_bias: float | None
    r_squared: float | None
    rho: float | None
    dof: int

    def to_dict(self) -> dict[str, int | float | None]:
        """Return JSON-compatible metric values."""
        return asdict(self)


def regression_metrics(
    observed: object, predicted: object, n_parameters: int = 2
) -> RegressionMetrics:
    """Calculate ECAM-style metrics; residuals are observed minus predicted.

    CVRMSE and NMBE are ratios rather than percentages. Lag-1 correlation uses
    the input order and is signed. Undefined ratios/correlations are ``None``.
    """
    y = _finite_vector(observed, "observed")
    yhat = _finite_vector(predicted, "predicted")
    if y.shape != yhat.shape:
        raise ValueError("observed and predicted must have the same length")
    if isinstance(n_parameters, bool) or not isinstance(n_parameters, (int, np.integer)):
        raise ValueError("n_parameters must be a non-negative integer")
    p = int(n_parameters)
    if p < 0:
        raise ValueError("n_parameters must be a non-negative integer")
    n = y.size
    dof = n - p
    if n == 0 or dof <= 0:
        raise ValueError("the number of observations must exceed n_parameters")

    residual = y - yhat
    mean_y = float(np.mean(y))
    sse = float(residual @ residual)
    rmse = float(np.sqrt(sse / dof))
    normalized_mean = mean_y if mean_y > 0 else None
    cvrmse = _ratio(rmse, normalized_mean)
    net_bias = float(np.sum(residual))
    nmbe = _ratio(net_bias, dof * normalized_mean) if normalized_mean is not None else None
    net_bias_ratio = _ratio(net_bias, n * normalized_mean) if normalized_mean is not None else None
    centered = y - mean_y
    total = float(centered @ centered)
    r_squared = None if total == 0 else float(1.0 - sse / total)
    rho = _lag_one_correlation(residual)
    return RegressionMetrics(n, p, rmse, cvrmse, nmbe, net_bias_ratio, r_squared, rho, dof)


def _finite_vector(values: object, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional numeric array") from exc
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional numeric array")
    return result


def _ratio(numerator: float, denominator: float | None) -> float | None:
    return None if denominator is None or denominator == 0 else float(numerator / denominator)


def _lag_one_correlation(residual: np.ndarray) -> float | None:
    if residual.size < 3:
        return None
    previous, current = residual[:-1], residual[1:]
    if np.std(previous) == 0 or np.std(current) == 0:
        return None
    return float(np.corrcoef(previous, current)[0, 1])
