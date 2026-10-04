from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from .models import FitResult, design_matrix_for_result


def critical_t(confidence: float, dof: float) -> float:
    """Two-sided Student-t critical value, equivalent to ECAM's TINV(1-ConfLvl, df)."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    dof = max(2.0, float(dof))
    return float(student_t.ppf((1.0 + confidence) / 2.0, dof))


def fractional_savings_uncertainty(
    *,
    cvrmse: float,
    n_points: int,
    n_prime: float,
    post_points: int,
    fractional_savings: float,
    confidence: float = 0.80,
    n_params: int = 2,
) -> float:
    """ECAM v4 fractional-savings uncertainty approximation.

    FSU = t/Fsave * 1.26 * CVRMSE * sqrt(n/n' * (1 + 2/n') / m)
    """
    if post_points <= 0:
        raise ValueError("post_points must be positive")
    if n_points <= 0 or n_prime <= 0:
        raise ValueError("n_points and n_prime must be positive")
    if fractional_savings == 0:
        return float("inf")
    tcrit = critical_t(confidence, max(2.0, n_prime - n_params))
    term = n_points / n_prime * (1.0 + 2.0 / n_prime) / post_points
    return float(tcrit / abs(fractional_savings) * 1.26 * abs(cvrmse) * np.sqrt(term))


def prediction_half_interval(result: FitResult, x_new, confidence: float = 0.80) -> np.ndarray:
    """Conditional prediction half-interval with fitted change points held fixed."""
    Xn = design_matrix_for_result(result, x_new)
    dof = max(2.0, result.metrics.n_prime - result.metrics.n_params)
    tcrit = critical_t(confidence, dof)
    mse = result.metrics.sse / (result.metrics.n - result.metrics.n_params)
    mean_var = np.einsum("ij,jk,ik->i", Xn, result.covariance, Xn)
    pred_var = np.maximum(0.0, mean_var + mse)
    return tcrit * np.sqrt(pred_var)


def projected_total_standard_error(result: FitResult, x_projection) -> float:
    """Standard error of a projected total, conditional on fitted change points.

    This directly evaluates the variance of the linear combination of regression
    coefficients. ECAM v4 describes its SEpm quantity in the same terms, although
    its spreadsheet calculates the piecewise cases with separate worksheet formulas.
    """
    Xn = design_matrix_for_result(result, x_projection)
    summed_design = np.sum(Xn, axis=0)
    variance = float(summed_design @ result.covariance @ summed_design.T)
    return float(np.sqrt(max(0.0, variance)))


def projected_total_uncertainty(result: FitResult, x_projection, confidence: float = 0.80) -> float:
    dof = max(2.0, result.metrics.n_prime - result.metrics.n_params)
    return critical_t(confidence, dof) * projected_total_standard_error(result, x_projection)
