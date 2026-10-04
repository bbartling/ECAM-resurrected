from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .models import FitResult
from .uncertainty import projected_total_uncertainty


@dataclass(frozen=True)
class SavingsResult:
    projected_baseline: float
    measured_or_projected_post: float
    savings: float
    savings_fraction: float
    uncertainty: float
    uncertainty_fraction_of_baseline: float
    confidence: float
    method: str


def avoided_energy_savings(
    baseline_model: FitResult,
    post_x,
    post_y,
    *,
    confidence: float = 0.80,
) -> SavingsResult:
    """ECAM-style avoided energy use: projected baseline minus measured post energy."""
    post_x = np.asarray(post_x, dtype=float).reshape(-1)
    post_y = np.asarray(post_y, dtype=float).reshape(-1)
    if post_x.shape != post_y.shape:
        raise ValueError("post_x and post_y must have the same length")
    mask = np.isfinite(post_x) & np.isfinite(post_y)
    post_x, post_y = post_x[mask], post_y[mask]
    if post_x.size == 0:
        raise ValueError("no finite post-period points")

    projected = float(np.sum(baseline_model.predict(post_x)))
    measured = float(np.sum(post_y))
    savings = projected - measured
    frac = savings / projected if projected != 0 else float("nan")
    uncert = projected_total_uncertainty(baseline_model, post_x, confidence)
    uncert_frac = uncert / projected if projected != 0 else float("nan")
    return SavingsResult(projected, measured, savings, frac, uncert, uncert_frac, confidence, "avoided_energy")


def normalized_savings(
    baseline_model: FitResult,
    post_model: FitResult,
    normalization_x,
    *,
    confidence: float = 0.80,
) -> SavingsResult:
    """ECAM-style normalized savings from baseline and post models on common weather."""
    x = np.asarray(normalization_x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("normalization_x has no finite values")
    base_total = float(np.sum(baseline_model.predict(x)))
    post_total = float(np.sum(post_model.predict(x)))
    savings = base_total - post_total
    frac = savings / base_total if base_total != 0 else float("nan")
    u_base = projected_total_uncertainty(baseline_model, x, confidence)
    u_post = projected_total_uncertainty(post_model, x, confidence)
    uncert = float(np.sqrt(u_base**2 + u_post**2))
    uncert_frac = uncert / base_total if base_total != 0 else float("nan")
    return SavingsResult(base_total, post_total, savings, frac, uncert, uncert_frac, confidence, "normalized")
