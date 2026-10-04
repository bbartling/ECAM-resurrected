from __future__ import annotations

from .models import FitResult
from .savings import SavingsResult


def fit_text(result: FitResult) -> str:
    m = result.metrics
    lines = [
        f"Model: {result.model_type}",
        "Parameters:",
    ]
    for k, v in result.params.items():
        lines.append(f"  {k}: {v:.6g}")
    lines.extend([
        "",
        "Regression metrics:",
        f"  Points: {m.n}",
        f"  RMSE: {m.rmse:.6g}",
        f"  R²: {m.r2:.6f}",
        f"  CV(RMSE): {m.cvrmse:.3%}",
        f"  Net determination bias: {m.ndb:.3%}",
        f"  ECAM lag-1 rho magnitude: {m.rho:.4f}",
        f"  Effective points n': {m.n_prime:.2f}",
        f"  AICc (modern selector): {m.aicc:.3f}",
        f"  Segment counts: {result.segment_counts}",
    ])
    return "\n".join(lines)


def savings_text(result: SavingsResult) -> str:
    return "\n".join([
        f"Method: {result.method}",
        f"Projected baseline: {result.projected_baseline:,.3f}",
        f"Measured/projected post: {result.measured_or_projected_post:,.3f}",
        f"Savings: {result.savings:,.3f}",
        f"Savings fraction: {result.savings_fraction:.2%}",
        f"Regression uncertainty @ {result.confidence:.0%}: ±{result.uncertainty:,.3f}",
        f"Uncertainty / baseline: {result.uncertainty_fraction_of_baseline:.2%}",
        "Note: as ECAM warned, this uncertainty only represents regression/model uncertainty.",
    ])
