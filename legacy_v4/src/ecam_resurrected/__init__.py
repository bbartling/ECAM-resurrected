"""ECAM Resurrected: NumPy/SciPy M&V change-point models inspired by ECAM v4."""

from .models import FitResult, fit_model, fit_best_model, predict
from .savings import SavingsResult, avoided_energy_savings, normalized_savings
from .uncertainty import fractional_savings_uncertainty, critical_t

__all__ = [
    "FitResult",
    "fit_model",
    "fit_best_model",
    "predict",
    "SavingsResult",
    "avoided_energy_savings",
    "normalized_savings",
    "fractional_savings_uncertainty",
    "critical_t",
]

__version__ = "0.1.0"
