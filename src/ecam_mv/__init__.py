"""Energy measurement and verification; an independent ECAM-inspired implementation."""

__version__ = "0.1.0a2"

from .analysis import Baseline, SavingsResult, avoided_energy, fit_baseline, normalized_savings
from .data import calendar_features, model_records, prepare_billing, prepare_hourly
from .datasets import practice_dataset
from .ingestion import MeterBatch, concat_meter_csvs, hourly_meter_records
from .metrics import RegressionMetrics, regression_metrics
from .models import FitResult, fit_model, select_model
from .validation import (
    baseline_precision_plan,
    ecam_fractional_savings_uncertainty,
    ecam_prediction_interval,
    ecam_regression_details,
    ecam_total_uncertainty,
    guideline14_check,
    precision_check,
)
from .weather import OpenMeteoClient, join_weather

__all__ = [
    "Baseline",
    "SavingsResult",
    "FitResult",
    "RegressionMetrics",
    "avoided_energy",
    "fit_baseline",
    "normalized_savings",
    "calendar_features",
    "prepare_billing",
    "prepare_hourly",
    "model_records",
    "practice_dataset",
    "MeterBatch",
    "concat_meter_csvs",
    "hourly_meter_records",
    "OpenMeteoClient",
    "join_weather",
    "regression_metrics",
    "fit_model",
    "select_model",
    "baseline_precision_plan",
    "ecam_fractional_savings_uncertainty",
    "ecam_total_uncertainty",
    "guideline14_check",
    "precision_check",
    "ecam_regression_details",
    "ecam_prediction_interval",
]
