"""Fit reproducible baselines and calculate energy savings under matched conditions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import regression_metrics
from .models import FitResult, fit_model, select_model
from .validation import (
    ecam_fractional_savings_uncertainty,
    ecam_regression_details,
    ecam_total_uncertainty,
    guideline14_check,
    precision_check,
)


def _validate_records(records: pd.DataFrame) -> None:
    if records.empty:
        raise ValueError("Records are empty")
    if not isinstance(records.index, pd.DatetimeIndex):
        raise ValueError("Records require a DatetimeIndex")
    if (
        records.index.isna().any()
        or records.index.has_duplicates
        or not records.index.is_monotonic_increasing
    ):
        raise ValueError("Records require unique, valid, chronological timestamps")
    if not {"energy", "temperature", "duration"}.issubset(records.columns):
        raise ValueError("Records require energy, temperature and duration")
    try:
        numeric = records[["energy", "temperature", "duration"]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Energy, temperature and duration must be numeric") from exc
    if not np.isfinite(numeric).all() or (numeric[:, 2] <= 0).any():
        raise ValueError("Records contain missing/nonfinite values or nonpositive duration")
    if records.attrs.get("frequency") not in {"hourly", "billing"}:
        raise ValueError("Prepare records with prepare_hourly or prepare_billing first")


def _keys(records: pd.DataFrame, group_by: str | None) -> np.ndarray:
    if group_by is None:
        return np.repeat("all", len(records))
    if group_by not in records or records[group_by].isna().any():
        raise ValueError(f"Missing or null grouping column {group_by!r}")
    return records[group_by].astype(str).to_numpy()


@dataclass
class Baseline:
    """A saved baseline, including units, frequency and optional category models."""

    models: dict[str, FitResult]
    metadata: dict
    group_by: str | None = None

    def _check(self, records: pd.DataFrame) -> np.ndarray:
        _validate_records(records)
        for key in ("frequency", "duration_unit", "energy_unit", "temperature_unit", "timezone"):
            if records.attrs.get(key) != self.metadata.get(key):
                raise ValueError(f"Baseline/reporting {key} mismatch")
        if records.attrs.get("predictor", "temperature") != self.metadata.get(
            "predictor", "temperature"
        ):
            raise ValueError("Baseline/reporting predictor mismatch")
        if records.attrs.get("predictor_unit", records.attrs.get("temperature_unit")) != (
            self.metadata.get("predictor_unit", self.metadata.get("temperature_unit"))
        ):
            raise ValueError("Baseline/reporting predictor_unit mismatch")
        keys = _keys(records, self.group_by)
        unknown = sorted(set(keys) - set(self.models))
        if unknown:
            raise ValueError(f"Reporting categories absent from baseline: {unknown}")
        return keys

    def predict(self, records: pd.DataFrame) -> pd.Series:
        """Predict record energy using each record's exposure and supplied weather."""
        keys = self._check(records)
        predicted = np.empty(len(records))
        for key in np.unique(keys):
            mask = keys == key
            subset = records.iloc[np.flatnonzero(mask)]
            predicted[mask] = self.models[key].predict(subset.temperature.to_numpy()) * (
                subset.duration.to_numpy()
            )
        return pd.Series(predicted, index=records.index, name="adjusted_baseline")

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "group_by": self.group_by,
            "metadata": self.metadata,
            "models": {key: model.to_dict() for key, model in self.models.items()},
        }

    @classmethod
    def from_dict(cls, document: dict) -> Baseline:
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ValueError("Unsupported baseline schema_version")
        metadata = document.get("metadata")
        models = document.get("models")
        group_by = document.get("group_by")
        if not isinstance(metadata, dict) or metadata.get("frequency") not in {"hourly", "billing"}:
            raise ValueError("Invalid baseline metadata")
        required = {"frequency", "duration_unit", "energy_unit", "temperature_unit", "timezone"}
        if not required.issubset(metadata):
            raise ValueError("Incomplete baseline metadata")
        if group_by is not None and not isinstance(group_by, str):
            raise ValueError("Invalid grouping column")
        if (
            not isinstance(models, dict)
            or not models
            or not all(isinstance(k, str) for k in models)
        ):
            raise ValueError("Invalid baseline models")
        if group_by is None and set(models) != {"all"}:
            raise ValueError("An ungrouped baseline requires a single 'all' model")
        return cls(
            {key: FitResult.from_dict(value) for key, value in models.items()}, metadata, group_by
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid baseline JSON") from exc
        return cls.from_dict(document)


def fit_baseline(
    records: pd.DataFrame,
    *,
    model: str = "3pC",
    group_by: str | None = None,
    min_segment: int = 3,
) -> Baseline:
    """Fit energy per hour/day; monthly fits enforce zero total-bill bias."""
    _validate_records(records)
    if records.attrs["frequency"] == "billing" and group_by is not None:
        raise ValueError("Billing baselines use one model; hourly records support category models")
    keys = _keys(records, group_by)
    models = {}
    for key in np.unique(keys):
        subset = records.iloc[np.flatnonzero(keys == key)]
        exposure = subset.duration.to_numpy(dtype=float)
        kwargs = {
            "min_segment": min_segment,
            "bias_weights": exposure if records.attrs["frequency"] == "billing" else None,
        }
        try:
            if model == "auto":
                models[key] = select_model(
                    subset.temperature.to_numpy(), subset.energy.to_numpy() / exposure, **kwargs
                )
            else:
                models[key] = fit_model(
                    subset.temperature.to_numpy(),
                    subset.energy.to_numpy() / exposure,
                    model=model,
                    **kwargs,
                )
        except ValueError as exc:
            raise ValueError(f"Cannot fit baseline category {key!r}: {exc}") from exc
    metadata = records.attrs.copy()
    metadata.update(
        {
            "training_start": records.index[0].isoformat(),
            "training_end": records.index[-1].isoformat(),
            "training_records": len(records),
            "billing_bias_constraint": records.attrs["frequency"] == "billing",
        }
    )
    baseline = Baseline(models, metadata, group_by)
    rate = records.energy.to_numpy() / records.duration.to_numpy()
    predicted_rate = baseline.predict(records).to_numpy() / records.duration.to_numpy()
    metrics = regression_metrics(rate, predicted_rate, sum(m.n_parameters for m in models.values()))
    metadata["ecam_statistics"] = {
        "n": len(records),
        "p": sum(m.n_parameters for m in models.values()),
        "rmse": metrics.rmse,
        "cvrmse": metrics.cvrmse,
        "rho": metrics.rho if metrics.rho is not None else 0,
        "baseline_temperature_mean": float(records.temperature.mean()),
        "baseline_temperature_variance": float(np.var(records.temperature.to_numpy())),
        "baseline_mean_rate": float(np.mean(rate)),
    }
    metadata["fit_validation"] = guideline14_check(metrics, metadata["frequency"]).to_dict()
    metadata["fit_metrics"] = metrics.to_dict()
    # ECAM bills report bias on billed totals; rate-model metrics use energy/day.
    measured_total = float(records.energy.sum())
    energy_bias = float((records.energy - baseline.predict(records)).sum())
    metadata["ecam_net_determination_bias"] = (
        energy_bias / measured_total if measured_total > 0 else None
    )
    metadata["ecam_segment_statistics"] = {
        key: ecam_regression_details(
            fit,
            records.temperature.to_numpy()[keys == key],
            rate[keys == key],
            frequency=metadata["frequency"],
        )
        for key, fit in models.items()
    }
    return baseline


@dataclass
class SavingsResult:
    records: pd.DataFrame
    summary: dict

    def to_dict(self) -> dict:
        return {"schema_version": 1, "summary": self.summary}

    def save(self, json_path: str | Path, csv_path: str | Path | None = None) -> None:
        Path(json_path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
        )
        if csv_path is not None:
            self.records.to_csv(csv_path, index_label="timestamp")


def _half_width(
    baseline: Baseline,
    records: pd.DataFrame,
    confidence: float,
    autocorrelation: bool,
    include_noise: bool,
) -> float:
    keys = baseline._check(records)
    variances = []
    for key in np.unique(keys):
        subset = records.iloc[np.flatnonzero(keys == key)]
        width = baseline.models[key].aggregate_uncertainty(
            subset.temperature.to_numpy(),
            exposure=subset.duration.to_numpy(),
            confidence=confidence,
            autocorrelation=autocorrelation,
            include_noise=include_noise,
        )
        variances.append(width**2)
    return float(np.sqrt(sum(variances)))


def avoided_energy(
    baseline: Baseline,
    reporting: pd.DataFrame,
    *,
    confidence: float = 0.9,
    adjustment: float | pd.Series = 0.0,
    autocorrelation: bool = False,
    uncertainty_method: str = "conditional",
) -> SavingsResult:
    """Reporting-condition avoided energy, with explicit nonroutine adjustment.

    A scalar adjustment is energy per record, not a total. A Series must match
    reporting timestamps exactly. Positive adjustment raises the counterfactual.
    Uncertainty excludes metering, adjustments and change-point estimation error.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if uncertainty_method not in {"conditional", "ecam"}:
        raise ValueError("uncertainty_method must be conditional or ecam")
    adjusted = baseline.predict(reporting)
    if isinstance(adjustment, pd.Series):
        if not adjustment.index.equals(reporting.index):
            raise ValueError("Adjustment timestamps must exactly match reporting records")
        try:
            adjustments = adjustment.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Adjustments must be numeric") from exc
    else:
        try:
            value = float(adjustment)
        except (TypeError, ValueError) as exc:
            raise ValueError("Adjustment must be a number or aligned Series") from exc
        adjustments = np.repeat(value, len(reporting))
    if not np.isfinite(adjustments).all():
        raise ValueError("Adjustment contains nonfinite values")
    result = reporting.copy()
    result["adjusted_baseline"] = adjusted
    result["nonroutine_adjustment"] = adjustments
    result["savings"] = adjusted.to_numpy() + adjustments - reporting.energy.to_numpy()
    result["cumulative_savings"] = result.savings.cumsum()
    keys = _keys(reporting, baseline.group_by)
    extrapolated = np.zeros(len(reporting), dtype=bool)
    for key, fit in baseline.models.items():
        mask = keys == key
        temps = reporting.temperature.to_numpy()[mask]
        extrapolated[mask] = (temps < fit.x_min) | (temps > fit.x_max)
    result["extrapolated"] = extrapolated
    total_baseline = float(adjusted.sum() + adjustments.sum())
    savings = float(result.savings.sum())
    half_width = _half_width(baseline, reporting, confidence, autocorrelation, True)
    ecam_stats = baseline.metadata.get("ecam_statistics")
    legacy_uncertainty = None
    legacy_fsu = None
    if ecam_stats is not None:
        legacy_uncertainty = ecam_total_uncertainty(
            n=ecam_stats["n"],
            p=ecam_stats["p"],
            reporting_count=len(reporting),
            rmse=ecam_stats["rmse"],
            baseline_temperature_mean=ecam_stats["baseline_temperature_mean"],
            baseline_temperature_variance=ecam_stats["baseline_temperature_variance"],
            reporting_temperature_mean=float(reporting.temperature.mean()),
            rho=ecam_stats["rho"],
            confidence=confidence,
            frequency=baseline.metadata["frequency"],
        )
        fraction = savings / total_baseline if total_baseline != 0 else 0
        months = (
            len(reporting)
            if baseline.metadata["frequency"] == "billing"
            else float(reporting.duration.sum()) / (365 / 12 * 24)
        )
        if ecam_stats["cvrmse"] is not None and ecam_stats["cvrmse"] >= 0:
            legacy_fsu = ecam_fractional_savings_uncertainty(
                n=ecam_stats["n"],
                p=ecam_stats["p"],
                reporting_count=len(reporting),
                cvrmse=ecam_stats["cvrmse"],
                savings_fraction=fraction,
                months=months,
                rho=ecam_stats["rho"],
                confidence=confidence,
                frequency=baseline.metadata["frequency"],
            )
    if uncertainty_method == "ecam":
        if legacy_uncertainty is None:
            raise ValueError("Saved model lacks ECAM baseline statistics; refit the baseline")
        half_width = legacy_uncertainty["uncertainty_half_width"]
    summary = {
        "frequency": baseline.metadata["frequency"],
        "energy_unit": baseline.metadata["energy_unit"],
        "reporting_records": len(reporting),
        "reporting_start": reporting.index[0].isoformat(),
        "reporting_end": reporting.index[-1].isoformat(),
        "adjusted_baseline": total_baseline,
        "nonroutine_adjustment": float(adjustments.sum()),
        "actual_energy": float(reporting.energy.sum()),
        "avoided_energy": savings,
        "fractional_savings": savings / total_baseline if total_baseline != 0 else None,
        "confidence": confidence,
        "uncertainty_half_width": half_width,
        "savings_lower": savings - half_width,
        "savings_upper": savings + half_width,
        "fractional_savings_uncertainty": half_width / abs(savings) if savings != 0 else None,
        "autocorrelation_adjusted": autocorrelation
        if uncertainty_method == "conditional"
        else baseline.metadata["frequency"] == "hourly",
        "uncertainty_method": uncertainty_method,
        "uncertainty_description": (
            "conditional regression covariance plus prediction noise"
            if uncertainty_method == "conditional"
            else "ECAM v6 StdErrorCalcsAll"
        ),
        "ecam_uncertainty": legacy_uncertainty,
        "ecam_ashrae_fsu": legacy_fsu,
        "precision": precision_check(half_width, savings, confidence=confidence),
        "extrapolated_records": int(extrapolated.sum()),
    }
    return SavingsResult(result, summary)


def normalized_savings(
    baseline: Baseline,
    post: Baseline,
    conditions: pd.DataFrame,
    *,
    confidence: float = 0.9,
    autocorrelation: bool = False,
    uncertainty_method: str = "conditional",
) -> SavingsResult:
    """Compare baseline and post models on identical supplied weather/schedules.

    Conditions are prepared records with duration, temperature and calendar
    groups. Their energy column is a placeholder and does not enter savings.
    Conditional intervals concern model means, excluding future noise; the
    explicitly selected ECAM method preserves its normalized SEpTotal formula.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if uncertainty_method not in {"conditional", "ecam"}:
        raise ValueError("uncertainty_method must be conditional or ecam")
    before, after = baseline.predict(conditions), post.predict(conditions)
    result = conditions.copy()
    result["normalized_baseline"] = before
    result["normalized_post"] = after
    result["savings"] = before - after
    result["cumulative_savings"] = result.savings.cumsum()
    half_width = np.hypot(
        _half_width(baseline, conditions, confidence, autocorrelation, False),
        _half_width(post, conditions, confidence, autocorrelation, False),
    )
    legacy_widths = []
    for fitted in (baseline, post):
        stats = fitted.metadata.get("ecam_statistics")
        if stats is not None:
            legacy_widths.append(
                ecam_total_uncertainty(
                    n=stats["n"],
                    p=stats["p"],
                    reporting_count=len(conditions),
                    rmse=stats["rmse"],
                    baseline_temperature_mean=stats["baseline_temperature_mean"],
                    baseline_temperature_variance=stats["baseline_temperature_variance"],
                    reporting_temperature_mean=float(conditions.temperature.mean()),
                    rho=stats["rho"],
                    confidence=confidence,
                    frequency=fitted.metadata["frequency"],
                )["uncertainty_half_width"]
            )
    ecam_width = float(np.hypot(*legacy_widths)) if len(legacy_widths) == 2 else None
    if uncertainty_method == "ecam":
        if ecam_width is None:
            raise ValueError("Saved models lack ECAM baseline statistics; refit both models")
        half_width = ecam_width
    savings, total = float(result.savings.sum()), float(before.sum())
    summary = {
        "frequency": baseline.metadata["frequency"],
        "energy_unit": baseline.metadata["energy_unit"],
        "conditions_records": len(conditions),
        "normalized_baseline": total,
        "normalized_post": float(after.sum()),
        "normalized_savings": savings,
        "fractional_savings": savings / total if total != 0 else None,
        "confidence": confidence,
        "uncertainty_half_width": float(half_width),
        "savings_lower": float(savings - half_width),
        "savings_upper": float(savings + half_width),
        "uncertainty_method": uncertainty_method,
        "uncertainty_description": (
            "conditional model means; independent baseline/post fits"
            if uncertainty_method == "conditional"
            else "ECAM v6 normalized baseline/post quadrature"
        ),
        "ecam_normalized_uncertainty_half_width": ecam_width,
        "autocorrelation_adjusted": (
            autocorrelation
            if uncertainty_method == "conditional"
            else baseline.metadata["frequency"] == "hourly"
        ),
        "precision": precision_check(float(half_width), savings, confidence=confidence),
    }
    return SavingsResult(result, summary)
