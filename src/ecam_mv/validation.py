"""Guideline 14 fit screens and ECAM v6 uncertainty/precision formula ports.

ECAM formula references: FormulasAllData2 (18421), FSU (19515),
StdErrorCalcsAll (19694), BaselinePrecisionCheck (27282). Python translation
of methods attributed to Copyright 2018 William E. Koran, P.E., Apache-2.0.
Changed: explicit inputs, finite/undefined handling and bounded planning.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import t

from .metrics import RegressionMetrics


@dataclass(frozen=True)
class FitValidation:
    frequency: str
    nmbe: float | None
    cvrmse: float | None
    nmbe_limit: float
    cvrmse_limit: float
    bias_pass: bool
    error_pass: bool
    passed: bool
    interpretation: str = "Guideline 14-2014 calibration screen; not M&V certification"

    def to_dict(self) -> dict:
        return asdict(self)


def guideline14_check(metrics: RegressionMetrics, frequency: str = "hourly") -> FitValidation:
    """Check |NMBE| <= 10%, CVRMSE <= 30% hourly; 5%,15% monthly.

    Ratios are fractions, not percentage values. This names the 2014 screen,
    not an assertion about every requirement of a later Guideline 14 edition.
    Nonpositive mean consumption makes normalized screens inapplicable.
    """
    if frequency == "hourly":
        bias_limit, error_limit = 0.10, 0.30
    elif frequency in {"billing", "monthly"}:
        bias_limit, error_limit = 0.05, 0.15
    else:
        raise ValueError("frequency must be hourly, billing or monthly")
    bias = metrics.nmbe is not None and math.isfinite(metrics.nmbe)
    error = metrics.cvrmse is not None and math.isfinite(metrics.cvrmse)
    bias_pass = bool(bias and abs(metrics.nmbe) <= bias_limit)
    error_pass = bool(error and 0 <= metrics.cvrmse <= error_limit)
    return FitValidation(
        frequency,
        metrics.nmbe,
        metrics.cvrmse,
        bias_limit,
        error_limit,
        bias_pass,
        error_pass,
        bias_pass and error_pass,
    )


def _inputs(n: int, p: int, confidence: float, rho: float, frequency: str) -> tuple[float, float]:
    if isinstance(n, bool) or isinstance(p, bool) or n <= p or p < 1:
        raise ValueError("Require n > p >= 1")
    if not all(math.isfinite(v) for v in (n, p, confidence, rho)):
        raise ValueError("Uncertainty inputs must be finite")
    if not 0 < confidence < 1 or not -1 <= rho <= 1:
        raise ValueError("Require 0 < confidence < 1 and -1 <= rho <= 1")
    if frequency not in {"hourly", "billing", "monthly"}:
        raise ValueError("frequency must be hourly, billing or monthly")
    # ECAM deliberately uses abs(rho), from sqrt(RSQ), and zero for bills.
    r = 0.0 if frequency in {"billing", "monthly"} else abs(rho)
    effective_n = max(1.0, n * (1 - r) / (1 + r))
    critical_t = float(t.ppf((1 + confidence) / 2, max(2.0, effective_n - p)))
    return effective_n, critical_t


def ecam_fractional_savings_uncertainty(
    *,
    n: int,
    p: int,
    reporting_count: int,
    cvrmse: float,
    savings_fraction: float,
    months: float,
    rho: float = 0.0,
    confidence: float = 0.9,
    frequency: str = "hourly",
) -> dict:
    """Port ECAM's ASHRAE FSU expression and empirical month correction.

    ``months`` is the modeled period length in equivalent calendar months;
    ECAM uses bill count for monthly data. Zero savings has undefined relative
    uncertainty and returns None. The polynomial is not extrapolated past a
    nonpositive correction factor. This function reports the legacy method.
    """
    effective_n, critical_t = _inputs(n, p, confidence, rho, frequency)
    if reporting_count <= 0 or not math.isfinite(reporting_count):
        raise ValueError("reporting_count must be positive")
    if not all(math.isfinite(v) for v in (cvrmse, savings_fraction, months)):
        raise ValueError("FSU values must be finite")
    if cvrmse < 0 or months <= 0:
        raise ValueError("Require cvrmse >= 0 and months > 0")
    a, b, c = (
        (-0.00022, 0.03306, 0.94054)
        if frequency in {"monthly", "billing"}
        else (-0.00024, 0.03535, 1.00286)
    )
    factor = a * months**2 + b * months + c
    if factor <= 0:
        raise ValueError("ECAM month polynomial is nonpositive outside its usable range")
    absolute_fraction_width = (
        critical_t
        * factor
        * cvrmse
        * math.sqrt(n / effective_n * (1 + 2 / effective_n) / reporting_count)
    )
    return {
        "method": "ECAM v6 ASHRAE FSU with empirical month correction",
        "confidence": confidence,
        "effective_n": effective_n,
        "critical_t": critical_t,
        "month_correction": factor,
        "fractional_savings_uncertainty": (
            absolute_fraction_width / abs(savings_fraction) if savings_fraction != 0 else None
        ),
        "savings_fraction_half_width": absolute_fraction_width,
    }


def ecam_total_uncertainty(
    *,
    n: int,
    p: int,
    reporting_count: int,
    rmse: float,
    baseline_temperature_mean: float,
    baseline_temperature_variance: float,
    reporting_temperature_mean: float,
    rho: float = 0.0,
    confidence: float = 0.9,
    frequency: str = "hourly",
    include_noise: bool = True,
) -> dict:
    """Port StdErrorCalcsAll model-estimate and prediction-noise quadrature.

    Monthly source uses 365/12 days per bill; hourly source uses 1 hour.
    A constant-temperature 1p model has no temperature-leverage term.
    """
    effective_n, critical_t = _inputs(n, p, confidence, rho, frequency)
    vals = (
        rmse,
        baseline_temperature_mean,
        baseline_temperature_variance,
        reporting_temperature_mean,
        reporting_count,
    )
    if not all(math.isfinite(v) for v in vals) or reporting_count <= 0:
        raise ValueError("Uncertainty values must be finite and count positive")
    if rmse < 0 or baseline_temperature_variance < 0:
        raise ValueError("RMSE and variance must be nonnegative")
    mean_difference = baseline_temperature_mean - reporting_temperature_mean
    if baseline_temperature_variance == 0:
        if p > 1 and mean_difference != 0:
            raise ValueError("Temperature extrapolation with zero baseline variance")
        leverage = 1.0
    else:
        leverage = 1 + mean_difference**2 / baseline_temperature_variance
    duration = 365 / 12 if frequency in {"billing", "monthly"} else 1.0
    estimate_se = duration * rmse * reporting_count / math.sqrt(n) * math.sqrt(leverage)
    noise_se = (
        duration * rmse * math.sqrt(reporting_count * n / effective_n) if include_noise else 0
    )
    half_width = critical_t * math.hypot(estimate_se, noise_se)
    return {
        "method": "ECAM v6 StdErrorCalcsAll",
        "confidence": confidence,
        "effective_n": effective_n,
        "critical_t": critical_t,
        "model_estimate_standard_error": estimate_se,
        "prediction_noise_standard_error": noise_se,
        "uncertainty_half_width": half_width,
        "billing_days_per_record": duration if frequency in {"billing", "monthly"} else None,
    }


def precision_check(
    uncertainty_half_width: float,
    savings: float,
    *,
    confidence: float = 0.9,
    required_precision: float | None = None,
) -> dict:
    """ECAM relative precision = half-width / |savings|; default limit 1-confidence."""
    if not all(math.isfinite(v) for v in (uncertainty_half_width, savings, confidence)):
        raise ValueError("Precision inputs must be finite")
    if uncertainty_half_width < 0 or not 0 < confidence < 1:
        raise ValueError("Require nonnegative half-width and 0 < confidence < 1")
    target = 1 - confidence if required_precision is None else required_precision
    if not math.isfinite(target) or target <= 0:
        raise ValueError("required_precision must be positive and finite")
    relative = uncertainty_half_width / abs(savings) if savings != 0 else None
    return {
        "confidence": confidence,
        "relative_precision": relative,
        "required_precision": target,
        "passed": relative is not None and relative <= target,
        "requirement_source": "ECAM default 1-confidence" if required_precision is None else "user",
    }


def baseline_precision_plan(
    *,
    n: int,
    p: int,
    rmse: float,
    baseline_mean_rate: float,
    expected_savings_fraction: float = 0.05,
    reporting_count: int | None = None,
    rho: float = 0.0,
    confidence: float = 0.9,
    frequency: str = "hourly",
    required_precision: float | None = None,
    maximum_points: int | None = None,
) -> dict:
    """Bounded ECAM-style planning at baseline-average weather/consumption.

    Vary prospective reporting count, keeping the fitted baseline fixed. This
    does not promise that collecting more data will improve the baseline fit.
    Unlike the legacy Solver, never mutate sample counts in an existing model.
    """
    effective_n, critical_t = _inputs(n, p, confidence, rho, frequency)
    if not all(math.isfinite(v) for v in (rmse, baseline_mean_rate, expected_savings_fraction)):
        raise ValueError("Planning inputs must be finite")
    if rmse < 0 or baseline_mean_rate <= 0 or expected_savings_fraction == 0:
        raise ValueError("Planning requires nonnegative RMSE, positive rate and nonzero savings")
    count = n if reporting_count is None else reporting_count
    maximum = 24 if frequency in {"monthly", "billing"} else 17520
    maximum = maximum if maximum_points is None else maximum_points
    if count <= 0 or maximum <= 0:
        raise ValueError("Planning counts must be positive")
    target = 1 - confidence if required_precision is None else required_precision
    if not math.isfinite(target) or target <= 0:
        raise ValueError("required_precision must be positive")
    scale = critical_t * rmse / abs(expected_savings_fraction * baseline_mean_rate)

    def relative(m: int) -> float:
        return scale * math.sqrt(1 / n + n / (effective_n * m))

    needed = next((m for m in range(1, maximum + 1) if relative(m) <= target), None)
    return {
        "confidence": confidence,
        "expected_savings_fraction": expected_savings_fraction,
        "reporting_count": count,
        "relative_precision": relative(count),
        "required_precision": target,
        "passed": relative(count) <= target,
        "reporting_points_needed": needed,
        "maximum_points": maximum,
        "model_uncertainty_floor": scale / math.sqrt(n),
        "assumption": "Fixed baseline; reporting at baseline-average weather and rate",
    }


def residual_flags(residuals, *, scale: float | None = None, threshold: float = 3) -> np.ndarray:
    """Flag standardized residuals without removing observations from a fit."""
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Residuals must be a finite nonempty vector")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive")
    deviation = float(np.std(values, ddof=1)) if scale is None and len(values) > 1 else scale
    if deviation is None:
        deviation = 0.0
    if not math.isfinite(deviation) or deviation < 0:
        raise ValueError("scale must be finite and nonnegative")
    return np.abs(values) > threshold * deviation


def ecam_regression_details(fit, temperature, usage, *, frequency="hourly") -> dict:
    """Segment standard errors and coefficient significance as in ECAM's tables.

    STEYX is the error of an independent straight-line regression within each
    segment, as in FormulasLow/Mid/HighData. It differs from the error of the
    globally continuous fit. Small segments use the overall model RMSE instead
    of producing undefined Excel errors. No observation is discarded.
    """
    x, y = np.asarray(temperature, dtype=float), np.asarray(usage, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not len(x) or not np.isfinite([x, y]).all():
        raise ValueError("Temperature and usage must be matched finite vectors")
    segment_parameters = {
        "1p": [1],
        "2p": [2],
        "3pH": [2, 1],
        "3pC": [1, 2],
        "3pHzero": [2, 0],
        "4p": [2, 2],
        "5p": [2, 1, 2],
        "6p": [2, 2, 2],
        "5pC": [1, 2, 2],
        "5pH": [2, 2, 1],
        "5pHzero": [2, 2, 0],
    }[fit.model]
    membership = np.searchsorted(fit.breakpoints, x, side="right")
    residuals = y - fit.predict(x)
    segments = []
    for segment, parameters in enumerate(segment_parameters):
        mask = membership == segment
        xs, ys = x[mask], y[mask]
        count = len(xs)
        if not count:
            raise ValueError("Cannot evaluate an empty model segment")
        mean_x = float(np.mean(xs))
        devsq = float(np.sum((xs - mean_x) ** 2))
        if fit.model == "1p":
            standard_error = float(fit.metrics.rmse)
        elif count > 2 and devsq > 0:
            local_design = np.column_stack([np.ones(count), xs - mean_x])
            local_coef = np.linalg.lstsq(local_design, ys, rcond=None)[0]
            local_residuals = ys - local_design @ local_coef
            standard_error = float(np.sqrt(np.sum(local_residuals**2) / (count - 2)))
        else:
            standard_error = float(fit.metrics.rmse)
        # Match the segment filter on the preceding point of the original
        # time-ordered residual pair, rather than correlating temperature-sorted rows.
        pair_mask = mask[:-1]
        prev, following = residuals[:-1][pair_mask], residuals[1:][pair_mask]
        rho = 0.0
        if len(prev) >= 3 and np.std(prev) > 0 and np.std(following) > 0:
            rho = abs(float(np.corrcoef(prev, following)[0, 1]))
        if frequency in {"monthly", "billing"}:
            rho = 0.0
        effective_n = max(1.0, count * (1 - rho) / (1 + rho))
        distances = [abs(mean_x - cp) for cp in fit.breakpoints]
        epsilon = min([1e-4] + [d / 4 for d in distances if d > 0])
        slope = float(
            (fit.predict([mean_x + epsilon])[0] - fit.predict([mean_x - epsilon])[0])
            / (2 * epsilon)
        )
        intercept = float(fit.predict([mean_x])[0] - slope * mean_x)
        slope_se = standard_error / math.sqrt(devsq) if devsq > 0 else None
        intercept_se = (
            standard_error * math.sqrt(1 / effective_n + mean_x**2 / devsq)
            if devsq > 0
            else standard_error / math.sqrt(effective_n)
        )
        segments.append(
            {
                "segment": segment,
                "n": count,
                "p": parameters,
                "temperature_mean": mean_x,
                "temperature_devsq": devsq,
                "standard_error": standard_error,
                "rho": rho,
                "effective_n": effective_n,
                "slope": slope,
                "intercept": intercept,
                "slope_standard_error": slope_se,
                "intercept_standard_error": intercept_se,
                "slope_t_statistic": slope / slope_se if slope_se and slope_se > 0 else None,
                "intercept_t_statistic": intercept / intercept_se if intercept_se > 0 else None,
            }
        )
    return {
        "method": "ECAM segment STEYX and coefficient statistics",
        "model": fit.model,
        "frequency": frequency,
        "segments": segments,
    }


def ecam_prediction_interval(fit, temperature, details: dict, *, confidence=0.9):
    """Port ECAM's per-segment Student-t prediction intervals (18778)."""
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    x = np.asarray(temperature, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Temperature must be a finite nonempty vector")
    if details.get("model") != fit.model:
        raise ValueError("Segment details do not match the model")
    membership = np.searchsorted(fit.breakpoints, x, side="right")
    half_width = np.empty(len(x))
    for segment in details["segments"]:
        mask = membership == segment["segment"]
        df = max(2, segment["effective_n"] - segment["p"])
        critical = float(t.ppf((1 + confidence) / 2, df))
        leverage = np.repeat(1 + 1 / segment["n"], np.count_nonzero(mask))
        if segment["temperature_devsq"] > 0:
            leverage += (x[mask] - segment["temperature_mean"]) ** 2 / segment["temperature_devsq"]
        half_width[mask] = critical * segment["standard_error"] * np.sqrt(leverage)
    prediction = fit.predict(x)
    return prediction - half_width, prediction + half_width
