"""Continuous change-point regression models used by ECAM M&V."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t

from .metrics import RegressionMetrics, regression_metrics

_MODEL_PARAMETERS = {
    "1p": (1, 0),
    "2p": (2, 0),
    "3pH": (2, 1),
    "3pC": (2, 1),
    "4p": (3, 1),
    "5p": (3, 2),
    "6p": (4, 2),
    "5pH": (3, 2),
    "5pC": (3, 2),
    "3pHzero": (1, 1),
    "5pHzero": (2, 2),
}
_SCHEMA_VERSION = 2


@dataclass
class FitResult:
    model: str
    breakpoints: tuple[float, ...]
    coefficients: np.ndarray
    covariance: np.ndarray
    noise_variance: float
    residuals: np.ndarray
    metrics: RegressionMetrics
    n_parameters: int
    x_min: float
    x_max: float
    selection: dict[str, Any] | None = None
    _weighted: bool = False

    def design_matrix(self, x: object) -> np.ndarray:
        values = _prediction_x(x)
        return _basis(self.model, values, self.breakpoints)

    def predict(self, x: object) -> np.ndarray:
        return self.design_matrix(x) @ self.coefficients

    def prediction_interval(
        self, x: object, confidence: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return prediction bounds for new observations at relative weight 1."""
        confidence = _validate_confidence(confidence)
        design = self.design_matrix(x)
        variance = np.einsum("ij,jk,ik->i", design, self.covariance, design)
        half_width = _critical_value(confidence, self.metrics.dof) * np.sqrt(
            np.maximum(0.0, variance) + self.noise_variance
        )
        center = design @ self.coefficients
        return center - half_width, center + half_width

    def aggregate_uncertainty(
        self,
        x: object,
        exposure: object | None = None,
        confidence: float = 0.9,
        autocorrelation: bool = False,
        include_noise: bool = True,
    ) -> float:
        """Return an aggregate half-width, including noise at relative weight 1."""
        confidence = _validate_confidence(confidence)
        design = self.design_matrix(x)
        exp = np.ones(design.shape[0]) if exposure is None else _finite_vector(exposure, "exposure")
        if exp.shape != (design.shape[0],):
            raise ValueError("exposure must have the same length as x")
        if np.any(exp <= 0):
            raise ValueError("exposure values must be positive")
        summed_design = exp @ design
        variance = float(summed_design @ self.covariance @ summed_design)
        if include_noise:
            variance += self.noise_variance * float(exp @ exp)
        if autocorrelation:
            rho = self.metrics.rho
            factor = 1.0 if rho is None else max(1.0, (1.0 + rho) / max(1e-12, 1.0 - rho))
            variance *= factor
        return float(_critical_value(confidence, self.metrics.dof) * np.sqrt(max(0.0, variance)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fitted model to finite JSON-compatible values."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "model": self.model,
            "breakpoints": list(self.breakpoints),
            "coefficients": self.coefficients.tolist(),
            "covariance": self.covariance.tolist(),
            "noise_variance": self.noise_variance,
            "residuals": self.residuals.tolist(),
            "metrics": self.metrics.to_dict(),
            "n_parameters": self.n_parameters,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "selection": self.selection,
            "weighted": self._weighted,
        }

    @classmethod
    def from_dict(cls, data: object) -> "FitResult":
        if not isinstance(data, dict) or data.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("unsupported or invalid FitResult schema")
        try:
            model = data["model"]
            if model not in _MODEL_PARAMETERS:
                raise ValueError("unknown model")
            coefficients = _finite_vector(data["coefficients"], "coefficients")
            covariance = np.asarray(data["covariance"], dtype=float)
            if isinstance(data["noise_variance"], bool):
                raise ValueError("noise_variance must be numeric")
            noise_variance = float(data["noise_variance"])
            residuals = _finite_vector(data["residuals"], "residuals")
            breakpoints = tuple(_finite_vector(data["breakpoints"], "breakpoints"))
            k, b = _MODEL_PARAMETERS[model]
            n_parameters = k + b
            if coefficients.size != k or covariance.shape != (k, k) or len(breakpoints) != b:
                raise ValueError("serialized model dimensions do not match model")
            if not np.all(np.isfinite(covariance)) or not np.allclose(covariance, covariance.T):
                raise ValueError("covariance must be finite and symmetric")
            covariance_scale = max(1.0, float(np.max(np.abs(covariance))))
            if np.linalg.eigvalsh(covariance).min() < -1e-10 * covariance_scale:
                raise ValueError("covariance must be positive semidefinite")
            if not np.isfinite(noise_variance) or noise_variance < 0:
                raise ValueError("noise_variance must be finite and nonnegative")
            serialized_n_parameters = data["n_parameters"]
            if (
                isinstance(serialized_n_parameters, bool)
                or not isinstance(serialized_n_parameters, int)
                or serialized_n_parameters != n_parameters
            ):
                raise ValueError("serialized n_parameters does not match model")
            metrics_data = data["metrics"]
            if not isinstance(metrics_data, dict):
                raise ValueError("serialized metrics must be an object")
            metrics = RegressionMetrics(**metrics_data)
            if (
                isinstance(metrics.n, bool)
                or not isinstance(metrics.n, int)
                or isinstance(metrics.p, bool)
                or not isinstance(metrics.p, int)
                or isinstance(metrics.dof, bool)
                or not isinstance(metrics.dof, int)
                or metrics.n != residuals.size
                or metrics.p != n_parameters
                or metrics.dof != metrics.n - metrics.p
                or metrics.dof <= 0
            ):
                raise ValueError("serialized metrics do not match model dimensions")
            _validate_serialized_metrics(metrics)
            x_min, x_max = float(data["x_min"]), float(data["x_max"])
            equal_x_is_valid = model == "1p" and x_min == x_max
            if not np.isfinite([x_min, x_max]).all() or (x_min >= x_max and not equal_x_is_valid):
                raise ValueError("invalid serialized x bounds")
            if any(
                breakpoints[i] >= breakpoints[i + 1] for i in range(len(breakpoints) - 1)
            ) or any(bp <= x_min or bp >= x_max for bp in breakpoints):
                raise ValueError("invalid serialized breakpoints")
            selection = data.get("selection")
            if selection is not None and not isinstance(selection, dict):
                raise ValueError("selection must be an object or null")
            json.dumps(selection, allow_nan=False)
            weighted = data.get("weighted", False)
            if not isinstance(weighted, bool):
                raise ValueError("weighted must be a boolean")
            return cls(
                model,
                breakpoints,
                coefficients,
                covariance,
                noise_variance,
                residuals,
                metrics,
                n_parameters,
                x_min,
                x_max,
                selection,
                weighted,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("serialized"):
                raise
            raise ValueError(f"invalid FitResult data: {exc}") from exc


def fit_model(
    temperature: object,
    usage: object,
    model: str = "3pC",
    *,
    weights: object | None = None,
    bias_weights: object | None = None,
    min_segment: int = 3,
) -> FitResult:
    """Fit a continuous hinge regression with deterministic breakpoint search.

    Coefficients are unconstrained, so negative slopes are permitted. Weights
    define WLS objective multipliers; ``bias_weights`` independently impose an
    exact weighted-usage equality constraint.
    """
    x, y = _training_vectors(temperature, usage)
    if model not in _MODEL_PARAMETERS:
        raise ValueError(f"unknown model {model!r}")
    if (
        isinstance(min_segment, bool)
        or not isinstance(min_segment, (int, np.integer))
        or min_segment < 1
    ):
        raise ValueError("min_segment must be a positive integer")
    k, breakpoint_count = _MODEL_PARAMETERS[model]
    p = k + breakpoint_count
    if x.size <= p:
        raise ValueError("the number of observations must exceed model n_parameters")
    fit_weights = (
        _positive_vector(weights, x.size, "weights") if weights is not None else np.ones(x.size)
    )
    fit_weights = fit_weights / np.mean(fit_weights)
    bias = (
        _positive_vector(bias_weights, x.size, "bias_weights") if bias_weights is not None else None
    )
    if bias is not None and not np.any(bias > 0):
        raise ValueError("bias_weights must contain positive values")
    if np.ptp(y) == 0 and model != "1p":
        raise ValueError("constant responses cannot identify the requested regression model")
    if breakpoint_count and np.unique(x).size < (breakpoint_count + 1) * min_segment:
        raise ValueError("insufficient distinct temperatures for requested segment coverage")

    sorted_x = np.sort(x)
    if breakpoint_count == 0:
        starts = [()]
    else:
        # A modest deterministic quantile grid keeps two-breakpoint searches
        # tractable while the bounded optimizer refines the best grid starts.
        lo, hi = min_segment - 1, x.size - min_segment
        grid_size = 11 if breakpoint_count == 2 else 15
        indices = np.linspace(lo, hi, min(grid_size, hi - lo + 1)).round().astype(int)
        candidate_values = np.unique(sorted_x[indices])
        starts = [
            bp
            for bp in combinations(candidate_values, breakpoint_count)
            if _segments_valid(x, bp, min_segment)
        ]
    if not starts:
        raise ValueError("no breakpoint configuration satisfies min_segment")

    best: tuple[float, tuple[float, ...], np.ndarray, np.ndarray, np.ndarray, float] | None = None
    for initial in starts:
        optimized = _refine_breakpoints(x, y, model, tuple(initial), fit_weights, bias, min_segment)
        if optimized is None:
            continue
        bps, coefficients, covariance, residuals, objective, noise_variance = optimized
        if best is None or objective < best[0]:
            best = (objective, bps, coefficients, covariance, residuals, noise_variance)
    if best is None:
        raise ValueError("no numerically identifiable fit satisfies the requested model")
    _, breakpoints, coefficients, covariance, residuals, noise_variance = best
    predicted = y - residuals
    metrics = regression_metrics(y, predicted, p)
    return FitResult(
        model=model,
        breakpoints=breakpoints,
        coefficients=coefficients,
        covariance=covariance,
        noise_variance=noise_variance,
        residuals=residuals,
        metrics=metrics,
        n_parameters=p,
        x_min=float(np.min(x)),
        x_max=float(np.max(x)),
        _weighted=weights is not None,
    )


def select_model(
    x: object,
    y: object,
    *,
    candidates: object | None = None,
    weights: object | None = None,
    bias_weights: object | None = None,
    min_segment: int = 3,
) -> FitResult:
    """Fit candidates and return the smallest finite-sample corrected AIC."""
    names = list(_MODEL_PARAMETERS) if candidates is None else list(candidates)
    if not names or any(name not in _MODEL_PARAMETERS for name in names):
        raise ValueError("candidates must contain one or more supported model IDs")
    fits: list[tuple[float, FitResult]] = []
    scores: dict[str, float] = {}
    rejected: dict[str, str] = {}
    n = len(_training_vectors(x, y)[0])
    normalized_weights = None
    if weights is not None:
        normalized_weights = _positive_vector(weights, n, "weights")
        normalized_weights = normalized_weights / np.mean(normalized_weights)
    for name in names:
        try:
            fit = fit_model(
                x, y, name, weights=weights, bias_weights=bias_weights, min_segment=min_segment
            )
            residual_sum_squares = float(fit.residuals @ fit.residuals)
            if normalized_weights is not None:
                residual_sum_squares = float(normalized_weights @ (fit.residuals**2))
            rss = max(residual_sum_squares, np.finfo(float).tiny)
            aic = n * np.log(rss / n) + 2 * fit.n_parameters
            if n <= fit.n_parameters + 1:
                raise ValueError("AICc is undefined when n <= parameters + 1")
            aicc = float(
                aic + 2 * fit.n_parameters * (fit.n_parameters + 1) / (n - fit.n_parameters - 1)
            )
            fits.append((aicc, fit))
            scores[name] = aicc
        except ValueError as exc:
            rejected[name] = str(exc)
    if not fits:
        raise ValueError(f"no candidate model could be fitted: {rejected}")
    score, chosen = min(fits, key=lambda item: item[0])
    chosen.selection = {
        "criterion": "AICc",
        "score": score,
        "scores": scores,
        "rejected": rejected,
    }
    return chosen


def _refine_breakpoints(x, y, model, initial, weights, bias, min_segment):
    count = len(initial)
    if not count:
        solved = _solve_linear(x, y, model, (), weights, bias)
        return None if solved is None else ((), *solved)

    # Breakpoints are optimized within quantile-derived feasible bounds. Segment
    # constraints are checked at every objective evaluation; invalid points are
    # assigned a smooth distance penalty to keep the deterministic search stable.
    lower = float(np.sort(x)[min_segment - 1])
    upper = float(np.sort(x)[-min_segment])
    best = None
    for seed in (initial,):

        def objective(bp, seed=seed):
            bps = tuple(float(v) for v in bp)
            if not _segments_valid(x, bps, min_segment):
                violation = sum(max(0, min_segment - c) for c in _segment_counts(x, bps))
                return 1e12 + violation * 1e10 + float(np.sum(np.square(bp - seed)))
            solved = _solve_linear(x, y, model, bps, weights, bias)
            return 1e12 if solved is None else solved[-1]

        constraints = [{"type": "ineq", "fun": lambda z: z[1] - z[0] - 1e-12}] if count == 2 else []
        result = minimize(
            objective,
            np.asarray(seed),
            method="SLSQP",
            bounds=[(lower, upper)] * count,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-10},
        )
        possible = [np.asarray(seed), np.asarray(result.x)]
        for bp in possible:
            bps = tuple(float(v) for v in bp)
            if _segments_valid(x, bps, min_segment):
                solved = _solve_linear(x, y, model, bps, weights, bias)
                if solved is not None and (best is None or solved[-1] < best[-1]):
                    best = (bps, *solved)
    return best


def _solve_linear(x, y, model, breakpoints, weights, bias):
    design = _basis(model, x, breakpoints)
    root_w = np.sqrt(weights)
    a, target = design * root_w[:, None], y * root_w
    _, _, rank, _ = np.linalg.lstsq(a, target, rcond=None)
    if rank < design.shape[1]:
        return None
    pinv_a = np.linalg.pinv(a)
    estimator = pinv_a * root_w[None, :]
    if bias is None:
        linear_map = estimator
    else:
        constraint = bias @ design
        if np.linalg.norm(constraint) == 0:
            return None
        normal_pseudoinverse = pinv_a @ pinv_a.T
        direction = normal_pseudoinverse @ constraint
        denom = float(constraint @ direction)
        if denom <= np.finfo(float).eps:
            return None
        linear_map = estimator + np.outer(direction / denom, bias - constraint @ estimator)
    coefficients = linear_map @ y
    residuals = y - design @ coefficients
    objective = float(np.sum(weights * residuals**2))
    dof = len(y) - design.shape[1] - len(breakpoints)
    sigma2 = float(objective / dof) if dof > 0 else 0.0
    scaled_map = linear_map / root_w[None, :]
    covariance = sigma2 * (scaled_map @ scaled_map.T)
    covariance = (covariance + covariance.T) * 0.5
    return coefficients, covariance, residuals, objective, sigma2


def _basis(model: str, x: np.ndarray, bp: tuple[float, ...]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if model == "1p":
        return np.ones((x.size, 1))
    if model == "2p":
        return np.column_stack((np.ones(x.size), x))
    if model in {"3pH", "3pHzero"}:
        heat = np.maximum(bp[0] - x, 0.0)
        return heat[:, None] if model.endswith("zero") else np.column_stack((np.ones(x.size), heat))
    if model == "3pC":
        return np.column_stack((np.ones(x.size), np.maximum(x - bp[0], 0.0)))
    if model == "4p":
        delta = x - bp[0]
        return np.column_stack((np.ones(x.size), np.minimum(delta, 0.0), np.maximum(delta, 0.0)))
    if model == "5p":
        return np.column_stack(
            (np.ones(x.size), np.maximum(bp[0] - x, 0.0), np.maximum(x - bp[1], 0.0))
        )
    if model == "5pC":
        return np.column_stack(
            (np.ones(x.size), np.maximum(x - bp[0], 0.0), np.maximum(x - bp[1], 0.0))
        )
    if model in {"5pH", "5pHzero"}:
        left = np.minimum(x - bp[0], 0.0)
        if model == "5pHzero":
            middle = np.minimum(x - bp[1], 0.0) - left
            return np.column_stack((left, middle))
        middle = np.maximum(x - bp[0], 0.0) - np.maximum(x - bp[1], 0.0)
        return np.column_stack((np.ones(x.size), left, middle))
    if model == "6p":
        first = x - bp[0]
        return np.column_stack(
            (
                np.ones(x.size),
                np.minimum(first, 0.0),
                np.maximum(first, 0.0),
                np.maximum(x - bp[1], 0.0),
            )
        )
    raise ValueError(f"unknown model {model!r}")


def _segments_valid(x, bp, minimum):
    return all(count >= minimum for count in _segment_counts(x, bp))


def _segment_counts(x, bp):
    if len(bp) == 1:
        return (int(np.sum(x < bp[0])), int(np.sum(x >= bp[0])))
    return (
        int(np.sum(x < bp[0])),
        int(np.sum((x >= bp[0]) & (x < bp[1]))),
        int(np.sum(x >= bp[1])),
    )


def _training_vectors(x, y):
    xv, yv = _finite_vector(x, "temperature"), _finite_vector(y, "usage")
    if xv.size != yv.size:
        raise ValueError("temperature and usage must have the same length")
    if xv.size == 0:
        raise ValueError("temperature and usage must not be empty")
    return xv, yv


def _finite_vector(value, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional numeric array") from exc
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional numeric array")
    return result


def _positive_vector(value, size, name):
    result = _finite_vector(value, name)
    if result.shape != (size,) or np.any(result <= 0):
        raise ValueError(f"{name} must contain one positive value per observation")
    return result


def _prediction_x(value):
    result = _finite_vector(value, "x")
    if result.size == 0:
        raise ValueError("x must not be empty")
    return result


def _validate_serialized_metrics(metrics: RegressionMetrics) -> None:
    for name in ("rmse", "cvrmse", "nmbe", "net_bias", "r_squared", "rho"):
        value = getattr(metrics, name)
        if value is not None and (isinstance(value, bool) or not np.isfinite(value)):
            raise ValueError(f"serialized metric {name} must be finite or null")
    if metrics.rmse is not None and metrics.rmse < 0:
        raise ValueError("serialized RMSE must be nonnegative")
    if metrics.cvrmse is not None and metrics.cvrmse < 0:
        raise ValueError("serialized CVRMSE must be nonnegative")
    if metrics.rho is not None and not -1 <= metrics.rho <= 1:
        raise ValueError("serialized rho must be in [-1, 1]")


def _validate_confidence(value):
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be strictly between 0 and 1") from exc
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    return confidence


def _critical_value(confidence, dof):
    return float(student_t.ppf((1 + confidence) / 2, max(1, dof)))
