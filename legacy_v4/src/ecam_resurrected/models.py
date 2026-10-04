from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

from .metrics import RegressionMetrics, clean_xy, compute_metrics

MODEL_ALIASES = {
    "2p": "2p",
    "2": "2p",
    "linear": "2p",
    "3p_heat": "3p_heat",
    "3p heat": "3p_heat",
    "3ph": "3p_heat",
    "3p_cool": "3p_cool",
    "3p cool": "3p_cool",
    "3pc": "3p_cool",
    "4p": "4p",
    "4": "4p",
    "5p": "5p",
    "5": "5p",
    "6p": "6p",
    "6": "6p",
}

MODEL_PARAM_COUNTS = {
    "2p": 2,
    "3p_heat": 3,
    "3p_cool": 3,
    "4p": 4,
    "5p": 5,
    "6p": 6,
}


@dataclass
class FitResult:
    model_type: str
    coefficients: np.ndarray
    change_points: tuple[float, ...]
    x: np.ndarray
    y: np.ndarray
    yhat: np.ndarray
    residuals: np.ndarray
    metrics: RegressionMetrics
    covariance: np.ndarray
    design_matrix: np.ndarray
    segment_counts: dict[str, int]

    @property
    def params(self) -> dict[str, float]:
        c = self.coefficients
        if self.model_type == "2p":
            return {"intercept": float(c[0]), "slope": float(c[1])}
        if self.model_type == "3p_heat":
            cp = self.change_points[0]
            return {"change_point": cp, "base": float(c[0]), "heating_slope": float(c[1])}
        if self.model_type == "3p_cool":
            cp = self.change_points[0]
            return {"change_point": cp, "base": float(c[0]), "cooling_slope": float(c[1])}
        if self.model_type == "4p":
            cp = self.change_points[0]
            return {
                "change_point": cp,
                "value_at_change_point": float(c[0]),
                "low_slope": float(c[1]),
                "high_slope": float(c[2]),
            }
        if self.model_type == "5p":
            cp1, cp2 = self.change_points
            return {
                "change_point_low": cp1,
                "change_point_high": cp2,
                "middle_base": float(c[0]),
                "low_slope": float(c[1]),
                "middle_slope": 0.0,
                "high_slope": float(c[2]),
            }
        cp1, cp2 = self.change_points
        return {
            "change_point_low": cp1,
            "change_point_high": cp2,
            "value_at_low_change_point": float(c[0]),
            "low_slope": float(c[1]),
            "middle_slope": float(c[2]),
            "high_slope": float(c[3]),
        }

    def predict(self, x_new) -> np.ndarray:
        return predict(self, x_new)


def _normalize_model(model_type: str) -> str:
    key = str(model_type).strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"unsupported model '{model_type}'. Choose from 2p, 3p_heat, 3p_cool, 4p, 5p, 6p")
    return MODEL_ALIASES[key]


def _design_matrix(model_type: str, x: np.ndarray, change_points: tuple[float, ...] = ()) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if model_type == "2p":
        return np.column_stack([np.ones_like(x), x])

    if model_type in {"3p_heat", "3p_cool", "4p"}:
        cp = float(change_points[0])
        dx = x - cp
        if model_type == "3p_heat":
            return np.column_stack([np.ones_like(x), np.minimum(dx, 0.0)])
        if model_type == "3p_cool":
            return np.column_stack([np.ones_like(x), np.maximum(dx, 0.0)])
        return np.column_stack([np.ones_like(x), np.minimum(dx, 0.0), np.maximum(dx, 0.0)])

    cp1, cp2 = map(float, change_points)
    if not cp1 < cp2:
        raise ValueError("lower change point must be less than upper change point")
    low_hinge = np.minimum(x - cp1, 0.0)
    high_hinge = np.maximum(x - cp2, 0.0)
    if model_type == "5p":
        return np.column_stack([np.ones_like(x), low_hinge, high_hinge])

    # 6P: three continuous linear segments. The clipped middle ramp makes each
    # coefficient directly interpretable as its segment slope.
    middle_ramp = np.clip(x - cp1, 0.0, cp2 - cp1)
    return np.column_stack([np.ones_like(x), low_hinge, middle_ramp, high_hinge])


def _segment_counts(model_type: str, x: np.ndarray, cps: tuple[float, ...]) -> dict[str, int]:
    if model_type == "2p":
        return {"all": int(x.size)}
    if len(cps) == 1:
        cp = cps[0]
        return {"low": int(np.sum(x < cp)), "high": int(np.sum(x >= cp))}
    cp1, cp2 = cps
    return {
        "low": int(np.sum(x < cp1)),
        "mid": int(np.sum((x >= cp1) & (x < cp2))),
        "high": int(np.sum(x >= cp2)),
    }


def _fit_fixed(model_type: str, x: np.ndarray, y: np.ndarray, cps: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    X = _design_matrix(model_type, x, cps)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < X.shape[1]:
        return beta, X, np.full_like(y, np.nan), float("inf")
    yhat = X @ beta
    residuals = y - yhat
    sse = float(np.sum(residuals**2))
    return beta, X, yhat, sse


def _valid_one_cp(x: np.ndarray, cp: float, min_segment_points: int) -> bool:
    return np.sum(x < cp) >= min_segment_points and np.sum(x >= cp) >= min_segment_points


def _valid_two_cp(x: np.ndarray, cp1: float, cp2: float, min_segment_points: int) -> bool:
    return (
        cp1 < cp2
        and np.sum(x < cp1) >= min_segment_points
        and np.sum((x >= cp1) & (x < cp2)) >= min_segment_points
        and np.sum(x >= cp2) >= min_segment_points
    )


def _one_cp_bounds(x: np.ndarray, min_segment_points: int) -> tuple[float, float]:
    sx = np.sort(np.unique(x))
    need = min_segment_points + 1
    if sx.size < 2 * need:
        raise ValueError("not enough unique x values for a change-point model")
    # ECAM v4 bounds its search away from the extremes (roughly the 3rd unique values).
    lo = float(sx[min_segment_points])
    hi = float(sx[-min_segment_points - 1])
    if not lo < hi:
        raise ValueError("x range is too narrow for a change-point model")
    return lo, hi


def _search_one_cp(model_type: str, x: np.ndarray, y: np.ndarray, min_segment_points: int, grid_size: int) -> float:
    lo, hi = _one_cp_bounds(x, min_segment_points)
    best_cp, best_sse = None, float("inf")
    left, right = lo, hi
    for _ in range(6):
        for cp in np.linspace(left, right, max(15, grid_size)):
            cp = float(cp)
            if not _valid_one_cp(x, cp, min_segment_points):
                continue
            _, _, _, sse = _fit_fixed(model_type, x, y, (cp,))
            if sse < best_sse:
                best_cp, best_sse = cp, sse
        if best_cp is None:
            raise ValueError("unable to find a valid change point")
        width = (right - left) / max(14, grid_size - 1)
        left = max(lo, best_cp - width)
        right = min(hi, best_cp + width)
        grid_size = 21
    return float(best_cp)


def _two_cp_bounds(x: np.ndarray, min_segment_points: int) -> tuple[float, float]:
    sx = np.sort(np.unique(x))
    if sx.size < 3 * min_segment_points + 2:
        raise ValueError("not enough unique x values for a two-change-point model")
    return float(sx[min_segment_points]), float(sx[-min_segment_points - 1])


def _search_two_cp(model_type: str, x: np.ndarray, y: np.ndarray, min_segment_points: int, grid_size: int) -> tuple[float, float]:
    lo, hi = _two_cp_bounds(x, min_segment_points)
    best = None
    best_sse = float("inf")
    left1, right1, left2, right2 = lo, hi, lo, hi
    coarse = max(17, min(grid_size, 41))
    for iteration in range(5):
        n_grid = coarse if iteration == 0 else 17
        g1 = np.linspace(left1, right1, n_grid)
        g2 = np.linspace(left2, right2, n_grid)
        for cp1 in g1:
            for cp2 in g2:
                cp1f, cp2f = float(cp1), float(cp2)
                if not _valid_two_cp(x, cp1f, cp2f, min_segment_points):
                    continue
                _, _, _, sse = _fit_fixed(model_type, x, y, (cp1f, cp2f))
                if sse < best_sse:
                    best_sse = sse
                    best = (cp1f, cp2f)
        if best is None:
            raise ValueError("unable to find valid two change points")
        step1 = (right1 - left1) / max(1, n_grid - 1)
        step2 = (right2 - left2) / max(1, n_grid - 1)
        left1, right1 = max(lo, best[0] - step1), min(hi, best[0] + step1)
        left2, right2 = max(lo, best[1] - step2), min(hi, best[1] + step2)
    return float(best[0]), float(best[1])


def fit_model(
    x,
    y,
    model_type: str = "2p",
    *,
    min_segment_points: int = 3,
    grid_size: int = 41,
) -> FitResult:
    """Fit an ECAM-style continuous temperature/energy model.

    ECAM v4 used grid search followed by Excel Solver. This implementation uses
    deterministic change-point grid refinement and NumPy least squares at each
    candidate point. Change points count as fitted parameters when computing RMSE,
    matching ECAM's nParams convention.
    """
    model_type = _normalize_model(model_type)
    x_arr, y_arr = clean_xy(x, y)
    if min_segment_points < 2:
        raise ValueError("min_segment_points must be at least 2")

    if model_type == "2p":
        cps: tuple[float, ...] = ()
    elif model_type in {"3p_heat", "3p_cool", "4p"}:
        cps = (_search_one_cp(model_type, x_arr, y_arr, min_segment_points, grid_size),)
    else:
        cps = _search_two_cp(model_type, x_arr, y_arr, min_segment_points, grid_size)

    beta, X, yhat, sse = _fit_fixed(model_type, x_arr, y_arr, cps)
    if not np.isfinite(sse):
        raise ValueError("model design matrix is rank deficient for these data")

    n_params = MODEL_PARAM_COUNTS[model_type]
    metrics = compute_metrics(y_arr, yhat, n_params)
    dof = y_arr.size - n_params
    mse = metrics.sse / dof
    covariance = mse * np.linalg.pinv(X.T @ X)

    return FitResult(
        model_type=model_type,
        coefficients=beta,
        change_points=tuple(map(float, cps)),
        x=x_arr,
        y=y_arr,
        yhat=yhat,
        residuals=y_arr - yhat,
        metrics=metrics,
        covariance=covariance,
        design_matrix=X,
        segment_counts=_segment_counts(model_type, x_arr, cps),
    )


def fit_best_model(
    x,
    y,
    *,
    models: Iterable[str] = ("2p", "3p_heat", "3p_cool", "4p", "5p", "6p"),
    criterion: str = "aicc",
    min_segment_points: int = 3,
    grid_size: int = 35,
) -> FitResult:
    """Modern convenience function; ECAM v4 itself asked the user to choose a model."""
    criterion = criterion.lower()
    if criterion not in {"aic", "aicc", "bic"}:
        raise ValueError("criterion must be aic, aicc, or bic")
    fitted: list[FitResult] = []
    errors: list[str] = []
    for model in models:
        try:
            fitted.append(
                fit_model(
                    x,
                    y,
                    model,
                    min_segment_points=min_segment_points,
                    grid_size=grid_size,
                )
            )
        except ValueError as exc:
            errors.append(f"{model}: {exc}")
    if not fitted:
        raise ValueError("no candidate model could be fit; " + "; ".join(errors))
    return min(fitted, key=lambda r: getattr(r.metrics, criterion))


def predict(result: FitResult, x_new) -> np.ndarray:
    x_arr = np.asarray(x_new, dtype=float).reshape(-1)
    X = _design_matrix(result.model_type, x_arr, result.change_points)
    return X @ result.coefficients


def design_matrix_for_result(result: FitResult, x_new) -> np.ndarray:
    return _design_matrix(result.model_type, np.asarray(x_new, dtype=float), result.change_points)
