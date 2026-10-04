from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    n_params: int
    sse: float
    rmse: float
    r2: float
    cvrmse: float
    ndb: float
    residual_std: float
    rho: float
    n_prime: float
    f_stat_ecam: float
    aic: float
    aicc: float
    bic: float


def clean_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same length")
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3:
        raise ValueError("at least 3 finite x/y pairs are required")
    return x_arr, y_arr


def lag1_rho_ecam(residuals: np.ndarray) -> float:
    """ECAM v4 uses sqrt(RSQ(r_t, r_t+1)), i.e. the magnitude of lag-1 correlation."""
    r = np.asarray(residuals, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    a, b = r[:-1], r[1:]
    if np.std(a, ddof=0) == 0 or np.std(b, ddof=0) == 0:
        return 0.0
    rho = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(rho):
        return 0.0
    return abs(rho)


def effective_sample_size(n: int, rho: float) -> float:
    rho = float(np.clip(rho, 0.0, 0.999999))
    return max(1.0, n * (1.0 - rho) / (1.0 + rho))


def compute_metrics(y, yhat, n_params: int) -> RegressionMetrics:
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    if y.shape != yhat.shape:
        raise ValueError("y and yhat must have the same length")
    n = int(y.size)
    if n <= n_params:
        raise ValueError(f"need more observations ({n}) than model parameters ({n_params})")

    residuals = y - yhat
    sse = float(np.sum(residuals**2))
    dof = n - n_params
    rmse = float(np.sqrt(sse / dof))
    mean_y = float(np.mean(y))
    cvrmse = float(rmse / mean_y) if mean_y != 0 else float("nan")
    y_sum = float(np.sum(y))
    ndb = float(np.sum(residuals) / y_sum) if y_sum != 0 else float("nan")
    sst = float(np.sum((y - mean_y) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else (1.0 if sse == 0 else float("nan"))
    residual_std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0
    rho = lag1_rho_ecam(residuals)
    n_prime = effective_sample_size(n, rho)

    # Mirrors the quantity ECAM labels F-statistic: DEVSQ(modeled) / MSE.
    explained_devsq = float(np.sum((yhat - np.mean(yhat)) ** 2))
    mse = sse / dof
    f_stat_ecam = explained_devsq / mse if mse > 0 else float("inf")

    # Information criteria are a modern addition used only for optional auto-selection.
    # Floor SSE at numerical precision so mathematically identical exact fits do not
    # let a more complex model win merely because of floating-point roundoff.
    sse_for_ic = max(sse, np.finfo(float).eps * max(float(np.sum(y**2)), 1.0))
    k = float(n_params)
    aic = float(n * np.log(sse_for_ic / n) + 2.0 * k)
    if n - n_params - 1 > 0:
        aicc = float(aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0))
    else:
        aicc = float("inf")
    bic = float(n * np.log(sse_for_ic / n) + k * np.log(n))

    return RegressionMetrics(
        n=n,
        n_params=n_params,
        sse=sse,
        rmse=rmse,
        r2=float(r2),
        cvrmse=cvrmse,
        ndb=ndb,
        residual_std=residual_std,
        rho=rho,
        n_prime=float(n_prime),
        f_stat_ecam=float(f_stat_ecam),
        aic=aic,
        aicc=aicc,
        bic=bic,
    )
