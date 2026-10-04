"""Matplotlib counterparts to ECAM's M&V diagnostic and tracking charts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .analysis import Baseline, SavingsResult, _half_width


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install plotting support with pip install 'ecam-mv[plot]'") from exc
    return plt


def model_diagnostics(
    fit,
    temperature,
    usage,
    *,
    confidence=0.9,
    rate_unit="kWh/hour",
    ecam_details=None,
    predictor_label="Outdoor temperature",
):
    """Scatter with model/prediction band, residual vs temperature, lag and histogram."""
    plt = _plt()
    x, y = np.asarray(temperature), np.asarray(usage)
    residuals = y - fit.predict(x)
    grid = np.linspace(min(x), max(x), 240)
    lower, upper = fit.prediction_interval(grid, confidence)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes[0, 0].scatter(x, y, s=10, alpha=0.4, color="#367a9b", label="Measured")
    axes[0, 0].plot(grid, fit.predict(grid), color="#d66b36", label=fit.model)
    axes[0, 0].fill_between(
        grid,
        lower,
        upper,
        color="#d66b36",
        alpha=0.16,
        label=f"{confidence:.0%} prediction interval (conditional)",
    )
    if ecam_details is not None:
        from .validation import ecam_prediction_interval

        legacy_lower, legacy_upper = ecam_prediction_interval(
            fit, grid, ecam_details, confidence=confidence
        )
        axes[0, 0].plot(grid, legacy_lower, color="#7874a9", linestyle=":", linewidth=0.8)
        axes[0, 0].plot(
            grid,
            legacy_upper,
            color="#7874a9",
            linestyle=":",
            linewidth=0.8,
            label=f"{confidence:.0%} ECAM segment prediction interval",
        )
    for cp in fit.breakpoints:
        axes[0, 0].axvline(cp, color="#777777", linestyle="--", linewidth=0.8)
    axes[0, 0].set(xlabel=predictor_label, ylabel=rate_unit, title="Energy model")
    axes[0, 0].legend(fontsize=8)
    scale = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1
    scale = scale or 1
    standardized = residuals / scale
    from matplotlib.colors import BoundaryNorm, ListedColormap

    ecam_colors = ListedColormap(
        np.array(
            [
                [13, 97, 40],
                [34, 140, 61],
                [66, 170, 88],
                [103, 195, 114],
                [146, 216, 144],
                [184, 234, 179],
                [217, 243, 213],
                [255, 255, 255],
                [255, 255, 210],
                [255, 236, 156],
                [255, 203, 90],
                [255, 161, 58],
                [254, 120, 42],
                [222, 63, 22],
                [180, 16, 25],
            ]
        )
        / 255
    )
    color_norm = BoundaryNorm(
        [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], 15, clip=True
    )
    axes[0, 1].scatter(x, standardized, c=standardized, cmap=ecam_colors, norm=color_norm, s=12)
    for threshold in (-3, 0, 3):
        axes[0, 1].axhline(threshold, color="#888888", linestyle="--", linewidth=0.7)
    axes[0, 1].set(xlabel=predictor_label, ylabel="Residual / residual SD", title="Residuals")
    axes[1, 0].scatter(residuals[:-1], residuals[1:], s=10, alpha=0.45, color="#367a9b")
    axes[1, 0].set(xlabel="Previous residual", ylabel="Current residual", title="Residual lag plot")
    axes[1, 1].hist(residuals, bins=min(35, max(5, len(y) // 8)), color="#367a9b", alpha=0.8)
    axes[1, 1].set(xlabel="Observed − modeled", ylabel="Records", title="Residual histogram")
    metrics = fit.metrics
    cv = f"{metrics.cvrmse:.1%}" if metrics.cvrmse is not None else "undefined"
    bias = f"{metrics.nmbe:.1%}" if metrics.nmbe is not None else "undefined"
    fig.suptitle(f"{fit.model} · n={metrics.n} · CVRMSE={cv} · NMBE={bias}", fontsize=14)
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    return fig


def history_plot(baseline: Baseline, records):
    plt = _plt()
    predicted = baseline.predict(records)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, layout="constrained")
    axes[0].plot(records.index, records.energy, color="#367a9b", linewidth=0.8, label="Measured")
    axes[0].plot(records.index, predicted, color="#d66b36", linewidth=0.8, label="Modeled")
    axes[0].set(
        ylabel=baseline.metadata["energy_unit"], title="Measured and modeled energy history"
    )
    axes[0].legend()
    residuals = records.energy - predicted
    axes[1].plot(records.index, residuals, color="#367a9b", linewidth=0.8)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(ylabel="Observed − modeled", title="Residual history")
    return fig


def load_profile_plot(records):
    """Mean rate by hour/day type, and calendar heatmap (DST repeats averaged)."""
    if records.attrs.get("frequency") != "hourly":
        raise ValueError("Hourly records are required for load profiles")
    plt = _plt()
    frame = records.copy()
    frame["rate"] = frame.energy / frame.duration
    frame["hour"] = frame.index.hour
    frame["daytype"] = (
        frame.daytype
        if "daytype" in frame
        else np.where(frame.index.dayofweek >= 5, "weekend", "weekday")
    )
    frame["date"] = frame.index.date
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for label, group in frame.groupby("daytype"):
        profile = group.groupby("hour").rate.mean()
        axes[0].plot(profile.index, profile, label=label)
    axes[0].set(
        xlabel="Local hour", ylabel=f"{records.attrs['energy_unit']}/hour", title="Load profiles"
    )
    axes[0].legend()
    heatmap = frame.pivot_table(index="date", columns="hour", values="rate", aggfunc="mean")
    heatmap = heatmap.reindex(columns=range(24))
    im = axes[1].imshow(heatmap.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    ticks = np.linspace(0, len(heatmap) - 1, min(7, len(heatmap))).astype(int)
    axes[1].set_yticks(ticks, [str(heatmap.index[i]) for i in ticks], fontsize=8)
    axes[1].set(xlabel="Local hour", title="Calendar energy map; repeated DST hours averaged")
    fig.colorbar(im, ax=axes[1], label=f"{records.attrs['energy_unit']}/hour")
    return fig


def savings_plot(baseline: Baseline, result: SavingsResult):
    """Energy history, record savings, and cumulative savings with confidence band."""
    plt = _plt()
    records, summary = result.records, result.summary
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
    axes[0].plot(records.index, records.energy, color="#367a9b", label="Measured")
    axes[0].plot(
        records.index,
        records.adjusted_baseline + records.nonroutine_adjustment,
        color="#d66b36",
        label="Adjusted baseline",
    )
    axes[0].set(ylabel=summary["energy_unit"], title="Reporting-period M&V")
    axes[0].legend()
    colors = np.where(records.savings >= 0, "#2a8c72", "#c74739")
    axes[1].scatter(records.index, records.savings, color=colors, s=10)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(
        ylabel=summary["energy_unit"], title="Savings by record; negative = increased energy"
    )
    positions = np.unique(np.linspace(0, len(records) - 1, min(120, len(records))).astype(int))
    widths = []
    for i in positions:
        prefix = records.iloc[: i + 1]
        width = _half_width(
            baseline, prefix, summary["confidence"], summary["autocorrelation_adjusted"], True
        )
        if summary["uncertainty_method"] == "ecam":
            from .validation import ecam_total_uncertainty

            stats = baseline.metadata["ecam_statistics"]
            width = ecam_total_uncertainty(
                n=stats["n"],
                p=stats["p"],
                reporting_count=len(prefix),
                rmse=stats["rmse"],
                baseline_temperature_mean=stats["baseline_temperature_mean"],
                baseline_temperature_variance=stats["baseline_temperature_variance"],
                reporting_temperature_mean=float(prefix.temperature.mean()),
                rho=stats["rho"],
                confidence=summary["confidence"],
                frequency=summary["frequency"],
            )["uncertainty_half_width"]
        widths.append(width)
    cumulative = records.cumulative_savings.iloc[positions].to_numpy()
    dates = records.index[positions]
    axes[2].plot(records.index, records.cumulative_savings, color="#2a8c72")
    axes[2].fill_between(
        dates,
        cumulative - widths,
        cumulative + widths,
        color="#2a8c72",
        alpha=0.2,
        label=f"{summary['confidence']:.0%} pointwise interval ({summary['uncertainty_method']})",
    )
    precision = summary["precision"]["relative_precision"]
    precision_text = "undefined" if precision is None else f"{precision:.1%}"
    axes[2].set(
        ylabel=summary["energy_unit"],
        title=f"Cumulative savings: {summary['avoided_energy']:,.1f} ± "
        f"{summary['uncertainty_half_width']:,.1f}; relative precision {precision_text}",
    )
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.18)
    return fig


def precision_plot(baseline: Baseline, *, expected_savings_fraction=0.05, confidence=0.9):
    """ECAM baseline precision planning under fixed baseline assumptions."""
    from .validation import baseline_precision_plan

    plt = _plt()
    stats = baseline.metadata["ecam_statistics"]
    frequency = baseline.metadata["frequency"]
    maximum = 24 if frequency == "billing" else 17520
    counts = np.unique(np.geomspace(1, maximum, 70).astype(int))
    plans = [
        baseline_precision_plan(
            n=stats["n"],
            p=stats["p"],
            rmse=stats["rmse"],
            baseline_mean_rate=stats["baseline_mean_rate"],
            expected_savings_fraction=expected_savings_fraction,
            reporting_count=int(count),
            rho=stats["rho"],
            confidence=confidence,
            frequency=frequency,
            maximum_points=1,
        )
        for count in counts
    ]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(counts, [p["relative_precision"] * 100 for p in plans], color="#367a9b")
    ax.axhline(
        (1 - confidence) * 100, linestyle="--", color="#d66b36", label="ECAM 1−confidence target"
    )
    ax.set(
        xscale="log",
        xlabel="Prospective reporting records",
        ylabel="Relative precision (%)",
        title=f"Precision planning: {expected_savings_fraction:.0%} expected savings at "
        f"{confidence:.0%} confidence\nFixed baseline and baseline-average weather/rate",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def report_plots(
    baseline: Baseline,
    records,
    *,
    result: SavingsResult | None = None,
    directory: str | Path,
    confidence: float = 0.9,
) -> list[Path]:
    """Save diagnostic/history/load-profile/precision and optional savings PNGs."""
    plt = _plt()
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    baseline._check(records)
    figures = {"history": history_plot(baseline, records)}
    if records.attrs["frequency"] == "hourly":
        figures["load_profiles"] = load_profile_plot(records)
    keys = (
        np.repeat("all", len(records))
        if baseline.group_by is None
        else records[baseline.group_by].astype(str)
    )
    predictor_unit = (
        baseline.metadata.get("predictor_unit") or baseline.metadata["temperature_unit"]
    )
    for key, fit in baseline.models.items():
        mask = np.asarray(keys == key)
        if not mask.any():
            continue
        subset = records.iloc[np.flatnonzero(mask)]
        label = re.sub(r"[^A-Za-z0-9_-]", "_", key)
        figures[f"model_{label}"] = model_diagnostics(
            fit,
            subset.temperature.to_numpy(),
            (subset.energy / subset.duration).to_numpy(),
            confidence=confidence,
            rate_unit=f"{baseline.metadata['energy_unit']}/{baseline.metadata['duration_unit']}",
            ecam_details=baseline.metadata.get("ecam_segment_statistics", {}).get(key),
            predictor_label=(
                f"{baseline.metadata.get('predictor', 'temperature')} ({predictor_unit})"
            ),
        )
    if baseline.metadata.get("ecam_statistics", {}).get("baseline_mean_rate", 0) > 0:
        figures["precision"] = precision_plot(baseline, confidence=confidence)
    if result is not None:
        figures["savings"] = savings_plot(baseline, result)
    paths = []
    for name, fig in figures.items():
        path = destination / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
