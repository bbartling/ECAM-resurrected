from __future__ import annotations

import csv
from pathlib import Path
import numpy as np


def csv_headers(path: str | Path) -> list[str]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            return [h.strip() for h in next(reader)]
        except StopIteration:
            return []


def read_xy_csv(path: str | Path, x_column: str, y_column: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        if x_column not in reader.fieldnames or y_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns '{x_column}' and '{y_column}'")
        for row in reader:
            try:
                x = float(row[x_column])
                y = float(row[y_column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
    if len(xs) < 3:
        raise ValueError("CSV contains fewer than 3 numeric x/y rows")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def write_fit_csv(path: str | Path, x, y, yhat, residuals) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "measured_y", "modeled_y", "residual"])
        for row in zip(x, y, yhat, residuals):
            writer.writerow([float(v) for v in row])
