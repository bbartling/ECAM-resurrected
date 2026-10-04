# ECAM Resurrected

**ECAM Resurrected** is an independent Python modernization of the core measurement and verification (M&V) regression workflows found in the 2016 ECAM v4 Excel add-in. It does **not** require Microsoft Excel.

This project was created in appreciation of the work of **Bill Koran, the original creator of Energy Charting and Metrics (ECAM)**, and the many contributors and organizations that helped develop and support ECAM over the years. The goal is to help preserve the engineering ideas behind this excellent, freely available building-energy analysis tool and carry them forward in a modern Python implementation.

ECAM Resurrected translates key concepts from the legacy **Energy Charting and Metrics (ECAM) Excel/VBA add-in** into maintainable, testable Python code. The project focuses particularly on ECAM's M&V functionality, including weather-dependent baseline regression, change-point models, weather normalization, savings calculations, regression statistics, and uncertainty analysis.

The intent is not to replace the original ECAM project or diminish the work that came before it. Instead, ECAM Resurrected aims to preserve that work, recognize the engineers who developed it, and make these useful building-energy analysis methods easier to study, maintain, extend, and use with modern data-analysis workflows.

## From Excel/VBA to Python

This project was initially developed by studying VBA source extracted from:

`ECAM_v4_2016-03-01.xlam`

The original ECAM implementation relied heavily on Excel worksheet formulas, named ranges, PivotTables, VBA forms, iterative change-point searches, and Excel Solver.

ECAM Resurrected separates the underlying engineering and statistical methods from that Excel-specific infrastructure. The Python implementation uses tools such as **NumPy and SciPy** for numerical analysis and optimization, along with a lightweight **Tkinter desktop interface** for users who prefer a graphical workflow.

The objective is to retain the useful M&V methodology and behavior of the legacy tool while providing a codebase that can be tested, inspected, automated, extended, and integrated into modern Python-based building analytics workflows.

> **Important:** ECAM Resurrected is an independent compatibility, preservation, and research project. It is **not an official release of ECAM** and is not affiliated with or endorsed by Lattice Energy Works, SBW Consulting, PNNL, BPA, Bill Koran, or other organizations or individuals associated with the original ECAM project. The name **“ECAM Resurrected”** is used as a descriptive project name for this independent modernization effort.

## Implemented models

- **2P** — ordinary two-parameter linear regression.
- **3P Heating** — low-temperature weather slope with a high-temperature plateau.
- **3P Cooling** — low-temperature plateau with a high-temperature weather slope.
- **4P** — two continuous slopes meeting at one change point.
- **5P** — heating slope + flat middle + cooling slope, with two change points.
- **6P** — three continuous linear segments with two change points.

The original ECAM v4 change-point routines used a grid search followed by Excel Solver. ECAM Resurrected uses deterministic grid refinement with NumPy least-squares fits at each candidate change point. This makes the calculations reproducible and removes the Excel Solver dependency.

## M&V statistics carried over from ECAM v4

The implementation includes RMSE, R², CV(RMSE), net determination bias, ECAM-style lag-1 residual autocorrelation magnitude, autocorrelation-adjusted effective sample size (`n'`), critical Student-t values, prediction uncertainty, ECAM's fractional-savings uncertainty approximation, avoided energy use, and normalized savings.

ECAM v4's important caveat is preserved: **the reported savings uncertainty represents regression/model uncertainty only.** It does not automatically include metering uncertainty, omitted-variable effects, model-form uncertainty, or poor extrapolation outside baseline operating conditions.

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Tkinter (normally included with Windows Python; on some Linux distributions it is a separate OS package)
- pytest only for running tests

## Install

From the project folder:

```bash
python -m pip install -e .
```

For development/testing:

```bash
python -m pip install -e ".[dev]"
```

## Launch the desktop UI

```bash
python -m ecam_resurrected
```

or after installation:

```bash
ecam-resurrected-ui
```

On Windows you can also double-click `launch_ecam_resurrected.bat` after installing the package.

### UI workflow

1. Choose a baseline CSV.
2. Select the temperature/independent-variable column and energy/dependent-variable column.
3. Choose a model, or **Auto (AICc)**.
4. Click **Fit Baseline**.
5. Optionally choose a post-period CSV and click **Savings**.
6. Export the fitted measured/modeled/residual data to CSV if desired.

The Auto mode is a **modern convenience** and was not ECAM v4's normal workflow. It evaluates available candidate models and selects the minimum AICc model. For formal M&V, model selection should still consider engineering plausibility and the project M&V plan, not only an information criterion.

## CLI examples

Fit a 5P model:

```bash
ecam-resurrected fit examples/example_baseline.csv \
  --x temperature --y energy --model 5p
```

Automatically choose by AICc:

```bash
ecam-resurrected fit examples/example_baseline.csv \
  --x temperature --y energy --model auto
```

Calculate avoided energy use:

```bash
ecam-resurrected savings examples/example_baseline.csv examples/example_post.csv \
  --x temperature --y energy --model 3p_heat --confidence 0.80
```

Export fitted values and residuals:

```bash
ecam-resurrected fit examples/example_baseline.csv \
  --x temperature --y energy --model 3p_heat --export fitted.csv
```

## Python API

```python
import numpy as np
from ecam_resurrected import fit_model, avoided_energy_savings

x_base = np.array([30, 40, 50, 60, 70, 80, 90], dtype=float)
y_base = np.array([180, 155, 130, 105, 100, 100, 100], dtype=float)

model = fit_model(x_base, y_base, "3p_heat")
print(model.params)
print(model.metrics.cvrmse)

x_post = x_base.copy()
y_post = y_base * 0.90
savings = avoided_energy_savings(model, x_post, y_post, confidence=0.80)
print(savings.savings, savings.uncertainty)
```

## Tests

```bash
pytest
```

The test suite includes exact synthetic data for all six ECAM model families, statistics, uncertainty functions, savings calculations, and CSV I/O.

## Fidelity notes

This is not a byte-for-byte translation of every VBA statement. The Excel-specific UI, PivotTable, named-range, worksheet-formatting, and menu code are intentionally omitted. The port targets the regression/M&V engine.

The largest deliberate numerical change is the removal of Excel Solver. The original VBA first searched change points and then allowed Solver to tweak change points, intercepts, and slopes while constraining net determination bias. NumPy least squares already forces the residual mean to approximately zero when an intercept is present, so this implementation fits the linear coefficients exactly for every candidate change point and refines the change points deterministically.

For additional source-to-Python mapping details, see `docs/ECAM_V4_PORT_NOTES.md`.
