# ECAM v4 port notes

This project was reconstructed from the VBA extracted from `ECAM_v4_2016-03-01.xlam`.
The important source modules in that VBA dump are:

- `modrMandVmodels.bas`: 2P, 3P heating, 3P cooling, 4P, 5P, and 6P model orchestration.
- `modrMandVformulas.bas`: regression equations, modeled values, residuals, RMSE, R², CV(RMSE), net determination bias, autocorrelation-adjusted point count, t-statistics, and prediction intervals.
- `modrMandVchgPtSearch.bas`: iterative change-point search.
- `modrMandVsolver.bas`: Excel Solver refinement.
- `modrMandVsavings.bas`: avoided-energy and normalized-savings calculations and projected-total regression uncertainty.
- `modrMandVuncertainty.bas`: confidence-level/fractional-savings UI hooks; the main numerical uncertainty formulas live in `modrMandVformulas.bas` and `modrMandVsavings.bas`.

## What is intentionally modernized

The original workbook mixed regression math with worksheet formulas, PivotTables, named ranges, forms, and Excel Solver. This port keeps the model families and key ECAM statistics but replaces worksheet/Solver machinery with deterministic NumPy least squares and grid-refined change-point searches.

The piecewise models are represented as continuous hinge regressions:

- 2P: intercept + slope*x
- 3P heating: flat high-temperature base plus low-temperature slope
- 3P cooling: flat low-temperature base plus high-temperature slope
- 4P: two slopes meeting at one change point
- 5P: low slope + flat middle + high slope with two change points
- 6P: three continuous slopes with two change points

ECAM counts change points as fitted parameters when calculating RMSE degrees of freedom; this port does the same.

## ECAM-style statistics retained

- RMSE = sqrt(SSE / (n - p))
- R² = 1 - SSE/SST
- CV(RMSE) = RMSE / mean(y)
- Net determination bias = sum(residuals) / sum(y)
- ECAM lag-1 residual autocorrelation magnitude = sqrt(RSQ(lagged residual pairs)) = abs(correlation)
- Effective points n' = max(1, n*(1-rho)/(1+rho))
- Critical t uses the two-sided confidence interval convention corresponding to Excel `TINV(1-ConfLvl, df)`
- Fractional savings uncertainty approximation retains ECAM's 1.26 factor and n/n' correction
- Avoided energy use = projected baseline energy - measured post energy
- Normalized savings = projected baseline under common conditions - projected post under common conditions
- Combined normalized-savings uncertainty uses root-sum-square of baseline/post regression uncertainty

## Important uncertainty limitation

The ECAM VBA explicitly warns that displayed savings uncertainty only includes regression uncertainty. It does not automatically account for measurement uncertainty, model-form error, omitted variables, or extrapolation outside the range represented by the baseline data. This port preserves that warning.
