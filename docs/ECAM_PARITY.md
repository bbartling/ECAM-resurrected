# Source assessment, numerical traceability and release limits

## What was evaluated

The local v6 add-in identifies itself as January 6, 2023 in
`a_NOTICE_AboutECAM` (15714–15770). The empty `ecam_v6_vba.txt` was regenerated
from that workbook; its VBA/code extraction has 44,989 lines. This includes
forms/resources as well as executable modules. The original v4 extraction was
not used. Selected M&V modules carry William E. Koran's Apache-2.0 notice.

The legacy M&V engine is separable in purpose, but tightly coupled in code to
active worksheets, global variables, named ranges, pivot fields and Excel Solver.
Replacing that state with explicit arrays, records and serialized model objects
allows repeatable tests and use from scripts, notebooks and later interfaces.
The current build is the library/CLI foundation; it does not reproduce the Excel
menus, equipment analysis, HVAC re-tuning, PNNL chart library or weather downloader.

## Model formulas

Let `x` be supplied outdoor temperature and `c1 < c2` the fitted change points.
Each row lists the basis multiplied by coefficients in their saved order.

| ID | Linear basis | Independent parameter count, including fitted change points |
|---|---|---|
| 1p | 1 | 1 |
| 2p | 1, x | 2 |
| 3pH | 1, max(c1-x,0) | 3 |
| 3pC | 1, max(x-c1,0) | 3 |
| 4p | 1, min(x-c1,0), max(x-c1,0) | 4 |
| 5p | 1, max(c1-x,0), max(x-c2,0) | 5 |
| 6p | 1, min(x-c1,0), max(x-c1,0), max(x-c2,0) | 6 |
| 5pC | 1, max(x-c1,0), max(x-c2,0) | 5 |
| 5pH | 1, min(x-c1,0), clip(x-c1,0,c2-c1) | 5 |
| 3pHzero | max(c1-x,0) | 2 |
| 5pHzero | min(x-c1,0), clip(x-c1,0,c2-c1)-(c2-c1) | 4 |

The 6p coefficients represent slope changes, not three independent segment
intercepts. All models are continuous. A heating/cooling label does not impose
a physical slope-sign constraint. Negative modeled energy is not silently
clipped; review the selected model and weather range.

## M&V and GL14 traceability

Line numbers refer to the regenerated `ecam_v6_vba.txt` in the parent directory.

| ECAM source behavior | Python function/record | Status |
|---|---|---|
| Models and variants, 7541–8362 | fit_model | Shapes preserved; search and parameterization replaced |
| Change-point search/Solver, 8719–9694 | fit_model/select_model | Deterministic numerical search; segment/rank guards |
| Daily-normalized bills and total bias, 4853–5087 | prepare_billing/fit_baseline | Preserved with explicit dates and constrained least squares |
| Residual sign, 18380–18410 | regression_metrics | Observed minus modeled |
| RMSE, R², CVRMSE, net determination bias, 18421–18440 | regression_metrics | n-p RMSE and normalized net bias preserved; NMBE additionally exposed |
| Chronological rho, nprime, Student-t, 18448–18457 | validation ECAM functions | abs(rho), nprime=max(1,n*(1-rho)/(1+rho)), df=max(2,nprime-p); bills rho=0 |
| Segment STEYX/standard errors/t-statistics, 18658–18773 | ecam_regression_details | Formula port with safe small-segment fallback |
| Segment prediction half-intervals, 18778–18784 | ecam_prediction_interval | Student-t * STEYX * sqrt(1+1/n+(x-xmean)²/DEVSQ) |
| FSU month correction, 18565–18580 and 19515–19528 | ecam_fractional_savings_uncertainty | Hourly (-0.00024,0.03535,1.00286); bills (-0.00022,0.03306,0.94054) |
| Aggregate model estimate and prediction noise, 19694–19744 | ecam_total_uncertainty | Source expression port, including 365/12 days per bill |
| Avoided energy and fractional savings, 19812–19841 | avoided_energy | Same sign and matched records; adjustments explicit |
| Relative precision, 2110/2123 | precision_check | Half-width / abs(savings) |
| Expected savings precision, 27282–27354 | baseline_precision_plan | Exposed ECAM 1-confidence convention; safe bounded planning |
| GL14 calibration screens | guideline14_check | Explicit 2014 hourly/monthly thresholds; not found as one standalone check in the extracted VBA |
| Common-weather normalized baseline/post, 20200 onward | normalized_savings | Supplied records; ECAM baseline/post SEpTotal quadrature or conditional model means |
| Residual formatting, 12130–12180 | plots.model_diagnostics | ECAM color palette and standardized-residual thresholds |
| Model/residual/lag/histogram/history charts, 16529–17482 | plots/report_plots | Matplotlib equivalents |
| SEM tracking, 21623–22692 | saved Baseline + avoided_energy | Repeatable evaluations and cumulative savings; no Excel sheet mutation |

Fit screens use fractions, not whole percentage numbers. The hourly thresholds
are |NMBE| <= 0.10 and CVRMSE <= 0.30; monthly are 0.05 and 0.15. The named
2014 screen is documented by the [ASHRAE Handbook](https://handbook.ashrae.org/Handbooks/F17/IP/F17_Ch19/f17_ch19_ip.aspx).
It is not a full assessment against Guideline 14-2023 or every M&V requirement.

## Deliberate deviations and precision limitations

* Zero-base variants use their independent parameter counts (2 and 4), while
  ECAM uses nominal labels 3 and 5 in some degrees-of-freedom formulas.
* The new model engine preserves signed residual correlation; the separately
  named ECAM calculations preserve ECAM's absolute correlation convention.
* Billing model fitting uses actual days. The explicitly named ECAM uncertainty
  port retains its fixed 365/12 days per bill; the conditional method instead
  uses every actual duration in covariance and prediction-noise propagation.
* Bias constraints target the **observed** billed total, which is random. Its
  covariance must be propagated rather than treating the constraint as a known,
  error-free total. The review tests protect against artificially zero mean error.
* Conditional covariance treats optimized breakpoints as fixed. Its n-p residual
  variance still includes the fitted breakpoint count. A bootstrap or full
  nonlinear uncertainty assessment has not been implemented.
* The ECAM segment STEYX port fits an independent straight line inside each
  segment, reflecting the legacy diagnostic formula. This is different from
  using globally continuous model residuals to estimate covariance.
* Small segments, zero means, perfect fits and zero savings have explicit
  handling rather than spreadsheet divide-by-zero results.
* ECAM's BaselinePrecisionCheck has a hard-coded Solver target of 0.01 as well as
  a stated `1-confidence` requirement. The Python planning function consistently
  uses its explicit target and keeps the fitted baseline unchanged. It estimates
  reporting records required under fixed assumptions, not additional baseline
  records or guaranteed gains from collecting them.
* Calendar plots average repeated DST hours for display only. Actual energy
  records keep both hours. Cumulative intervals are pointwise, not simultaneous.
* Grouped model uncertainty assumes independent category coefficient estimates.
  Consecutive residuals within a category may be separated in clock time, so
  the optional AR(1) adjustment remains an approximation.

Uncertainty includes regression components only. It excludes meter accuracy,
nonroutine-adjustment uncertainty and errors in assumptions. This is also stated
in ECAM's `UncertaintyComment` at 20582–20590. Fits should be reviewed with
holdout data, temperature coverage and an actual M&V plan before project use.

## Review record and next parity milestone

Luna built models/metrics and tests. Parent review identified and redirected the
5pHzero flat-segment error, raw versus normalized net bias, fixed-observed-total
covariance, constant 1p handling, weighted-noise scaling and JSON validation.
The parent added record/CLI/plot/formula tests using independently specified
expected values rather than calculating expectations with the same implementation.

The next acceptance milestone is a small set of Excel-produced v6 reference
results: hourly 3p/4p/5p/6p, unequal-length monthly bills, baseline/post normalized
savings, ECAM FSU and total uncertainty at multiple confidence levels. Compare
coefficients, breakpoints, predictions, metrics and reporting totals within
stated tolerances. Synthetic tests validate the new implementation, but cannot
establish full Excel parity. Package is therefore alpha and not published.
