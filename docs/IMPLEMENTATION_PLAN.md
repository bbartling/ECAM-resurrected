# ECAM M&V implementation plan

Source assessed: `/home/ben/Documents/ECAM/ecam_v6_vba.txt`, extracted from
`ECAM_v6r6_2023-01-06.xlam` (January 6, 2023). The supplied text was empty;
the v6 workbook was re-extracted with oletools. The old v4 text was not used.

## Scope and architecture

Build a Python library first, with a thin CLI and optional static plots. Preserve
the energy M&V workflow rather than Excel's workbook, named-range, pivot-table,
Solver and UI architecture. Hourly energy records and billing periods are the
two public input forms. Inputs use explicit units, timestamp conventions,
period boundaries, and durations. Never infer kW versus kWh from values.

1. `data.py`: validate timestamps, finite numeric values and coverage; integrate
   fixed-duration average power or sum interval energy into hourly energy;
   calendar/day types and occupancy schedules; duration-normalized bills.
2. `models.py`, `metrics.py`: continuous change-point regression, numerical
   diagnostics, model selection, versioned model serialization.
3. `analysis.py`: group-specific baselines; reporting-condition adjusted
   baseline, avoided energy, cumulative savings, nonroutine adjustments,
   baseline/post normalization under supplied common weather, uncertainty.
4. `cli.py`, `plots.py`: CSV-in, JSON/CSV-out, residual/scatter/history plots.
5. `tests/`: analytical expected values, independently derived formulas, bad
   inputs, data semantics, billing bias, model recovery, serialization, CLI.
6. `pyproject.toml`, CI: src layout, editable pip install, optional uv lock/sync,
   wheel/sdist creation, isolated wheel install and metadata verification.

## Legacy feature map (line numbers in the extracted v6 text)

| VBA feature | Source | Python treatment |
|---|---|---|
| Select meter/billing data | mod1DataSelect 14106; modResample5Monthly 21246 | Explicit CSV/DataFrame schemas |
| Rate vs use | UseOrRate 4717; UserFormUseOrRate 34031 | Explicit energy/power conversion |
| Monthly fit and total-bill bias | MVcontrolMonthly 4853; MVnoMonthlyBias 4942; weighted Solver 8743 | Fit energy/day; equality constraint on total predicted energy |
| Daytypes, occupancy, holiday filters | mod6Schedules 13020; modrMandV0Daytyping 27722 | Calendar fields, explicit schedules/holiday dates, grouped baselines |
| Models | modrMandVmodels 7541 | 1p, 2p, 3pH, 3pC, 4p, 5p, 6p, 5pH, 5pC, 3pHzero, 5pHzero |
| Search and Solver | modrMandVchgPtSearch 9309; modrMandVsolver 8719 | Deterministic grid initialization plus bounded refinement |
| Regression statistics | FormulasAllData2 18421 | Residual = observed - modeled; n-p RMSE; CVRMSE; NMBE; net bias; R²; signed lag-1 correlation |
| Savings and normalization | modrMandVsavings 19355; NormalizeAnnual 20200 vicinity | Reporting savings and supplied-common-weather normalized savings |
| Uncertainty | StdErrorCalcsAll 19694; FormulasAllData2 18421 | Conditional regression covariance, shared coefficient error, prediction noise, stated AR(1) approximation |
| Residual and savings charts | mod9lSpecialCharts 16529; Outliers 12481 | Optional plots and explicit residual flags; no automatic removal |
| Ongoing tracking | modSEMmonthly 21623; modSEMinterval 21865 | Repeat reporting against saved model, cumulative savings |

Excluded: equipment point definitions, chiller/air-handler/zone/economizer
diagnostics, PNNL re-tuning charts, equipment tonnage and HVAC efficiency,
benchmarking, Excel UI manipulation, XML workbook interoperability and weather
downloads. User-supplied typical-year weather remains in scope for M&V.

## Numerical decisions

For fixed breakpoints, use linear least squares on continuous hinge bases.
One/two change points count as fitted parameters in n-p diagnostics. Zero-base
variants use actual independent parameter counts (2 and 4), unlike ECAM's
nominal 3/5 labels. Require coverage on each segment and reject rank-deficient
fits. Compare candidates with AICc; expose all diagnostics and rejection reasons.
Training fit metrics are not a substitute for holdout assessment.

Billing-period targets are energy / days. Apply a linear equality constraint
`sum(days * predicted_rate) = sum(energy)` when fitting bills, as ECAM does.
This constraint is distinct from choosing duration-weighted least squares.
Hourly targets are energy / hours. Convert predictions back to record energy
before reporting sums. Do not silently impute or omit records.

For hourly residual correlation, preserve timestamp order. ECAM uses
`sqrt(RSQ(...))`, which loses negative correlation; retain signed lag-1 rho.
Monthly uncertainty assumes independent billing residuals by default, as ECAM
does. For hourly data, disclose the optional conservative AR(1) variance
inflation. Report uncertainty as conditional on fitted breakpoints; a full
change-point/bootstrap validation remains a future milestone.

Savings = adjusted baseline + user-supplied nonroutine adjustment - actual.
Adjusted baseline and actual must cover the same records. Aggregate prediction
variance must include shared coefficient covariance, not sum individual
interval half-widths. No automatic claims of IPMVP/ASHRAE compliance.

## Review and release gate

Luna implements the numerical engine and its tests. The parent agent implements
records, M&V orchestration, CLI and packaging, independently reviews Luna's
numerics, redirects issues and adds regression tests. Verify pip editable
installation and uv workflow; run unit/integration tests and lint; build and
check both archives; install the wheel into a fresh environment outside the
source tree. Ship as alpha until comparisons with Excel-generated v6 reference
outputs are available. No workbook results were supplied to establish parity.

Package name `ecam-mv` is provisional; verify its availability before publishing.
No external repository or PyPI publication occurs in this build.
