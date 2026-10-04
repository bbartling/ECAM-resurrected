# ecam-mv

Python measurement and verification for hourly meter records and monthly/utility
billing periods, inspired by the M&V workflow in ECAM v6r6 (January 6, 2023).
An independent Python implementation. No Excel, HVAC diagnostics or PNNL
re-tuning features are required or included.

This is an alpha development package, not yet published to PyPI. See
`docs/IMPLEMENTATION_PLAN.md` for the source assessment and implementation scope.
The numerical tests establish analytical behavior; Excel reference-output
equivalence has not yet been established.

## Install locally

```bash
git clone https://github.com/bbartling/ECAM-resurrected.git
cd ECAM-resurrected
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

Or with uv:

```bash
uv sync --extra dev
uv run pytest
uv build --no-sources
```

## Use the Python API

```python
import pandas as pd
from ecam_mv import prepare_hourly, calendar_features, fit_baseline, avoided_energy
from ecam_mv.plots import report_plots

training = calendar_features(prepare_hourly(pd.read_csv("baseline.csv")))
reporting = calendar_features(prepare_hourly(pd.read_csv("reporting.csv")))
baseline = fit_baseline(training, model="3pC", group_by="daytype")
baseline.save("baseline_model.json")
print(baseline.metadata["fit_validation"])
result = avoided_energy(baseline, reporting, confidence=0.90, uncertainty_method="ecam")
result.save("savings.json", "savings.csv")
report_plots(baseline, reporting, result=result, directory="charts")
```

Hourly CSV: `timestamp,energy,temperature`. Energy is kWh per interval by
default; temperature is Celsius. Explicitly change units if needed. Power data
uses `kind="power"`, a declared interval and start/end convention:

```python
hourly = prepare_hourly(raw, timestamp="Date", usage="kW", kind="power",
                        interval_minutes=15, timestamp_position="end")
```

Here 4 kW over four 15-minute intervals is 4 kWh for the hour, not 16 kWh.
No data gaps, duplicate timestamps or ambiguous DST times are silently repaired.
Use explicit UTC offsets and a local timezone for DST-aware calendars.

Billing CSV: `start,end,energy,temperature`; end is exclusive. Temperature must
be the mean for the actual bill period. `prepare_billing` records calendar days,
including leap years; `fit_baseline` fits energy per day with zero total-bill
bias, and reporting predictions use the actual reporting bill lengths.

```python
from ecam_mv import prepare_billing
bills = prepare_billing(pd.read_csv("baseline_bills.csv"))
baseline = fit_baseline(bills, model="3pH")
```

Models: `1p`, `2p`, `3pH`, `3pC`, `4p`, `5p`, `6p`, `5pH`, `5pC`,
`3pHzero`, `5pHzero`, or `auto` (AICc selection with candidate scores/rejections).
Slopes are unconstrained; review their physical plausibility and residuals.
The coefficient meanings and deviations from Excel are in `docs/ECAM_PARITY.md`.

## Command line

```bash
ecam-mv fit baseline.csv --model 3pC --out baseline.json --plots baseline_charts
ecam-mv savings reporting.csv --baseline baseline.json --out savings.json \
  --out-csv savings.csv --uncertainty-method ecam --plots savings_charts
ecam-mv fit baseline_bills.csv --frequency billing --model 3pH --out monthly.json
ecam-mv normalize typical_year.csv --baseline baseline.json --post post.json \
  --out normalized.json --out-csv normalized.csv
```

Each command accepts column mappings, units, holidays, meter intervals and
timestamp conventions; see `ecam-mv fit --help`. `prepare` writes CSV plus a
metadata sidecar. API baselines consume prepared DataFrames with metadata;
CLI analysis commands prepare the original input CSV themselves, or read a
previously prepared CSV and its metadata sidecar with `--prepared`.

## Concatenate meters and join historical weather

```bash
ecam-mv join-weather /path/to/original_meter_csvs \
  --location 'Lake Geneva' --region Wisconsin --country US \
  --timezone America/Chicago --timestamp Date --usage kW --kind power \
  --interval-minutes 15 --timestamp-position end --dst infer \
  --out-dir work/school_weather
# Equivalent standalone script:
python examples/join_school_weather.py /path/to/original_meter_csvs --out-dir work/school_weather
```

This writes concatenated `intervals.csv`, processed `hourly.csv`, `weather.csv`,
matching metadata sidecars, and `audit.json`. Inputs are not changed. Existing
outputs are never overwritten; choose a fresh destination for another run.
Coordinates can replace `--location` using `--latitude`, `--longitude`, and an
explicit meter `--timezone`. Place-name matches must be unambiguous: Lake Geneva,
Wisconsin must not silently select Lake Geneva, Florida.

Dry bulb (`temperature_2m`, Celsius) is the default independent variable.
Humidity, dew point, precipitation, pressure, wind, cloud cover, and solar
radiation remain as reference columns. Weather comes from the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
using a consistent ERA5 model and UTC epoch timestamps. It is gridded reanalysis,
not a measured weather station at the school. Meter timestamps are localized
before sorting or shifting interval-end labels; joins use interval-start UTC
floor-hours and cannot multiply meter rows. Temperature is an instantaneous
hourly value. Precipitation and radiation describe the preceding hour; repeated
quarter-hour reference values must **not** be summed as interval totals.

The school records have real gaps and one partial final hour. These remain
visible as missing energy/zero duration or partial coverage, not manufactured
zeros. Strict modeling rejects incomplete hours. Explicit exclusion produces
an audit of retained/excluded records and known excluded energy:

```bash
ecam-mv fit work/school_weather/hourly.csv --prepared --exclude-incomplete \
  --model 3pC --out work/school_baseline.json --plots work/school_charts
# Select another numeric predictor (one independent variable, not multivariate):
ecam-mv fit work/school_weather/hourly.csv --prepared --exclude-incomplete \
  --predictor relative_humidity_2m --predictor-unit '%' --model 2p \
  --out work/humidity_model.json --plots work/humidity_charts
```

Null reference-only columns do not invalidate a dry-bulb model. Null, infinite,
or nonnumeric selected inputs do. Do not choose a predictor merely for a better
fit: review its physical justification, units, leakage, and residuals. Baselines
record predictor identity and reject mismatched reporting inputs. For raw custom
CSV columns, declare `--predictor-unit`; for weather columns, units can be read
from the sidecar. Python usage:

```python
from ecam_mv import model_records, fit_baseline
from ecam_mv.workflows import load_records
hourly = load_records("work/school_weather/hourly.csv")
clean = model_records(hourly, missing="drop")  # Explicit, audited exclusions
model = fit_baseline(clean, model="3pC")
humidity = model_records(hourly, predictor="relative_humidity_2m",
                         predictor_unit="%", missing="drop")
```

Intervention dates and an M&V plan are still needed for actual savings; a fitted
whole-history model alone does not establish an efficiency intervention.

## Practice data

```python
from ecam_mv import practice_dataset
practice = practice_dataset()  # 8,760 synthetic hourly records, no network needed
```

```bash
python examples/practice.py --out work/practice
ecam-mv fit work/practice/practice_hourly.csv --prepared --model 5p \
  --out work/practice_model.json --plots work/practice_charts
```

The deterministic practice generator is included in the installed package.
`examples/data/practice_hourly.csv` is the generated CSV for repository users;
it is explicitly synthetic, contains a known 5p model, and is not real school
savings evidence.

The repository also includes **one processed school practice CSV**,
[`datasets/school_hourly_weather.csv`](datasets/school_hourly_weather.csv), with
its metadata sidecar and audit. It was shared with the user's redistribution
authorization; the raw exports remain in a recoverable local archive and are
not committed. See [`datasets/README.md`](datasets/README.md) for source,
attribution, timestamp assumptions and known coverage gaps.

```bash
mkdir -p work
ecam-mv fit datasets/school_hourly_weather.csv --prepared --exclude-incomplete \
  --model 3pC --group-by occupied --out work/school_baseline.json --plots work/school_charts
```

Real school data is repository-only: it is not included in the wheel or sdist.
The example dry-bulb/occupied model's CVRMSE was about 44%, failing the hourly
GL14 error screen. Treat it as practice, not a validated savings baseline.
Intervention dates and a defensible M&V plan are still required.

The older v4 prototype is preserved in [`legacy_v4/`](legacy_v4/), including its
original license, but it is not installed by this package. The modern import is
`ecam_mv` and command is `ecam-mv`. See [`docs/MIGRATION.md`](docs/MIGRATION.md).

## Guideline 14, uncertainty and plots

`guideline14_check` provides the explicit **Guideline 14-2014 calibration
screen**: hourly |NMBE| <= 10%, CVRMSE <= 30%; monthly |NMBE| <= 5%, CVRMSE <= 15%.
These screens are documented in the [ASHRAE Handbook](https://handbook.ashrae.org/Handbooks/F17/IP/F17_Ch19/f17_ch19_ip.aspx).
They are model-fit screens, not proof of savings precision or a full compliance
assessment. API metrics use fractions; plots display percentages. Bill metrics
are on daily-normalized usage, matching ECAM's M&V fitting basis.
`baseline.metadata["ecam_net_determination_bias"]` separately reports bias on
actual energy totals, including unequal-length bills.

The named ECAM ports preserve `nprime`, Student-t confidence, absolute residual
autocorrelation (zero for bills), empirical FSU correction, total model-estimate
plus prediction-noise error, per-segment STEYX prediction intervals and
coefficient statistics. Reports expose `ecam_uncertainty`, `ecam_ashrae_fsu` and
`precision` separately. ECAM's default precision target is `1-confidence`;
it is identified as an ECAM convention, not a universal GL14 requirement.

`uncertainty_method="ecam"` selects the legacy total-uncertainty formula.
`"conditional"` uses shared coefficient covariance and actual record durations;
`autocorrelation=True` applies an explicitly approximate conservative AR(1)
inflation. Conditional intervals hold the fitted breakpoints fixed. Normalized
savings compare baseline and post means on supplied common weather/schedules,
with separate model-mean uncertainty and independent-fit assumptions. Selecting
`uncertainty_method="ecam"` in `normalized_savings` preserves ECAM's quadrature
of the baseline and post total-prediction uncertainties, including noise.

Meter errors, nonroutine-adjustment errors and uncertain breakpoints are not
included. An alpha release cannot claim Excel v6 numerical equivalence without
reference outputs. See `docs/ECAM_PARITY.md` for the exact differences.

Optional Matplotlib figures include model scatter and both prediction bands,
residual/lag/histogram charts, measured/modeled histories, day-type load profiles,
calendar maps, cumulative savings intervals, and precision-planning curves.
Residual colors follow ECAM's green/white/yellow/red thresholds. The cumulative
band is pointwise, not a simultaneous confidence band over the entire history.

Run synthetic demonstrations (not real project savings):

```bash
uv run python examples/demo.py --out work/demo
```

## Develop and prepare a release

```bash
uv run pytest --cov=ecam_mv
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv build --no-sources
uv run twine check dist/*
uv run python examples/check_release.py dist/*
```

The same distribution works with ordinary pip; uv is optional. GitHub CI checks
Python 3.11–3.13, builds wheel and source archives, and installs the wheel outside
the checkout. The repository includes Apache-2.0 licensing and ECAM attribution.
The legacy prototype, add-ins, extracted VBA and real school meter records are
not bundled in package distributions.

Before publishing, confirm the final available PyPI name,
validate against known ECAM outputs, and configure a PyPI token or trusted
publisher. For a TestPyPI trial, build locally and use
`python -m twine upload --repository testpypi dist/*`. For production, use the
documented [uv publishing workflow](https://docs.astral.sh/uv/guides/package/).
Nothing has been uploaded by this development task.
