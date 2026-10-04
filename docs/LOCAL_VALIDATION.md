# Local verification — October 4, 2026

Canonical checkout: `/home/ben/Desktop/ECAM-resurrected`, branch `develop`.
Distribution: `ecam-mv 0.1.0a2`; import: `ecam_mv`; CLI: `ecam-mv`.
Local runtime: Python 3.12.3 on Linux.

## Verification scope

Final migrated-checkout results: **149 tests passed, 87% coverage**. Ruff lint
and format checks, `uv lock --check`, wheel/sdist builds, Twine metadata checks,
and release-content checks all passed. Ordinary pip installed the built wheel
in a fresh environment; imports, the CLI, practice data, school-data filtering,
and baseline save/load/prediction passed from `/tmp`, outside the checkout.
The modern installation does not expose the archived `ecam_resurrected` package.

Before migration, the reviewed v6 development package passed 142 unit/integration
tests with 87% coverage. Ruff lint/format checks, pip editable installation,
`uv sync --extra dev`, wheel/sdist builds, and Twine metadata checks passed.
An isolated installed wheel was tested outside the checkout, including imports,
synthetic practice data, processed-data loading, filtering and numerical prediction.

Migration adds four repository-only tests covering real dataset row counts,
energy totals, explicit exclusions, both DST folds/spring gaps, and portable
provenance. These tests intentionally skip when run from a package source archive
without the repository-only school dataset. The new checkout is independently
installed and verified before commit/push. A further three tests verify that
release archives reject repository-only data and the archived prototype. The
source archive uses exact, root-relative file selection; CI checks both built
archives. See [Hatch's build configuration](https://hatch.pypa.io/latest/config/build/)
for the difference between glob patterns and explicit selection.
Hosted CI covers Python 3.11–3.13; only local Python 3.12 has been verified at the
time of this note.

## Live school meter/weather integration

- Four original CSV exports contain 139,825 interval-average kW readings.
- Meter timezone: America/Chicago; interval-end assumption, 15 minutes.
- DST inference preserves the repeated fall hours; UTC duplicates are rejected.
- Concatenation and weather joins preserve rows and 2,753,981.6 kWh.
- The processed hourly CSV has 35,029 rows: 72 missing meter hours and one
  partial final hour. No meter energy is interpolated or manufactured.
- All dry-bulb and reference weather values matched Open-Meteo ERA5 for the
  requested span. Independent UTC lookups verified every interval/hourly match.
- Strict modeling rejects incomplete hours. Explicit exclusions retain 34,956
  complete hours and report 73 excluded rows and 8.7 kWh of known partial energy.
- An exploratory 3pC/occupied model had aggregate CVRMSE 43.95%, failing the
  hourly GL14 error screen. This is practice data, not validated savings evidence.

The real processed CSV, metadata and audit are in `datasets/`, with provider
authorization and Open-Meteo/Copernicus attribution. The raw exports were moved
to a recoverable local archive; they are not committed. Source checksums remain
in the audit. The wheel and sdist exclude real meter data, raw VBA and the old
v4 prototype. The old prototype and its original license remain in `legacy_v4/`.

## Scientific release boundary

The package is still alpha; Excel-produced ECAM v6 reference-output equivalence
has not been established. GL14 calibration screens are not full certification.
Conditional intervals hold fitted breakpoints fixed; ECAM formula ports preserve
documented legacy approximations. Meter and nonroutine-adjustment uncertainty
are not included. See `ECAM_PARITY.md` for details. No PyPI upload is part of this
migration/commit/push workflow.
