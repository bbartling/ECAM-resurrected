# Migration from the v4 prototype to ecam-mv

The repository root now contains **ecam-mv 0.1.0a2**, a Python library and
command-line tool for hourly and billing-period measurement and verification.
The earlier ECAM Resurrected v4 prototype has been moved intact to
[`legacy_v4/`](../legacy_v4/). Its source, tests, examples, launchers, and MIT
license remain there as a historical archive; they are not installed or
discovered by the modern project configuration.

## What changed

The current Python distribution is named `ecam-mv`, imports as `ecam_mv`, and
provides the `ecam-mv` command. The old `ecam_resurrected` import and
`ecam-resurrected` command are not part of the current installation. The v4
Tkinter desktop interface and Windows launchers are archived with the prototype;
the current project provides a Python API, CLI, and optional plotting library.

The root project targets Python 3.11 or newer. Install it from a checkout with
pip:

```bash
git clone https://github.com/bbartling/ECAM-resurrected.git
cd ECAM-resurrected
python -m venv .venv
# macOS or Linux
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
```

The optional `uv` workflow is:

```bash
uv sync --extra dev
```

Example modern imports and commands are in the root [README](../README.md).
The repository's `examples/` directory contains current API and CLI examples.
The old examples and tests are preserved under `legacy_v4/` and may require the
v4 prototype's own environment and dependencies; they do not describe the
modern package interface.

## Data and validation

The `datasets/` directory contains processed hourly school meter data joined to
Open-Meteo/Copernicus ERA5 weather and its provenance audit. No original meter
exports or quarter-hour interval file are included. The school data is for
practice and exploratory analysis, not a Guideline 14 qualified savings model.
See [the dataset notes](../datasets/README.md) for coverage, known gaps,
attribution, and interpretation limits. The generated practice dataset shipped
with the package is synthetic and is suitable for reproducible examples without
network access.

Original meter exports were preserved in a separate local recovery archive
before migration. That archive is not part of this repository or its public
release.
