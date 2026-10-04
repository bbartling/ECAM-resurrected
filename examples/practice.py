"""Generate reproducible practice CSV and ECAM-style diagnostic figures."""

import argparse
from pathlib import Path

from ecam_mv import fit_baseline
from ecam_mv.datasets import practice_dataset
from ecam_mv.plots import report_plots
from ecam_mv.workflows import save_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("work/practice"))
    args = parser.parse_args()
    records = practice_dataset()
    save_records(records, args.out / "practice_hourly.csv")
    # A short sample keeps the demonstration's breakpoint search quick.
    training = records.iloc[::6].copy()
    baseline = fit_baseline(training, model="5p")
    baseline.save(args.out / "practice_baseline.json")
    report_plots(baseline, training, directory=args.out / "charts")
    print(f"Synthetic practice dataset and diagnostics: {args.out}")


if __name__ == "__main__":
    main()
