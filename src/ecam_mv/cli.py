"""CSV workflows for local, reproducible measurement and verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .analysis import Baseline, avoided_energy, fit_baseline, normalized_savings
from .data import calendar_features, model_records, prepare_billing, prepare_hourly
from .workflows import load_records, prepare_weather_dataset, save_records


def _record_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("csv", type=Path)
    parser.add_argument("--frequency", choices=["hourly", "billing"], default="hourly")
    parser.add_argument("--timestamp", default="timestamp")
    parser.add_argument("--usage", default="energy")
    parser.add_argument("--temperature", default="temperature")
    parser.add_argument("--predictor", help="Any numeric independent-variable column")
    parser.add_argument("--predictor-unit", help="Units of the selected column (e.g. C, %, units)")
    parser.add_argument(
        "--prepared", action="store_true", help="Read prepared CSV + metadata sidecar"
    )
    parser.add_argument(
        "--exclude-incomplete",
        action="store_true",
        help="Exclude invalid selected data or incomplete hours explicitly; audit exclusions",
    )
    parser.add_argument("--no-weather", action="store_true", help="Only appropriate for a 1p model")
    parser.add_argument("--start", default="start", help="Billing start column")
    parser.add_argument("--end", default="end", help="Billing exclusive end column")
    parser.add_argument("--kind", choices=["energy", "power"], default="energy")
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--timestamp-position", choices=["start", "end"], default="start")
    parser.add_argument("--timezone")
    parser.add_argument("--energy-unit", default="kWh")
    parser.add_argument("--temperature-unit", choices=["C", "F"], default="C")
    parser.add_argument("--allow-negative", action="store_true")
    parser.add_argument("--holiday", action="append", default=[], help="Holiday date; repeatable")


def _records(args: argparse.Namespace) -> pd.DataFrame:
    if args.no_weather and args.predictor:
        raise ValueError("--no-weather cannot be combined with --predictor")
    if args.prepared:
        records = load_records(args.csv)
        if args.timezone and args.timezone != records.attrs.get("timezone"):
            raise ValueError("Prepared-record timezone mismatch")
        if args.no_weather:
            records["temperature"] = 0.0
            records.attrs.update(predictor="none", predictor_unit="C", temperature_unit="C")
        records = model_records(
            records,
            predictor=args.predictor,
            predictor_unit=args.predictor_unit,
            missing="drop" if args.exclude_incomplete else "raise",
        )
        if records.attrs["record_filter"]["excluded_records"]:
            print(json.dumps(records.attrs["record_filter"], indent=2), file=sys.stderr)
        return calendar_features(records, holidays=args.holiday)
    frame = pd.read_csv(args.csv)
    if args.predictor and args.predictor != "temperature" and not args.predictor_unit:
        raise ValueError("Supply --predictor-unit when selecting a custom raw CSV column")
    common = {
        "usage": args.usage,
        "temperature": None if args.no_weather else (args.predictor or args.temperature),
        "energy_unit": args.energy_unit,
        "temperature_unit": args.predictor_unit or args.temperature_unit,
        "allow_negative": args.allow_negative,
    }
    if args.frequency == "billing":
        records = prepare_billing(frame, start=args.start, end=args.end, **common)
    else:
        records = prepare_hourly(
            frame,
            timestamp=args.timestamp,
            kind=args.kind,
            interval_minutes=args.interval_minutes,
            timestamp_position=args.timestamp_position,
            timezone=args.timezone,
            **common,
        )
    return calendar_features(model_records(records), holidays=args.holiday)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ecam-mv", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    weather = commands.add_parser(
        "join-weather", help="Concat meter CSVs and join historical Open-Meteo weather"
    )
    weather.add_argument("paths", nargs="+", type=Path, help="CSV files or a directory of CSVs")
    weather.add_argument("--out-dir", type=Path, required=True)
    weather.add_argument("--location")
    weather.add_argument("--region")
    weather.add_argument("--country", default="US")
    weather.add_argument("--latitude", type=float)
    weather.add_argument("--longitude", type=float)
    weather.add_argument("--timezone")
    weather.add_argument("--timestamp", default="Date")
    weather.add_argument("--usage", default="kW")
    weather.add_argument("--energy-unit", default="kWh")
    weather.add_argument("--kind", choices=["energy", "power"], default="power")
    weather.add_argument("--interval-minutes", type=int, default=15)
    weather.add_argument("--timestamp-position", choices=["start", "end"], default="end")
    weather.add_argument("--dst", choices=["raise", "infer"], default="raise")
    weather.add_argument("--allow-negative", action="store_true")
    weather.add_argument("--missing-weather", choices=["raise", "flag"], default="raise")
    prepare = commands.add_parser("prepare", help="Validate and aggregate raw records")
    _record_arguments(prepare)
    prepare.add_argument("--out", type=Path, required=True)
    fit = commands.add_parser("fit", help="Fit and save a baseline with GL14 fit screens")
    _record_arguments(fit)
    fit.add_argument("--model", default="3pC")
    fit.add_argument("--group-by", choices=["daytype", "occupied", "weekday"])
    fit.add_argument("--min-segment", type=int, default=3)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--plots", type=Path, help="Write ECAM-style baseline diagnostics")
    savings = commands.add_parser("savings", help="Evaluate reporting-period avoided energy")
    _record_arguments(savings)
    savings.add_argument("--baseline", type=Path, required=True)
    savings.add_argument("--out", type=Path, required=True, help="JSON report")
    savings.add_argument("--out-csv", type=Path)
    savings.add_argument("--confidence", type=float, default=0.9)
    savings.add_argument(
        "--adjustment", type=float, default=0.0, help="Energy adjustment per record"
    )
    savings.add_argument("--autocorrelation", action="store_true")
    savings.add_argument(
        "--uncertainty-method", choices=["conditional", "ecam"], default="conditional"
    )
    savings.add_argument("--plots", type=Path)
    normalize = commands.add_parser("normalize", help="Compare models on common weather/schedules")
    _record_arguments(normalize)
    normalize.add_argument("--baseline", type=Path, required=True)
    normalize.add_argument("--post", type=Path, required=True)
    normalize.add_argument("--confidence", type=float, default=0.9)
    normalize.add_argument(
        "--uncertainty-method", choices=["conditional", "ecam"], default="conditional"
    )
    normalize.add_argument("--out", type=Path, required=True)
    normalize.add_argument("--out-csv", type=Path)
    try:
        args = parser.parse_args(argv)
        if args.command == "join-weather":
            report = prepare_weather_dataset(
                args.paths,
                args.out_dir,
                location=args.location,
                region=args.region,
                country=args.country,
                latitude=args.latitude,
                longitude=args.longitude,
                timezone=args.timezone,
                timestamp=args.timestamp,
                usage=args.usage,
                energy_unit=args.energy_unit,
                kind=args.kind,
                interval_minutes=args.interval_minutes,
                timestamp_position=args.timestamp_position,
                dst=args.dst,
                allow_negative=args.allow_negative,
                missing_weather=args.missing_weather,
            )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        records = _records(args)
        if args.command == "prepare":
            save_records(records, args.out)
        elif args.command == "fit":
            baseline = fit_baseline(
                records, model=args.model, group_by=args.group_by, min_segment=args.min_segment
            )
            baseline.save(args.out)
            print(json.dumps(baseline.metadata["fit_validation"], indent=2))
            if args.plots:
                from .plots import report_plots

                report_plots(baseline, records, directory=args.plots)
        elif args.command == "savings":
            baseline = Baseline.load(args.baseline)
            result = avoided_energy(
                baseline,
                records,
                confidence=args.confidence,
                adjustment=args.adjustment,
                autocorrelation=args.autocorrelation,
                uncertainty_method=args.uncertainty_method,
            )
            result.save(args.out, args.out_csv)
            print(json.dumps(result.summary, indent=2, allow_nan=False))
            if args.plots:
                from .plots import report_plots

                report_plots(baseline, records, result=result, directory=args.plots)
        else:
            result = normalized_savings(
                Baseline.load(args.baseline),
                Baseline.load(args.post),
                records,
                confidence=args.confidence,
                uncertainty_method=args.uncertainty_method,
            )
            result.save(args.out, args.out_csv)
            print(json.dumps(result.summary, indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError) as exc:
        print(f"ecam-mv: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
