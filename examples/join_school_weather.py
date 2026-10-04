"""Standalone convenience script; the same workflow is available as ecam-mv join-weather."""

import argparse
from pathlib import Path

from ecam_mv.workflows import prepare_weather_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meter_directory", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_weather_dataset(
        [args.meter_directory],
        args.out_dir,
        location="Lake Geneva",
        region="Wisconsin",
        country="US",
        timezone="America/Chicago",
        timestamp="Date",
        usage="kW",
        kind="power",
        interval_minutes=15,
        timestamp_position="end",
        dst="infer",
    )
    print(f"Validated meter/weather dataset saved to {args.out_dir}")
    print(report["meter"])


if __name__ == "__main__":
    main()
