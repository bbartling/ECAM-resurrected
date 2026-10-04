from __future__ import annotations

import argparse
import json

from .io import read_xy_csv, write_fit_csv
from .models import fit_model, fit_best_model
from .report import fit_text, savings_text
from .savings import avoided_energy_savings


def _fit_from_args(args):
    x, y = read_xy_csv(args.csv, args.x, args.y)
    if args.model == "auto":
        return fit_best_model(x, y, criterion=args.criterion, grid_size=args.grid_size)
    return fit_model(x, y, args.model, grid_size=args.grid_size)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ecam-resurrected", description="ECAM Resurrected M&V regression tool")
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit a baseline temperature/energy model")
    fit.add_argument("csv")
    fit.add_argument("--x", required=True, help="temperature/independent variable column")
    fit.add_argument("--y", required=True, help="energy/dependent variable column")
    fit.add_argument("--model", default="auto", choices=["auto", "2p", "3p_heat", "3p_cool", "4p", "5p", "6p"])
    fit.add_argument("--criterion", default="aicc", choices=["aic", "aicc", "bic"])
    fit.add_argument("--grid-size", type=int, default=35)
    fit.add_argument("--export", help="optional CSV with measured/modeled/residual values")

    sv = sub.add_parser("savings", help="calculate avoided energy use against a post-period CSV")
    sv.add_argument("baseline_csv")
    sv.add_argument("post_csv")
    sv.add_argument("--x", required=True)
    sv.add_argument("--y", required=True)
    sv.add_argument("--model", default="auto", choices=["auto", "2p", "3p_heat", "3p_cool", "4p", "5p", "6p"])
    sv.add_argument("--confidence", type=float, default=0.80)
    sv.add_argument("--grid-size", type=int, default=35)

    args = parser.parse_args(argv)
    if args.command == "fit":
        result = _fit_from_args(args)
        print(fit_text(result))
        if args.export:
            write_fit_csv(args.export, result.x, result.y, result.yhat, result.residuals)
        return 0

    bx, by = read_xy_csv(args.baseline_csv, args.x, args.y)
    if args.model == "auto":
        model = fit_best_model(bx, by, grid_size=args.grid_size)
    else:
        model = fit_model(bx, by, args.model, grid_size=args.grid_size)
    px, py = read_xy_csv(args.post_csv, args.x, args.y)
    result = avoided_energy_savings(model, px, py, confidence=args.confidence)
    print(fit_text(model))
    print()
    print(savings_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
