import json

import numpy as np
import pandas as pd
import pytest

from ecam_mv.cli import main


def test_csv_fit_and_savings_cli(tmp_path):
    x = np.tile(np.arange(24), 3)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(x), freq="h"),
            "energy": 10 + 0.5 * x,
            "temperature": x,
        }
    )
    data, baseline, summary, results = [
        tmp_path / name for name in ("data.csv", "baseline.json", "summary.json", "savings.csv")
    ]
    frame.to_csv(data, index=False)
    assert main(["fit", str(data), "--model", "2p", "--out", str(baseline)]) == 0
    frame.energy -= 1
    frame.to_csv(data, index=False)
    assert (
        main(
            [
                "savings",
                str(data),
                "--baseline",
                str(baseline),
                "--out",
                str(summary),
                "--out-csv",
                str(results),
            ]
        )
        == 0
    )
    assert json.loads(summary.read_text())["summary"]["avoided_energy"] == pytest.approx(72)
    assert "cumulative_savings" in pd.read_csv(results)


def test_bad_input_cli_returns_useful_error(tmp_path, capsys):
    data = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["bad"], "energy": [1], "temperature": [1]}).to_csv(
        data, index=False
    )
    assert main(["fit", str(data), "--out", str(tmp_path / "out.json")]) == 2
    assert "ecam-mv:" in capsys.readouterr().err
