from pathlib import Path
import numpy as np
from ecam_resurrected.io import csv_headers, read_xy_csv


def test_csv_io(tmp_path: Path):
    p = tmp_path / "d.csv"
    p.write_text("temp,kwh\n50,100\n60,120\nbad,130\n70,140\n", encoding="utf-8")
    assert csv_headers(p) == ["temp", "kwh"]
    x, y = read_xy_csv(p, "temp", "kwh")
    assert np.allclose(x, [50, 60, 70])
    assert np.allclose(y, [100, 120, 140])
