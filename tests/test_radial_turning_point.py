import csv
from pathlib import Path

import pytest


def _rows_by_amplitude():
    path = Path(__file__).parents[1] / "data" / "radial_benchmarks.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return {float(row["a0"]): row for row in csv.DictReader(stream) if row["a0"] != "0.080"}


def test_radial_stability_crossing():
    rows = _rows_by_amplitude()
    assert float(rows[0.05]["sigma2"]) > 0.0
    assert abs(float(rows[0.10]["sigma2"])) < 5e-6
    assert float(rows[0.105]["sigma2"]) < 0.0
    assert float(rows[0.105]["sigma2"]) == pytest.approx(-7.11e-5, rel=0.04)
    assert float(rows[0.105]["center_c"]) == pytest.approx(-3.47e-2, rel=0.01)

