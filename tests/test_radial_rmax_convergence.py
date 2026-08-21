import csv
from pathlib import Path

import numpy as np


def test_radau_domain_plateau():
    path = Path(__file__).parents[1] / "data" / "radial_benchmarks.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    plateau = [
        row
        for row in rows
        if row["a0"] == "0.080" and row["method"] == "Radau" and float(row["r_max"]) >= 25
    ]
    values = np.array([float(row["sigma2"]) for row in plateau])
    assert len(values) == 3
    assert np.ptp(values) / np.mean(values) < 3e-4
    assert all(int(row["node_count"]) == 0 for row in plateau)

