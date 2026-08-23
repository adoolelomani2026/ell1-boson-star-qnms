import csv
import json
from pathlib import Path

import numpy as np


DATA = Path("reports")


def _rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_refined_stability_dataset_has_one_resolved_crossing():
    rows = _rows("radial/refined_radial_stability.csv")
    sigma2 = np.array([float(row["sigma2_ground"]) for row in rows])
    assert len(rows) == 13
    assert np.count_nonzero(sigma2[:-1] * sigma2[1:] <= 0.0) == 1
    assert all(int(row["node_count"]) == 0 for row in rows)
    assert max(float(row["unscaled_generalized_residual"]) for row in rows) < 1e-6
    summary = json.loads(
        (DATA / "radial" / "refined_radial_stability_summary.json").read_text(encoding="utf-8")
    )
    lower, upper = summary["bracket"]
    assert lower < summary["estimated_a0_at_sigma2_zero"] < upper


def test_gravitational_diagnostics_remain_horizonless():
    rows = _rows("background/gravitational_diagnostics.csv")
    assert len(rows) == 21
    assert max(float(row["max_two_m_over_r"]) for row in rows) < 1.0
    assert min(float(row["alpha_center"]) for row in rows) > 0.0
    assert all(float(row["exterior_kretschmann_99"]) > 0.0 for row in rows)


def test_extension_scaling_dataset_spans_requested_mass_range():
    rows = _rows("extensions/physics_extension_scaling.csv")
    masses = np.array([float(row["boson_mass_ev"]) for row in rows])
    assert len(rows) == 32
    assert masses[0] == 1e-22
    assert masses[-1] == 1e9
    assert np.all(np.diff(masses) > 0.0)


def test_solver_sensitivity_datasets_have_expected_families():
    background = _rows("background/background_sensitivity.csv")
    radial = _rows("radial/radial_sensitivity.csv")
    assert {row["experiment"] for row in background} == {
        "outer_domain",
        "mesh_seed",
        "collocation_tolerance",
    }
    assert sum(row["method"] == "nonlinear_global_bvp" for row in radial) == 4
    assert sum(row["method"] == "chebyshev_mode_0" for row in radial) == 10
    assert sum(row["method"] == "chebyshev_mode_1" for row in radial) == 10
