import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "reports" / "axial" / name).read_text(encoding="utf-8"))


def test_seven_background_branch_is_counted_and_continuous():
    record = load("axial_qnm_branch.json")
    rows = record["rows"]
    assert len(rows) == 7
    assert record["acceptance"] == {
        "all_local_windings_one": True,
        "all_phase_resolved": True,
        "all_below_first_scalar_threshold": True,
    }
    assert all(row["local_count"]["winding_number"] == 1 for row in rows)
    assert all(row["local_count"]["maximum_phase_increment"] < np.pi / 3 for row in rows)
    assert all(row["two_domain_spread_real"] < 6e-12 for row in rows)
    assert all(row["two_domain_spread_imag"] < 1e-13 for row in rows)
    overlaps = [row["overlap_with_previous"] for row in rows[1:]]
    assert min(overlaps) > 0.999
    assert np.all(np.diff([row["sigma_real"] for row in rows]) > 0)
    assert np.all(np.diff([-row["sigma_imag"] for row in rows]) > 0)


def test_strict_and_expanded_central_counts_are_phase_resolved():
    record = load("axial_qnm_discovery.json")
    for key in ("strict_pi_over_4_contour_count", "expanded_local_contour_count"):
        count = record[key]
        assert count["winding_number"] == 1
        assert count["phase_resolution_pass"]
        assert count["maximum_phase_increment"] < np.pi / 4


def test_time_domain_matrix_uses_current_frequency_reference():
    convergence = load("axial_time_domain_convergence.json")
    checkpoint = load("axial_qnm_checkpoint.json")
    reference = convergence["frequency_domain_reference"]
    assert np.isclose(reference["sigma_real"], checkpoint["mode"]["sigma_real_mean"], atol=1e-15)
    assert np.isclose(reference["sigma_imag"], checkpoint["mode"]["sigma_imag_mean"], atol=1e-15)
    rows = convergence["runs"]
    finest = next(row for row in rows if row["points"] == 2500)
    coarsest = next(row for row in rows if row["points"] == 1500)
    assert finest["relative_frequency_error"] < coarsest["relative_frequency_error"]
    assert convergence["summary"]["maximum_relative_frequency_error"] < 1.1e-6
    assert not convergence["summary"]["damping_magnitude_measured_independently"]


def test_determinant_residual_names_are_unambiguous():
    discovery = load("axial_qnm_discovery.json")
    assert "conditioned_matching_determinant_residual_relative_to_local_offset" in discovery["refinement"]
    assert "relative_evans_determinant_residual" not in discovery["refinement"]
    checkpoint = load("axial_qnm_checkpoint.json")
    for row in checkpoint["matching_setups"]:
        assert "exterior_algebra_evans_residual_relative_to_fixed_frequency_offset" in row
        assert "conditioned_matching_determinant_residual_relative_to_local_offset" in row
