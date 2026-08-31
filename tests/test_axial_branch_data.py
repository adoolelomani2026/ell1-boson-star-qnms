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


def test_constraint_monitor_has_resolved_step_scan_and_richardson_production():
    record = load("axial_constraint_checkpoint.json")
    assert record["production_derivative_method"] == "richardson"
    assert record["production_derivative_step_multiplier"] == 112.0
    fourth_order = record["derivative_step_rows"]
    richardson = record["richardson_step_rows"]
    assert len(fourth_order) == 17
    assert len(richardson) == 7
    assert min(row["relative_l2"] for row in fourth_order) < 7e-9
    assert min(row["relative_linf"] for row in fourth_order) < 9e-8
    assert min(row["relative_linf"] for row in richardson) < 1.3e-7
    assert max(row["relative_linf"] for row in record["resolution_rows"]) < 3e-7


def test_branch_powerlaw_fit_is_reproducible_and_narrowly_scoped():
    fit = load("branch_powerlaw_fit.json")
    assert fit["sample_size"] == 7
    assert np.isclose(fit["p"], 4.957702714258897, atol=1e-12)
    assert np.isclose(fit["A"], 0.04279825996339585, rtol=1e-12)
    assert fit["maximum_absolute_fractional_residual"] < 0.0024
    assert "not a demonstrated weak-field" in fit["warning"]


def test_center_start_scan_does_not_control_the_tiny_damping():
    summary = load("axial_numerical_robustness.json")["center_start_summary"]
    assert summary["sample_count"] == 9
    assert summary["all_converged"]
    assert summary["sigma_real_span"] < 1e-11
    assert summary["sigma_imag_span"] < 5e-15
    assert summary["fractional_damping_span"] < 5e-9


def test_ode_tolerance_scan_is_stable_over_factor_twenty():
    summary = load("axial_numerical_robustness.json")["ode_tolerance_summary"]
    assert summary["sample_count"] == 5
    assert summary["all_converged"]
    assert summary["sigma_real_span"] < 5e-11
    assert summary["sigma_imag_span"] < 5e-15


def test_matching_determinants_are_locally_holomorphic_at_resolved_steps():
    summary = load("axial_numerical_robustness.json")["holomorphy_summary"]
    assert summary["resolved_step_minimum"] == 5e-7
    assert summary["raw_maximum_resolved_mismatch"] < 2e-7
    assert summary["exterior_algebra_maximum_resolved_mismatch"] < 1.1e-7


def test_quadtree_pilot_assigns_the_local_axial_pole():
    result = load("axial_quadtree_census_pilot.json")
    assert result["complete_inside_declared_cut_free_window"]
    assert result["counted_zeros"] == 1
    assert len(result["assigned_poles"]) == 1
    assert result["leaves"][0]["maximum_phase_increment"] < np.pi / 4
    assert result["leaves"][0]["relative_residual"] < 1e-8


def test_ordinary_ell0_background_matches_published_turning_point():
    result = json.loads(
        (ROOT / "reports" / "ordinary_ell0" / "background_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(result["acceptance"].values())
    assert result["absolute_errors"]["maximum_mass"] < 5e-4
    assert result["absolute_errors"]["critical_phi_c"] < 5e-4
