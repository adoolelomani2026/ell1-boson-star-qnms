"""Certify the isolated axial pole with complex scaling and stabilized planes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import (
    matching_evans_determinant,
    matching_raw_determinant,
    matching_singular_value,
)
from nonradial.axial_spectrum import count_modes_adaptive, refine_scaled_root
from nonradial.riemann_sheet import SidebandSheet
from radial.coefficients import RadialBackground


def refinement_record(result) -> dict[str, object]:
    return {
        "pole_real": result.pole.real,
        "pole_imag": result.pole.imag,
        "relative_residual": result.relative_residual,
        "converged": result.converged,
        "iterations": result.iterations,
        "evaluations": result.evaluations,
    }


def main() -> None:
    solution = BackgroundSolution.load(
        ROOT / "reports/axial/background_ell1_a008_axial_refined.npz"
    )
    background = RadialBackground(solution, geometry_representation="hermite")
    sheet = SidebandSheet.physical_lower_half_plane()
    seed = 0.0493977304 - 5.96519e-7j
    rows = []
    for r_match in (12.0, 14.0):
        for r_far in (300.0, 400.0):
            options = {
                "sheet": sheet,
                "r_match": r_match,
                "r_end": 35.0,
                "r_far": r_far,
                "asymptotic_order": 3,
                "exterior_method": "complex_scaled",
                "rtol": 2.0e-8,
                "atol": 2.0e-10,
            }
            evans = refine_scaled_root(
                seed,
                background,
                matching_options=options,
                determinant=matching_evans_determinant,
                derivative_step=2.0e-8,
                step_tolerance=2.0e-13,
                residual_tolerance=1.0e-5,
                maximum_iterations=10,
            )
            raw = refine_scaled_root(
                evans.pole,
                background,
                matching_options=options,
                determinant=matching_raw_determinant,
                derivative_step=2.0e-8,
                step_tolerance=2.0e-13,
                residual_tolerance=1.0e-5,
                maximum_iterations=10,
            )
            singular_value = matching_singular_value(
                evans.pole, background, **options
            )
            rows.append(
                {
                    "r_match": r_match,
                    "r_end": 35.0,
                    "ray_endpoint_control": r_far,
                    "evans_refinement": refinement_record(evans),
                    "direct_determinant_refinement": refinement_record(raw),
                    "cross_determinant_pole_difference": abs(raw.pole - evans.pole),
                    "orthonormal_subspace_singular_value": singular_value,
                }
            )
            seed = evans.pole

    poles = np.asarray(
        [
            complex(
                row["evans_refinement"]["pole_real"],
                row["evans_refinement"]["pole_imag"],
            )
            for row in rows
        ]
    )
    baseline_options = {
        "sheet": sheet,
        "r_match": 14.0,
        "r_end": 35.0,
        "r_far": 300.0,
        "asymptotic_order": 3,
        "exterior_method": "complex_scaled",
        "rtol": 2.0e-8,
        "atol": 2.0e-10,
    }
    local_count = count_modes_adaptive(
        background,
        real_bounds=(0.0493, 0.0495),
        imaginary_bounds=(-1.5e-6, -1.0e-7),
        initial_points_per_edge=8,
        maximum_phase_step=np.pi / 16.0,
        maximum_refinements=10,
        minimum_uniform_refinements=2,
        required_stable_refinements=2,
        matching_options=baseline_options,
        determinant=matching_evans_determinant,
    )
    acceptance = {
        "all_refinements_converged": all(
            row["evans_refinement"]["converged"]
            and row["direct_determinant_refinement"]["converged"]
            for row in rows
        ),
        "orthonormal_rank_loss_confirmed": max(
            row["orthonormal_subspace_singular_value"] for row in rows
        )
        < 1.0e-7,
        "matching_and_ray_controls_stable": bool(
            np.ptp(poles.real) < 1.0e-7 and np.ptp(poles.imag) < 1.0e-8
        ),
        "cross_determinant_agreement": max(
            row["cross_determinant_pole_difference"] for row in rows
        )
        < 1.0e-7,
        "local_count_is_one": local_count["winding_number"] == 1,
        "local_contour_phase_resolved": local_count["phase_resolution_pass"],
        "local_count_stable": local_count["count_stability_pass"],
    }
    report = {
        "calculation": "v0.7 complex-scaled isolated-pole checkpoint",
        "background": {
            "a1_0": solution.a0,
            "omega": solution.omega,
            "adm_mass": solution.adm_mass,
        },
        "sheet": {
            "name": sheet.name,
            "signs": [sheet.plus_sign, sheet.minus_sign],
            "branch_points": sheet.branch_points(solution.omega),
        },
        "normalization_note": (
            "Positive real segment scalings preserve Evans phase and zeros; "
            "root refinement solves real and imaginary equations without a "
            "holomorphy assumption."
        ),
        "configuration_rows": rows,
        "evans_pole_spread": {
            "real": float(np.ptp(poles.real)),
            "imaginary": float(np.ptp(poles.imag)),
        },
        "local_contour": local_count,
        "acceptance": acceptance,
        "checkpoint_pass": all(acceptance.values()),
    }
    target = ROOT / "reports/axial/axial_complex_scaled_checkpoint_v07.json"
    target.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
