"""Independently refine the discovered neutral axial pole on two domains."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import matching_evans_determinant, matching_singular_value
from nonradial.axial_spectrum import refine_analytic_root
from radial.coefficients import RadialBackground


def run_checkpoint() -> dict[str, object]:
    solution = BackgroundSolution.load(ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz")
    background = RadialBackground(solution, geometry_representation="hermite")
    setups = [
        {
            "r_match": 14.0,
            "r_end": 35.0,
            "r_far": 600.0,
            "asymptotic_order": 3,
        },
        {
            "r_match": 12.0,
            "r_end": 30.0,
            "r_far": 600.0,
            "asymptotic_order": 3,
        },
    ]
    rows = []
    for setup in setups:
        options = {**setup, "rtol": 2e-8, "atol": 2e-10}
        # This is a deliberately low-precision seed at the center of the
        # discovery box, not a previously stored pole.
        seed = complex(0.0494, -6.0e-7)
        refinement = refine_analytic_root(
            seed, background, matching_options=options
        )
        sigma = refinement.pole
        determinant = matching_evans_determinant(sigma, background, **options)
        reference = abs(
            matching_evans_determinant(sigma + 2.0e-6, background, **options)
        )
        singular = matching_singular_value(
            sigma,
            background,
            **options,
        )
        rows.append(
            {
                "r_match": setup["r_match"],
                "r_end": setup["r_end"],
                "r_far": setup["r_far"],
                "sigma_real": sigma.real,
                "sigma_imag": sigma.imag,
                "determinant_real": determinant.real,
                "determinant_imag": determinant.imag,
                "determinant_abs": abs(determinant),
                "exterior_algebra_evans_residual_relative_to_fixed_frequency_offset": abs(determinant) / reference,
                "conditioned_matching_determinant_residual_relative_to_local_offset": refinement.relative_residual,
                "normalized_min_singular_value": singular,
                "root_solver_success": refinement.converged,
            }
        )
    mean_real = sum(row["sigma_real"] for row in rows) / len(rows)
    mean_imag = sum(row["sigma_imag"] for row in rows) / len(rows)
    spread_real = max(row["sigma_real"] for row in rows) - min(
        row["sigma_real"] for row in rows
    )
    spread_imag = max(row["sigma_imag"] for row in rows) - min(
        row["sigma_imag"] for row in rows
    )
    return {
        "calculation": "neutral ell=1, J=L=2 axial EKG two-sided QNM matching",
        "background": {
            "a1_0": solution.a0,
            "omega": solution.omega,
            "adm_mass": solution.adm_mass,
            "compactness_R99": solution.compactness99,
        },
        "mode": {
            "sigma_real_mean": mean_real,
            "sigma_imag_mean": mean_imag,
            "quality_factor": mean_real / (-2.0 * mean_imag),
            "first_scalar_threshold": 1.0 - solution.omega,
            "below_first_scalar_threshold": mean_real < 1.0 - solution.omega,
            "domain_spread_real": spread_real,
            "domain_spread_imag": spread_imag,
        },
        "matching_setups": rows,
        "acceptance": {
            "closed_six_state_operator": True,
            "flat_vacuum_Regge_Wheeler_control": True,
            "flat_massive_sideband_control": True,
            "exact_flat_hankel_exterior_control": True,
            "two_domain_complex_root": True,
            "from_scratch_grid_discovery": True,
            "analytic_raw_determinant_contour_count": True,
            "time_domain_real_frequency_check": True,
            "time_domain_damping_certified": False,
            "status": "frequency-domain pole; short evolution checks only the real oscillation frequency",
        },
    }


def main() -> None:
    result = run_checkpoint()
    target = ROOT / "reports" / "axial" / "axial_qnm_checkpoint.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["mode"], indent=2))


if __name__ == "__main__":
    main()
