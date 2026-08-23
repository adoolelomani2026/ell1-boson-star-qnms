"""Continue and locally count the long-lived axial pole across stable stars."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution, scan_sequence
from nonradial.axial_ekg import interior_mode_profile, matching_mode_coefficients
from nonradial.axial_spectrum import count_modes_adaptive, refine_analytic_root
from radial.coefficients import RadialBackground


AMPLITUDES = (0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095)


def seed_frequency(amplitude: float) -> complex:
    """Smooth continuation seed; no stored branch result is loaded."""

    delta = amplitude - 0.08
    return complex(
        0.049397730785 + 0.3941 * delta - 0.96 * delta**2,
        -5.96519327e-7 - 7.22e-6 * delta,
    )


def main() -> None:
    solutions = scan_sequence(
        1,
        AMPLITUDES,
        r_max=80.0,
        points=1800,
        tolerance=2e-9,
        max_nodes=120000,
    )
    rows: list[dict[str, object]] = []
    profiles: list[np.ndarray] = []
    profile_radii = np.linspace(0.05, 14.0, 500)
    for solution in solutions:
        tag = f"{int(round(solution.a0 * 1000)):03d}"
        solution.save(ROOT / "reports" / "axial" / f"background_ell1_a{tag}_axial_branch.npz")
        background = RadialBackground(solution, geometry_representation="hermite")
        roots = []
        setups = ((14.0, 35.0), (12.0, 30.0))
        seed = seed_frequency(solution.a0)
        for r_match, r_end in setups:
            refinement = refine_analytic_root(
                seed,
                background,
                matching_options={
                    "r_match": r_match,
                    "r_end": r_end,
                    "r_far": 600.0,
                    "asymptotic_order": 3,
                    "rtol": 3e-8,
                    "atol": 3e-10,
                },
                maximum_iterations=8,
            )
            if not refinement.converged:
                raise RuntimeError(f"root refinement failed at a1_0={solution.a0}")
            roots.append(refinement.pole)
            seed = refinement.pole
        sigma = sum(roots) / len(roots)
        contour = count_modes_adaptive(
            background,
            real_bounds=(sigma.real - 5e-5, sigma.real + 5e-5),
            imaginary_bounds=(-1.6e-6, 1.0e-7),
            initial_points_per_edge=6,
            maximum_phase_step=np.pi / 3.0,
            matching_options={
                "r_match": 14.0,
                "r_end": 35.0,
                "r_far": 300.0,
                "asymptotic_order": 3,
                "rtol": 3e-7,
                "atol": 3e-9,
            },
        )
        if contour["winding_number"] != 1 or not contour["phase_resolution_pass"]:
            raise RuntimeError(f"count failed at a1_0={solution.a0}: {contour}")
        coefficients, _, singular = matching_mode_coefficients(
            sigma,
            background,
            r_match=14.0,
            r_end=35.0,
            r_far=600.0,
            asymptotic_order=3,
            rtol=3e-8,
            atol=3e-10,
        )
        profile = interior_mode_profile(
            profile_radii,
            sigma,
            background,
            coefficients,
            rtol=3e-9,
            atol=3e-11,
        )
        profile /= np.linalg.norm(profile)
        profiles.append(profile)
        rows.append(
            {
                "a1_0": solution.a0,
                "omega": solution.omega,
                "adm_mass": solution.adm_mass,
                "compactness_R99": solution.compactness99,
                "sigma_real": sigma.real,
                "sigma_imag": sigma.imag,
                "quality_factor": sigma.real / (-2.0 * sigma.imag),
                "first_scalar_threshold": 1.0 - solution.omega,
                "below_first_scalar_threshold": sigma.real < 1.0 - solution.omega,
                "two_domain_spread_real": abs(roots[0].real - roots[1].real),
                "two_domain_spread_imag": abs(roots[0].imag - roots[1].imag),
                "matching_minimum_singular_value": singular,
                "local_count": contour,
            }
        )
        print(solution.a0, sigma, contour["final_contour_points"])
    for index, row in enumerate(rows):
        if index == 0:
            row["overlap_with_previous"] = None
        else:
            overlap = abs(np.vdot(profiles[index - 1], profiles[index]))
            row["overlap_with_previous"] = float(overlap)
    result = {
        "calculation": "seven-background continuation of the neutral ell=1 J=L=2 axial pole",
        "method": "two-domain holomorphic refinement, local exterior-algebra count, and profile-overlap tracking",
        "amplitudes": list(AMPLITUDES),
        "rows": rows,
        "acceptance": {
            "all_local_windings_one": all(row["local_count"]["winding_number"] == 1 for row in rows),
            "all_phase_resolved": all(row["local_count"]["phase_resolution_pass"] for row in rows),
            "all_below_first_scalar_threshold": all(row["below_first_scalar_threshold"] for row in rows),
        },
    }
    (ROOT / "reports" / "axial" / "axial_qnm_branch.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
