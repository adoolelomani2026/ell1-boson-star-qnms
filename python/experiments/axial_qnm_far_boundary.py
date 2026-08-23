"""Certify asymptotic-order and far-boundary convergence of the axial pole."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_spectrum import refine_analytic_root
from radial.coefficients import RadialBackground


def main() -> None:
    solution = BackgroundSolution.load(ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz")
    background = RadialBackground(solution, geometry_representation="hermite")
    common = {"r_match": 14.0, "r_end": 35.0, "rtol": 2e-7, "atol": 2e-9}
    seed = complex(0.04939773, -5.96e-7)

    order_rows = []
    for order in (1, 2, 3):
        refinement = refine_analytic_root(
            seed,
            background,
            matching_options={**common, "r_far": 300.0, "asymptotic_order": order},
        )
        order_rows.append(
            {
                "asymptotic_order": order,
                "sigma_real": refinement.pole.real,
                "sigma_imag": refinement.pole.imag,
                "relative_residual": refinement.relative_residual,
                "converged": refinement.converged,
                "evaluations": refinement.evaluations,
            }
        )
        if order == 3:
            seed = refinement.pole

    radius_rows = []
    for r_far in (120.0, 200.0, 300.0, 600.0, 900.0):
        refinement = refine_analytic_root(
            seed,
            background,
            matching_options={**common, "r_far": r_far, "asymptotic_order": 3},
        )
        radius_rows.append(
            {
                "r_far": r_far,
                "sigma_real": refinement.pole.real,
                "sigma_imag": refinement.pole.imag,
                "relative_residual": refinement.relative_residual,
                "converged": refinement.converged,
                "evaluations": refinement.evaluations,
            }
        )
        seed = refinement.pole

    result = {
        "calculation": "curvature-corrected axial-QNM asymptotic convergence",
        "boundary_model": "Regge-Wheeler O(r^-3) master series and massive-scalar Coulomb O(r^-3) series",
        "matching": common,
        "asymptotic_order_rows": order_rows,
        "far_radius_rows": radius_rows,
        "order_2_to_3_shift": {
            "real": abs(order_rows[2]["sigma_real"] - order_rows[1]["sigma_real"]),
            "imaginary": abs(order_rows[2]["sigma_imag"] - order_rows[1]["sigma_imag"]),
        },
        "rfar_600_to_900_shift": {
            "real": abs(radius_rows[-1]["sigma_real"] - radius_rows[-2]["sigma_real"]),
            "imaginary": abs(radius_rows[-1]["sigma_imag"] - radius_rows[-2]["sigma_imag"]),
        },
    }
    target = ROOT / "reports" / "axial" / "axial_qnm_far_boundary.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
