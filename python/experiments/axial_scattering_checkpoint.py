"""Compute flux-normalized neutral axial scattering and conversion.

The raw exterior basis uses unit endpoint amplitudes, so its gravitational and
massive-scalar columns do not carry equal canonical flux.  For the two open
channels the unique positive weight ratio follows from the off-diagonal
reciprocity identity S^dagger W S=W.  The two diagonal identities are then
independent conservation tests rather than inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import scattering_matrix as solve_scattering_matrix
from radial.coefficients import RadialBackground


def complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def scattering_matrix(
    frequency: float,
    background: RadialBackground,
    *,
    r_match: float,
    r_end: float,
) -> tuple[np.ndarray, float]:
    return solve_scattering_matrix(
        frequency,
        background,
        r_match=r_match,
        r_end=r_end,
        r_far=600.0,
        asymptotic_order=3,
        rtol=2e-11,
        atol=2e-13,
    )


def flux_audit(matrix: np.ndarray) -> dict[str, object]:
    sgg, sgs = matrix[0]
    ssg, sss = matrix[1]
    ratio_complex = -np.conjugate(sgg) * sgs / (
        np.conjugate(ssg) * sss
    )
    ratio = float(ratio_complex.real)
    if ratio <= 0.0:
        raise RuntimeError("the inferred open-channel flux metric is not positive")
    gravity_reflection = float(abs(sgg) ** 2)
    gravity_conversion = float(ratio * abs(ssg) ** 2)
    scalar_conversion = float(abs(sgs) ** 2 / ratio)
    scalar_reflection = float(abs(sss) ** 2)
    gravity_balance = gravity_reflection + gravity_conversion
    scalar_balance = scalar_conversion + scalar_reflection
    normalized = np.array(
        (
            (sgg, sgs / np.sqrt(ratio)),
            (np.sqrt(ratio) * ssg, sss),
        ),
        dtype=complex,
    )
    unity_defect = np.linalg.norm(
        normalized.conj().T @ normalized - np.eye(2), ord=2
    )
    return {
        "raw_matrix": [[complex_pair(value) for value in row] for row in matrix],
        "scalar_to_gravity_flux_weight_ratio": ratio,
        "reciprocity_ratio_imaginary_fraction": float(
            abs(ratio_complex.imag) / ratio
        ),
        "gravity_incident": {
            "gravity_fraction": gravity_reflection,
            "scalar_fraction": gravity_conversion,
            "sum": gravity_balance,
        },
        "scalar_incident": {
            "gravity_fraction": scalar_conversion,
            "scalar_fraction": scalar_reflection,
            "sum": scalar_balance,
        },
        "unitarity_spectral_defect": float(unity_defect),
    }


def run_checkpoint() -> dict[str, object]:
    solution = BackgroundSolution.load(ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz")
    background = RadialBackground(solution, geometry_representation="hermite")
    rows = []
    for frequency in (0.16, 0.18, 0.20, 0.25, 0.30):
        matrix, residual = scattering_matrix(
            frequency, background, r_match=14.0, r_end=35.0
        )
        row = {"sigma": frequency, "linear_residual": residual}
        row.update(flux_audit(matrix))
        rows.append(row)

    domains = []
    for r_match, r_end in ((12.0, 30.0), (14.0, 35.0), (16.0, 40.0)):
        matrix, residual = scattering_matrix(
            0.20, background, r_match=r_match, r_end=r_end
        )
        audit = flux_audit(matrix)
        domains.append(
            {
                "r_match": r_match,
                "r_end": r_end,
                "linear_residual": residual,
                "gravity_to_scalar_fraction": audit["gravity_incident"]["scalar_fraction"],
                "scalar_to_gravity_fraction": audit["scalar_incident"]["gravity_fraction"],
                "gravity_balance": audit["gravity_incident"]["sum"],
                "scalar_balance": audit["scalar_incident"]["sum"],
                "unitarity_spectral_defect": audit["unitarity_spectral_defect"],
            }
        )
    return {
        "calculation": "neutral ell=1, J=L=2 axial two-channel scattering",
        "background": {
            "a1_0": solution.a0,
            "omega": solution.omega,
            "adm_mass": solution.adm_mass,
            "first_open_scalar_threshold": 1.0 - solution.omega,
        },
        "normalization": (
            "positive diagonal reciprocity metric inferred algebraically from the "
            "two-channel S matrix; this is not labeled a canonical physical flux"
        ),
        "frequency_scan": rows,
        "domain_audit_at_sigma_0p20": domains,
    }


def main() -> None:
    result = run_checkpoint()
    target = ROOT / "reports" / "axial" / "axial_scattering_checkpoint.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["frequency_scan"]:
        print(
            row["sigma"],
            row["gravity_incident"],
            row["scalar_incident"],
            row["unitarity_spectral_defect"],
        )


if __name__ == "__main__":
    main()
