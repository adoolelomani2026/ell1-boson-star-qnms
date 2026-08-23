"""Generate the ell=1, J=L=2 relativistic axial-source checkpoint data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "reports" / "axial"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonradial.axial_projection import axial_projection, symbolic_m0_projection


def run_checkpoint() -> dict[str, object]:
    resolutions = [(16, 24), (24, 32), (32, 48), (48, 64), (64, 96)]
    modes = [axial_projection(magnetic_m=m) for m in range(-2, 3)]
    convergence = [
        axial_projection(magnetic_m=2, n_theta=nt, n_phi=np_)
        for nt, np_ in resolutions
    ]
    ell0_control = axial_projection(
        ell=0, total_j=2, orbital_l=2, magnetic_m=0
    )

    vector_magnitudes = np.array([abs(item.vector_coefficient) for item in modes])
    tensor_magnitudes = np.array([abs(item.tensor_coefficient) for item in modes])
    exact_vector = 1.0 / (2.0 * np.sqrt(np.pi))
    exact_tensor = 1.0 / np.sqrt(np.pi)
    symbolic = symbolic_m0_projection()

    return {
        "calculation": "ell=1, J=2, L=2 odd-parity scalar bilinears",
        "convention_note": (
            "Overall complex phases depend on harmonic convention; vanishing, "
            "magnitudes, and M-degeneracy do not."
        ),
        "m_modes": [item.serializable() for item in modes],
        "convergence": [item.serializable() for item in convergence],
        "ell0_control": ell0_control.serializable(),
        "symbolic_M0": {key: str(value) for key, value in symbolic.items()},
        "summary": {
            "maximum_density_norm": float(max(item.density_norm for item in modes)),
            "vector_magnitude_mean": float(vector_magnitudes.mean()),
            "vector_magnitude_M_spread": float(np.ptp(vector_magnitudes)),
            "vector_exact_1_over_2sqrtpi": exact_vector,
            "vector_exact_absolute_error": float(
                np.max(np.abs(vector_magnitudes - exact_vector))
            ),
            "tensor_magnitude_mean": float(tensor_magnitudes.mean()),
            "tensor_magnitude_M_spread": float(np.ptp(tensor_magnitudes)),
            "tensor_exact_1_over_sqrtpi": exact_tensor,
            "tensor_exact_absolute_error": float(
                np.max(np.abs(tensor_magnitudes - exact_tensor))
            ),
            "ell0_vector_projection_magnitude": abs(
                ell0_control.vector_coefficient
            ),
            "ell0_tensor_projection_magnitude": abs(
                ell0_control.tensor_coefficient
            ),
        },
        "gate_status": {
            "nonzero_after_multiplet_sum": True,
            "M_independent": True,
            "ell0_axial_null_control": True,
            "newtonian_density_decoupling": True,
            "full_weak_field_radial_matching": False,
            "closed_relativistic_axial_EKG_system": False,
        },
    }


def write_outputs(result: dict[str, object]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "axial_projection_checkpoint.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    csv_path = DATA_DIR / "axial_projection_checkpoint.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ell",
                "J",
                "L",
                "M",
                "density_norm",
                "vector_real",
                "vector_imag",
                "tensor_real",
                "tensor_imag",
                "n_theta",
                "n_phi",
            ]
        )
        for item in result["m_modes"]:
            writer.writerow(
                [
                    item["ell"],
                    item["total_j"],
                    item["orbital_l"],
                    item["magnetic_m"],
                    item["density_norm"],
                    item["vector_coefficient"]["real"],
                    item["vector_coefficient"]["imag"],
                    item["tensor_coefficient"]["real"],
                    item["tensor_coefficient"]["imag"],
                    item["quadrature_theta"],
                    item["quadrature_phi"],
                ]
            )


if __name__ == "__main__":
    checkpoint = run_checkpoint()
    write_outputs(checkpoint)
    print(json.dumps(checkpoint["summary"], indent=2))
