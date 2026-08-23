"""Coherent-state occupation-noise toy estimate for the axial QNM.

Model specification: the three ell=1 magnetic modes are equal-amplitude
coherent states, the stress tensor is normal ordered in the asymptotic vacuum,
and only total Poisson occupation noise is retained.  The response uses the
computed Noether-charge sequence, N proportional to Q(a1^0), rather than an
assumed square-root relation for a self-gravitating equilibrium.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import matching_determinant, matching_singular_value
from radial.coefficients import RadialBackground


def complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def coherent_qnm_statistics(
    particle_number: float,
    *,
    sigma: complex,
    first_derivative: complex,
    second_derivative: complex,
    charge: float,
    charge_first_derivative: float,
    charge_second_derivative: float,
) -> dict[str, object]:
    if particle_number <= 0.0:
        raise ValueError("particle_number must be positive")
    amplitude_noise_coefficient = charge / charge_first_derivative
    amplitude_variance_coefficient = amplitude_noise_coefficient**2
    amplitude_mean_coefficient = (
        -0.5 * charge**2 * charge_second_derivative / charge_first_derivative**3
    )
    standard_deviation = (
        abs(first_derivative)
        * abs(amplitude_noise_coefficient)
        / np.sqrt(particle_number)
    )
    mean_shift = (
        first_derivative * amplitude_mean_coefficient
        + 0.5 * second_derivative * amplitude_variance_coefficient
    ) / particle_number
    return {
        "particle_number": float(particle_number),
        "delta_a1_0_times_sqrt_N": float(abs(amplitude_noise_coefficient)),
        "qnm_absolute_standard_deviation": float(standard_deviation),
        "qnm_fractional_standard_deviation": float(standard_deviation / abs(sigma)),
        "mean_qnm_shift": complex_pair(mean_shift),
        "mean_qnm_fractional_shift_magnitude": float(abs(mean_shift) / abs(sigma)),
    }


def main() -> None:
    branch_data = json.loads((ROOT / "reports" / "axial" / "axial_qnm_branch.json").read_text())
    certified_modes = {
        round(float(row["a1_0"]), 3): complex(row["sigma_real"], row["sigma_imag"])
        for row in branch_data["rows"]
    }
    center_sigma = certified_modes[0.08]
    h = 0.005
    low = certified_modes[0.075]
    high = certified_modes[0.085]
    first = (high - low) / (2.0 * h)
    second = (high - 2.0 * center_sigma + low) / h**2

    with (ROOT / "reports" / "background" / "ell1_sequence.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        sequence = list(csv.DictReader(stream))
    local_charge = {
        round(float(row["a0"]), 3): float(row["noether_charge"])
        for row in sequence
        if 0.074 < float(row["a0"]) < 0.086
    }
    charge_low, charge_center, charge_high = (
        local_charge[0.075], local_charge[0.08], local_charge[0.085]
    )
    charge_first = (charge_high - charge_low) / (2.0 * h)
    charge_second = (charge_high - 2.0 * charge_center + charge_low) / h**2

    mode_audit = []
    for amplitude in (0.075, 0.085):
        sigma = certified_modes[amplitude]
        tag = f"{int(round(amplitude * 1000)):03d}"
        solution = BackgroundSolution.load(
            ROOT / "reports" / "axial" / f"background_ell1_a{tag}_axial_branch.npz"
        )
        background = RadialBackground(solution, geometry_representation="hermite")
        determinant = matching_determinant(
            sigma,
            background,
            r_match=14.0,
            r_end=35.0,
            rtol=2e-7,
            atol=2e-9,
        )
        singular = matching_singular_value(
            sigma,
            background,
            r_match=14.0,
            r_end=35.0,
            rtol=2e-7,
            atol=2e-9,
        )
        mode_audit.append(
            {
                "a1_0": amplitude,
                "sigma": complex_pair(sigma),
                "determinant_abs": float(abs(determinant)),
                "normalized_min_singular_value": singular,
            }
        )

    with (ROOT / "reports" / "extensions" / "physics_extension_scaling.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        scaling = list(csv.DictReader(stream))
    statistics = []
    for source in scaling:
        row = {
            "boson_mass_ev": float(source["boson_mass_ev"]),
        }
        row.update(
            coherent_qnm_statistics(
                float(source["particle_number"]),
                sigma=center_sigma,
                first_derivative=first,
                second_derivative=second,
                charge=charge_center,
                charge_first_derivative=charge_first,
                charge_second_derivative=charge_second,
            )
        )
        statistics.append(row)

    result = {
        "calculation": "coherent-state occupation-noise toy estimate for the axial QNM",
        "state": (
            "product coherent state in the three ell=1 magnetic modes with "
            "equal mean occupation and Poisson total-number variance"
        ),
        "renormalization": (
            "normal ordering relative to the asymptotic Minkowski vacuum; "
            "vacuum-polarization terms are zero by this specified prescription"
        ),
        "response_model": "adiabatic one-mode map N proportional to the computed Noether charge Q(a1^0)",
        "qnm_branch_source": "reports/axial/axial_qnm_branch.json; every input mode has two domains and a locally counted contour",
        "charge_response": {
            "Q": charge_center,
            "dQ_d_a1_0": charge_first,
            "d2Q_d_a1_0_squared": charge_second,
            "Q_over_dQ_d_a1_0": charge_center / charge_first,
        },
        "mean_semiclassical_equations": (
            "identical to the classical EKG equations at leading coherent-state order"
        ),
        "qnm_sensitivity": {
            "center_sigma": complex_pair(center_sigma),
            "d_sigma_d_a1_0": complex_pair(first),
            "d2_sigma_d_a1_0_squared": complex_pair(second),
            "side_mode_audit": mode_audit,
        },
        "mass_scaling": statistics,
        "largest_fractional_standard_deviation": max(
            row["qnm_fractional_standard_deviation"] for row in statistics
        ),
        "largest_mean_fractional_shift": max(
            row["mean_qnm_fractional_shift_magnitude"] for row in statistics
        ),
    }
    target = ROOT / "reports" / "extensions" / "semiclassical_coherent_qnm.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sensitivity": result["qnm_sensitivity"],
        "largest_fractional_standard_deviation": result["largest_fractional_standard_deviation"],
        "largest_mean_fractional_shift": result["largest_mean_fractional_shift"],
    }, indent=2))


if __name__ == "__main__":
    main()
