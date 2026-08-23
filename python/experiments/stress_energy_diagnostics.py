"""Evaluate effective density, pressures, anisotropy, and energy conditions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from background.ell_boson_star import scan_sequence


def stress_components(solution):
    ell = solution.ell
    kappa = 2 * ell + 1
    r = solution.r
    angular = ell * (ell + 1.0) / r**2
    inv_gamma2 = 1.0 - 2.0 * solution.mass / r
    kinetic_t = solution.omega**2 * solution.psi**2 / solution.alpha**2
    kinetic_r = inv_gamma2 * solution.dpsi**2
    mass_term = solution.psi**2
    angular_term = angular * solution.psi**2
    factor = 0.5 * kappa
    rho = factor * (kinetic_t + kinetic_r + mass_term + angular_term)
    pressure_r = factor * (kinetic_t + kinetic_r - mass_term - angular_term)
    pressure_t = factor * (kinetic_t - kinetic_r - mass_term)
    return rho, pressure_r, pressure_t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/background/stress_energy_diagnostics.csv"),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("reports/background/stress_energy_profiles.csv"),
    )
    args = parser.parse_args()

    amplitudes = np.linspace(0.02, 0.12, 21)
    solutions = scan_sequence(1, amplitudes)
    rows = []
    profiles = []
    profile_amplitudes = {0.04, 0.08, 0.12}
    for solution in solutions:
        rho, pressure_r, pressure_t = stress_components(solution)
        peak = float(np.max(rho))
        support = rho > peak * 1e-10
        anisotropy = pressure_t - pressure_r
        rows.append(
            {
                "a0": solution.a0,
                "rho_peak": peak,
                "r_at_rho_peak": float(solution.r[np.argmax(rho)]),
                "min_rho_plus_pr_over_peak": float(np.min((rho + pressure_r)[support]) / peak),
                "min_rho_plus_pt_over_peak": float(np.min((rho + pressure_t)[support]) / peak),
                "min_rho_minus_abs_pr_over_peak": float(
                    np.min((rho - np.abs(pressure_r))[support]) / peak
                ),
                "min_rho_minus_abs_pt_over_peak": float(
                    np.min((rho - np.abs(pressure_t))[support]) / peak
                ),
                "max_abs_anisotropy_over_peak": float(
                    np.max(np.abs(anisotropy)[support]) / peak
                ),
                "r_at_max_abs_anisotropy": float(
                    solution.r[np.argmax(np.abs(anisotropy) * support)]
                ),
            }
        )
        if any(np.isclose(solution.a0, value) for value in profile_amplitudes):
            indices = np.unique(
                np.linspace(0, solution.r.size - 1, 320).astype(int)
            )
            for index in indices:
                profiles.append(
                    {
                        "a0": solution.a0,
                        "r": solution.r[index],
                        "rho": rho[index],
                        "pressure_r": pressure_r[index],
                        "pressure_t": pressure_t[index],
                        "anisotropy": anisotropy[index],
                    }
                )

    for path, records in ((args.output, rows), (args.profiles, profiles)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f"wrote {len(rows)} stress-energy summary rows")
    print(f"wrote {len(profiles)} stress-energy profile rows")


if __name__ == "__main__":
    main()
