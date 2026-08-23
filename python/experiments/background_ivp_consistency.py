"""Re-integrate collocation backgrounds with an independent IVP algorithm."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from background.ell_boson_star import _rhs, solve_by_continuation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/background/background_ivp_consistency.csv"),
    )
    args = parser.parse_args()
    rows = []
    for a0 in (0.04, 0.08, 0.10):
        solution = solve_by_continuation(1, a0)
        initial = np.array(
            [
                solution.mass[0],
                solution.alpha[0],
                solution.psi[0],
                solution.dpsi[0],
            ]
        )
        for endpoint in (10.0, 15.0, 20.0):
            integration = solve_ivp(
                lambda radius, state: _rhs(
                    np.array([radius]), state[:, None], solution.omega, 1
                )[:, 0],
                (solution.r[0], endpoint),
                initial,
                method="DOP853",
                rtol=1e-11,
                atol=1e-13,
                dense_output=True,
            )
            radii = np.geomspace(solution.r[0], endpoint, 1200)
            ivp = integration.sol(radii)
            collocation = np.vstack(
                [
                    np.interp(radii, solution.r, field)
                    for field in (
                        solution.mass,
                        solution.alpha,
                        solution.psi,
                        solution.dpsi,
                    )
                ]
            )
            scales = np.maximum(np.max(np.abs(collocation), axis=1), 1e-12)
            errors = np.max(
                np.abs(ivp - collocation) / scales[:, None], axis=1
            )
            rows.append(
                {
                    "a0": a0,
                    "endpoint": endpoint,
                    "ivp_success": integration.success,
                    "ivp_function_evaluations": integration.nfev,
                    "mass_scaled_max_difference": errors[0],
                    "alpha_scaled_max_difference": errors[1],
                    "psi_scaled_max_difference": errors[2],
                    "dpsi_scaled_max_difference": errors[3],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} independent-IVP comparison rows")


if __name__ == "__main__":
    main()
