"""Run background and radial sensitivity experiments around the benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from background.ell_boson_star import solve_background, solve_by_continuation
from radial.bvp import solve_radial_bvp
from radial.spectral import solve_radial_spectrum


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--background-output",
        type=Path,
        default=Path("reports/background/background_sensitivity.csv"),
    )
    parser.add_argument(
        "--radial-output",
        type=Path,
        default=Path("reports/radial/radial_sensitivity.csv"),
    )
    args = parser.parse_args()

    cases: list[tuple[str, float, int, float]] = []
    cases.extend(("outer_domain", rmax, 800, 1e-7) for rmax in (40.0, 60.0, 80.0, 100.0))
    cases.extend(("mesh_seed", 80.0, points, 1e-7) for points in (400, 600, 800, 1200))
    cases.extend(("collocation_tolerance", 80.0, 800, tol) for tol in (1e-5, 1e-6, 1e-7, 1e-8))
    background_rows = []
    baseline_seed = solve_by_continuation(1, 0.08)
    for family, rmax, points, tolerance in cases:
        solution = solve_background(
            1,
            0.08,
            r_max=rmax,
            points=points,
            tolerance=tolerance,
            seed=baseline_seed,
        )
        background_rows.append(
            {
                "experiment": family,
                "r_max": rmax,
                "initial_points": points,
                "tolerance": tolerance,
                "omega": solution.omega,
                "adm_mass": solution.adm_mass,
                "noether_charge": solution.noether_charge,
                "r99": solution.r99,
                "r999": solution.r999,
                "max_ode_residual": solution.max_ode_residual,
                "tail_residual": solution.tail_residual,
            }
        )
    write_rows(args.background_output, background_rows)

    background = solve_by_continuation(1, 0.08)
    radial_rows = []
    for epsilon in (5e-4, 1e-3, 2e-3, 4e-3):
        mode = solve_radial_bvp(
            background,
            sigma2_guess=2.40043e-4,
            center_c_guess=-2.7836e-2,
            epsilon=epsilon,
            r_max=40.0,
            points=350,
            tolerance=3e-7,
        )
        radial_rows.append(
            {
                "method": "nonlinear_global_bvp",
                "epsilon": epsilon,
                "r_max": 40.0,
                "resolution": 350,
                "sigma2": mode.sigma2,
                "node_count": mode.node_count,
                "residual": mode.max_dense_pointwise_relative_residual,
                "condition_number": "",
            }
        )
    for points in (40, 50, 60, 80, 100, 120, 140, 160, 180, 200):
        modes = solve_radial_spectrum(
            background,
            points=points,
            epsilon=1e-3,
            r_max=40.0,
            sigma2_min=-1e-3,
            sigma2_max=0.02,
        )
        for mode_index in (0, 1):
            mode = next(item for item in modes if item.node_count == mode_index)
            radial_rows.append(
                {
                    "method": f"chebyshev_mode_{mode_index}",
                    "epsilon": 1e-3,
                    "r_max": 40.0,
                    "resolution": points,
                    "sigma2": mode.sigma2,
                    "node_count": mode.node_count,
                    "residual": mode.unscaled_generalized_residual,
                    "condition_number": mode.eigenvalue_condition_number,
                }
            )
    write_rows(args.radial_output, radial_rows)
    print(f"wrote {len(background_rows)} background sensitivity rows")
    print(f"wrote {len(radial_rows)} radial sensitivity rows")


if __name__ == "__main__":
    main()
