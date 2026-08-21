"""Regenerate the versioned nonlinear-BVP radial certification table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from background.ell_boson_star import solve_by_continuation
from .bvp import solve_radial_bvp

BACKGROUND_COMMIT = "600c3677290bb05e8ac0901b945f7f98806af98c"
SOLVER_VERSION = "radial-v0.2-bvp-provisional"


def _case(a0, r_max, sigma2_guess, center_c_guess):
    background = solve_by_continuation(1, a0)
    mode = solve_radial_bvp(
        background,
        sigma2_guess=sigma2_guess,
        center_c_guess=center_c_guess,
        epsilon=1e-3,
        r_max=r_max,
        points=350,
        tolerance=3e-6,
    )
    return {
        "background_commit": BACKGROUND_COMMIT,
        "solver_version": SOLVER_VERSION,
        "formulation": "nonlinear_global_bvp",
        "a0": a0,
        "omega": background.omega,
        "alpha_c": background.alpha[0],
        "epsilon": mode.epsilon,
        "r_max": mode.r_max,
        "initial_points": 350,
        "bvp_tolerance": 3e-6,
        "sigma2": mode.sigma2,
        "center_c": mode.center_c,
        "physical_F1": mode.physical_boundary_residual[0],
        "physical_F2": mode.physical_boundary_residual[1],
        "max_collocation_residual": mode.max_collocation_residual,
        "node_count": mode.node_count,
        "nodes_used": mode.nodes_used,
        "solver_status": "converged" if mode.success else "failed",
        "environment_lock": "environment/requirements-lock.txt",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        (0.08, 20.0, 2.4e-4, -2.78e-2),
        (0.08, 25.0, 2.4e-4, -2.78e-2),
        (0.08, 30.0, 2.4e-4, -2.78e-2),
        (0.08, 40.0, 2.4e-4, -2.78e-2),
        (0.05, 25.0, 3.8e-4, -1.92e-2),
        (0.10, 25.0, 4.0e-6, -3.33e-2),
        (0.105, 25.0, -7.1e-5, -3.47e-2),
    ]
    rows = [_case(*case) for case in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} BVP certification rows to {args.output}")


if __name__ == "__main__":
    main()
