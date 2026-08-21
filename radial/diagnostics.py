"""Regenerate the versioned radial benchmark table (a deliberately slow run)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from background.ell_boson_star import solve_by_continuation
from .shooting import solve_radial_mode


BACKGROUND_COMMIT = "600c3677290bb05e8ac0901b945f7f98806af98c"


def _case(a0, r_max, method, rtol, atol, epsilon, guess, bracket):
    background = solve_by_continuation(1, a0)
    mode = solve_radial_mode(
        background,
        sigma2_guess=guess,
        sigma2_bracket=bracket,
        epsilon=epsilon,
        r_max=r_max,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return {
        "background_commit": BACKGROUND_COMMIT,
        "a0": a0,
        "omega": background.omega,
        "alpha_c": background.alpha[0],
        "epsilon": epsilon,
        "r_max": r_max,
        "method": method,
        "rtol": rtol,
        "atol": atol,
        "sigma2": mode.sigma2,
        "center_c": mode.center_c,
        "boundary_residual_norm": float(np.max(np.abs(mode.residual / mode.residual_scale))),
        "node_count": mode.node_count,
        "solver_status": "converged" if mode.root_success and mode.ivp_success else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for radius in (20.0, 25.0, 30.0, 40.0):
        cases.append((0.08, radius, "Radau", 3e-8, 3e-10, 1e-5, 2.4e-4, (2e-4, 3e-4)))
    cases.extend(
        [
            (0.08, 25.0, "DOP853", 1e-8, 1e-10, 1e-3, 2.4e-4, (2e-4, 3e-4)),
            (0.08, 25.0, "DOP853", 1e-8, 1e-10, 1e-5, 2.4e-4, (2e-4, 3e-4)),
            (0.08, 25.0, "DOP853", 3e-9, 3e-11, 1e-5, 2.4e-4, (2e-4, 3e-4)),
            (0.05, 25.0, "DOP853", 2e-8, 2e-10, 1e-5, 3.8e-4, (3e-4, 5e-4)),
            (0.10, 25.0, "DOP853", 2e-8, 2e-10, 1e-5, 4e-6, (-5e-5, 5e-5)),
            (0.105, 25.0, "DOP853", 2e-8, 2e-10, 1e-5, -7.1e-5, (-1.5e-4, -1e-7)),
        ]
    )
    rows = [_case(*case) for case in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} radial benchmark rows to {args.output}")


if __name__ == "__main__":
    main()

