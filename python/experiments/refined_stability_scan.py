"""Resolve the fundamental radial stability crossing on a fine amplitude grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from background.ell_boson_star import scan_sequence
from radial.spectral import solve_radial_spectrum


def zero_crossing_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return the interpolated root and the two samples that bracket it."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("x and y must be matching one-dimensional arrays")
    indices = np.flatnonzero(y[:-1] * y[1:] <= 0.0)
    if indices.size != 1:
        raise ValueError("the scan must contain exactly one sign-changing interval")
    index = int(indices[0])
    x0, x1 = x[index], x[index + 1]
    y0, y1 = y[index], y[index + 1]
    root = x0 - y0 * (x1 - x0) / (y1 - y0)
    return float(root), float(x0), float(x1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/radial/refined_radial_stability.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/radial/refined_radial_stability_summary.json"),
    )
    parser.add_argument("--a0-min", type=float, default=0.0990)
    parser.add_argument("--a0-max", type=float, default=0.1020)
    parser.add_argument("--count", type=int, default=13)
    parser.add_argument("--spectral-points", type=int, default=100)
    args = parser.parse_args()

    amplitudes = np.linspace(args.a0_min, args.a0_max, args.count)
    backgrounds = scan_sequence(
        1, amplitudes, r_max=80.0, points=800, tolerance=1e-7
    )
    rows = []
    for background in backgrounds:
        spectrum = solve_radial_spectrum(
            background,
            points=args.spectral_points,
            epsilon=1e-3,
            r_max=40.0,
            sigma2_min=-5e-4,
            sigma2_max=3e-3,
        )
        ground = next(mode for mode in spectrum if mode.node_count == 0)
        rows.append(
            {
                "a0": background.a0,
                "omega": background.omega,
                "adm_mass": background.adm_mass,
                "sigma2_ground": ground.sigma2,
                "signed_sigma_ground": np.copysign(
                    np.sqrt(abs(ground.sigma2)), ground.sigma2
                ),
                "scaled_generalized_residual": ground.scaled_generalized_residual,
                "unscaled_generalized_residual": ground.unscaled_generalized_residual,
                "eigenvalue_condition_number": ground.eigenvalue_condition_number,
                "node_count": ground.node_count,
            }
        )

    root, lower, upper = zero_crossing_linear(
        np.array([row["a0"] for row in rows]),
        np.array([row["sigma2_ground"] for row in rows]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "linear interpolation across the unique fine-grid sign change",
        "spectral_points": args.spectral_points,
        "r_max": 40.0,
        "epsilon": 1e-3,
        "estimated_a0_at_sigma2_zero": root,
        "bracket": [lower, upper],
        "half_bracket_width_as_grid_bound": 0.5 * (upper - lower),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} refined stability rows to {args.output}")
    print(f"zero crossing a0={root:.10f} bracketed by [{lower}, {upper}]")


if __name__ == "__main__":
    main()
