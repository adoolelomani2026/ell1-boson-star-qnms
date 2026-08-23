"""Regenerate the ell=1 sequence and add strong-field diagnostic columns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from background.ell_boson_star import scan_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/background/gravitational_diagnostics.csv"),
    )
    parser.add_argument("--a0-min", type=float, default=0.02)
    parser.add_argument("--a0-max", type=float, default=0.12)
    parser.add_argument("--count", type=int, default=21)
    parser.add_argument("--r-max", type=float, default=80.0)
    parser.add_argument("--points", type=int, default=800)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    amplitudes = np.linspace(args.a0_min, args.a0_max, args.count)
    solutions = scan_sequence(
        1,
        amplitudes,
        r_max=args.r_max,
        points=args.points,
        tolerance=args.tolerance,
    )
    rows = []
    for solution in solutions:
        local_compactness = 2.0 * solution.mass / solution.r
        surface_redshift_99 = (1.0 - 2.0 * solution.adm_mass / solution.r99) ** -0.5 - 1.0
        exterior_kretschmann_99 = 48.0 * solution.adm_mass**2 / solution.r99**6
        rows.append(
            {
                "a0": solution.a0,
                "omega": solution.omega,
                "adm_mass": solution.adm_mass,
                "noether_charge": solution.noether_charge,
                "binding_mass_minus_charge": solution.adm_mass
                - solution.noether_charge,
                "alpha_center": solution.alpha[0],
                "central_redshift": 1.0 / solution.alpha[0] - 1.0,
                "r99": solution.r99,
                "max_two_m_over_r": float(np.max(local_compactness)),
                "surface_redshift_99": float(surface_redshift_99),
                "exterior_kretschmann_99": float(exterior_kretschmann_99),
                "max_ode_residual": solution.max_ode_residual,
                "tail_residual": solution.tail_residual,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} strong-field rows to {args.output}")


if __name__ == "__main__":
    main()
