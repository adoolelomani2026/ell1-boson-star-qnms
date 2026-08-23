"""Dimensional gravitational, electromagnetic, and quantum scaling audit.

The equilibrium and radial solvers use G = c = hbar = mu = 1 and a neutral
complex scalar. This module does not modify those solutions. It translates a
dimensionless benchmark over a user-specified boson-mass grid and evaluates
small parameters that indicate when omitted electromagnetic or quantum physics
could matter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PLANCK_MASS_EV = 1.220890e28
SOLAR_MASS_EV = 1.115416e66
HBAR_C_EV_KM = 1.973269804e-10
FINE_STRUCTURE = 7.2973525693e-3


@dataclass(frozen=True)
class ExtensionScale:
    boson_mass_ev: float
    mass_solar: float
    r99_km: float
    particle_number: float
    finite_n_fractional_scale: float
    planck_suppression: float
    charge_balance_in_units_of_e: float


def physical_scales(
    boson_mass_ev: float,
    *,
    dimensionless_mass: float,
    dimensionless_charge: float,
    dimensionless_r99: float,
) -> ExtensionScale:
    """Translate one neutral solution in the standard natural-unit scaling.

    The charge-balance column is the constituent electric charge for which
    pairwise Coulomb and Newtonian gravitational forces are comparable:
    alpha * (q/e)^2 approximately equals (mu/M_Pl)^2. It is a diagnostic
    threshold, not a charged-star solution.
    """
    if boson_mass_ev <= 0:
        raise ValueError("boson_mass_ev must be positive")
    if min(dimensionless_mass, dimensionless_charge, dimensionless_r99) <= 0:
        raise ValueError("dimensionless benchmark values must be positive")

    ratio = boson_mass_ev / PLANCK_MASS_EV
    particle_number = dimensionless_charge / ratio**2
    return ExtensionScale(
        boson_mass_ev=float(boson_mass_ev),
        mass_solar=float(
            dimensionless_mass
            * PLANCK_MASS_EV**2
            / (boson_mass_ev * SOLAR_MASS_EV)
        ),
        r99_km=float(dimensionless_r99 * HBAR_C_EV_KM / boson_mass_ev),
        particle_number=float(particle_number),
        finite_n_fractional_scale=float(1.0 / np.sqrt(particle_number)),
        planck_suppression=float(ratio**2),
        charge_balance_in_units_of_e=float(ratio / np.sqrt(FINE_STRUCTURE)),
    )


def generate_audit(
    masses_ev: np.ndarray,
    *,
    dimensionless_mass: float = 1.171271161113,
    dimensionless_charge: float = 1.212966,
    dimensionless_r99: float = 11.16733443,
) -> list[ExtensionScale]:
    return [
        physical_scales(
            float(mass),
            dimensionless_mass=dimensionless_mass,
            dimensionless_charge=dimensionless_charge,
            dimensionless_r99=dimensionless_r99,
        )
        for mass in masses_ev
    ]


def write_audit(path: Path, rows: list[ExtensionScale]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/extensions/physics_extension_scaling.csv"),
    )
    parser.add_argument("--mass-min-ev", type=float, default=1e-22)
    parser.add_argument("--mass-max-ev", type=float, default=1e9)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    if args.count < 2:
        raise ValueError("count must be at least two")
    masses = np.geomspace(args.mass_min_ev, args.mass_max_ev, args.count)
    rows = generate_audit(masses)
    write_audit(args.output, rows)
    print(f"wrote {len(rows)} scale-audit rows to {args.output}")


if __name__ == "__main__":
    main()
