"""Audit competing effective-radius definitions on one background profile."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

from .ell_boson_star import BackgroundSolution, _rhs, solve_by_continuation


@dataclass(frozen=True)
class RadiusAudit:
    ell: int
    a0: float
    omega: float
    adm_mass_finite: float
    adm_mass_extrapolated: float
    eta_at_published_radius: float
    published_radius: float
    r99_mass: float
    r999_mass: float
    r99_proper_energy: float
    r99_noether_charge: float
    r99_mass_finite: float
    r99_mass_extrapolated: float
    extrapolation_anchor: float


def _fraction_radius(r: np.ndarray, cumulative: np.ndarray, fraction: float) -> float:
    if cumulative[-1] <= 0.0:
        raise ValueError("cumulative quantity must have a positive total")
    return float(np.interp(fraction * cumulative[-1], cumulative, r))


def _extrapolated_mass(solution: BackgroundSolution) -> tuple[float, float]:
    """Estimate the exponentially small scalar mass beyond an asymptotic anchor.

    The scalar is continued with Eq. (45) of arXiv:2103.15012 while the exterior
    metric is held Schwarzschild. The anchor is selected where the field has
    fallen below 1e-7 of its maximum but remains numerically resolved.
    """
    r = solution.r
    psi = np.abs(solution.psi)
    relative = psi / np.max(psi)
    candidates = np.flatnonzero((relative < 1e-7) & (relative > 1e-12))
    index = int(candidates[0]) if candidates.size else int(0.75 * (r.size - 1))
    anchor = float(r[index])
    psi_anchor = float(solution.psi[index])
    mass_anchor = float(solution.mass[index])
    omega = solution.omega
    ell = solution.ell
    kappa = 2 * ell + 1
    decay = np.sqrt(1.0 - omega**2)
    power = 1.0 + solution.adm_mass * (1.0 - 2.0 * omega**2) / decay

    def tail_mass_density(radius: float) -> float:
        field = psi_anchor * np.exp(-decay * (radius - anchor)) * (radius / anchor) ** (-power)
        derivative = field * (-decay - power / radius)
        inv_gamma2 = max(1.0 - 2.0 * solution.adm_mass / radius, 1e-14)
        alpha2 = inv_gamma2
        angular = ell * (ell + 1) / radius**2
        return 0.5 * kappa * radius**2 * (
            inv_gamma2 * derivative**2
            + (omega**2 / alpha2 + 1.0 + angular) * field**2
        )

    tail, _ = quad(tail_mass_density, anchor, np.inf, epsabs=1e-14, epsrel=1e-10, limit=200)
    return mass_anchor + float(tail), anchor


def audit_radii(solution: BackgroundSolution, published_radius: float = 12.75) -> RadiusAudit:
    r = solution.r
    gamma = solution.gamma
    rhs = _rhs(r, np.vstack((solution.mass, solution.alpha, solution.psi, solution.dpsi)), solution.omega, solution.ell)
    mass_density = rhs[0]
    proper_energy = cumulative_trapezoid(gamma * mass_density, r, initial=0.0)
    kappa = 2 * solution.ell + 1
    charge_density = kappa * solution.omega * solution.psi**2 * gamma * r**2 / solution.alpha
    charge = cumulative_trapezoid(charge_density, r, initial=0.0)
    extrapolated_mass, anchor = _extrapolated_mass(solution)

    finite_mass = solution.adm_mass
    r99_extrapolated = float(np.interp(0.99 * extrapolated_mass, solution.mass, r))
    return RadiusAudit(
        ell=solution.ell,
        a0=solution.a0,
        omega=solution.omega,
        adm_mass_finite=finite_mass,
        adm_mass_extrapolated=extrapolated_mass,
        eta_at_published_radius=float(np.interp(published_radius, r, solution.mass) / finite_mass),
        published_radius=published_radius,
        r99_mass=solution.r99,
        r999_mass=solution.r999,
        r99_proper_energy=_fraction_radius(r, proper_energy, 0.99),
        r99_noether_charge=_fraction_radius(r, charge, 0.99),
        r99_mass_finite=solution.r99,
        r99_mass_extrapolated=r99_extrapolated,
        extrapolation_anchor=anchor,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0", type=float, default=0.10)
    parser.add_argument("--published-radius", type=float, default=12.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    solution = solve_by_continuation(1, args.a0)
    audit = audit_radii(solution, args.published_radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(audit), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(audit), indent=2))


if __name__ == "__main__":
    main()

