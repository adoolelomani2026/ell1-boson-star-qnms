"""Static, ground-state ell-boson stars in the Alcubierre et al. normalization."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp


@dataclass(frozen=True)
class BackgroundSolution:
    ell: int
    a0: float
    omega: float
    r: np.ndarray
    mass: np.ndarray
    alpha: np.ndarray
    psi: np.ndarray
    dpsi: np.ndarray
    noether_charge: float
    r99: float
    compactness99: float
    r999: float
    compactness999: float
    max_ode_residual: float
    tail_residual: float
    solver_status: int
    solver_message: str

    @property
    def gamma(self) -> np.ndarray:
        return 1.0 / np.sqrt(1.0 - 2.0 * self.mass / self.r)

    @property
    def adm_mass(self) -> float:
        return float(self.mass[-1])

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            ell=self.ell,
            a0=self.a0,
            omega=self.omega,
            r=self.r,
            mass=self.mass,
            alpha=self.alpha,
            gamma=self.gamma,
            psi=self.psi,
            dpsi=self.dpsi,
            noether_charge=self.noether_charge,
            r99=self.r99,
            compactness99=self.compactness99,
            r999=self.r999,
            compactness999=self.compactness999,
            max_ode_residual=self.max_ode_residual,
            tail_residual=self.tail_residual,
            solver_status=self.solver_status,
            solver_message=self.solver_message,
        )


def _rhs(r: np.ndarray, y: np.ndarray, omega: float, ell: int) -> np.ndarray:
    mass, alpha, psi, dpsi = y
    kappa = 2 * ell + 1
    angular = ell * (ell + 1) / r**2
    inv_gamma2 = 1.0 - 2.0 * mass / r
    gamma2 = 1.0 / inv_gamma2

    mass_p = 0.5 * kappa * r**2 * (
        inv_gamma2 * dpsi**2 + (omega**2 / alpha**2 + 1.0 + angular) * psi**2
    )
    log_alpha_p = gamma2 * (
        mass / r**2
        + 0.5
        * kappa
        * r
        * (inv_gamma2 * dpsi**2 + (omega**2 / alpha**2 - 1.0 - angular) * psi**2)
    )
    alpha_p = alpha * log_alpha_p
    log_gamma_p = gamma2 * (mass_p / r - mass / r**2)
    dpsi_p = -(
        2.0 / r + log_alpha_p - log_gamma_p
    ) * dpsi - gamma2 * (omega**2 / alpha**2 - 1.0 - angular) * psi
    return np.vstack((mass_p, alpha_p, dpsi, dpsi_p))


def _center_mass(r0: float, leading: float, ell: int) -> float:
    """Leading regular-center Misner-Sharp mass."""
    kappa = 2 * ell + 1
    coefficient = 0.5 * kappa * leading**2 * ell * (2 * ell + 1)
    return coefficient * r0 ** (2 * ell + 1) / (2 * ell + 1)


def _initial_guess(r: np.ndarray, ell: int, a0: float, omega: float) -> np.ndarray:
    kappa = 2 * ell + 1
    leading = a0 / kappa
    decay = np.sqrt(max(1.0 - omega**2, 0.02))
    psi = leading * r**ell * np.exp(-decay * r)
    dpsi = psi * (ell / r - decay)
    density_shape = r**2 * psi**2
    mass_shape = cumulative_trapezoid(density_shape, r, initial=0.0)
    # The dilute branch approaches zero mass with the center amplitude.  A
    # square-root scaling is a useful Newtonian-limit continuation seed and,
    # unlike a fixed O(1) mass guess, does not attract the collocation solve to
    # a relativistic branch deeper in the mass-frequency spiral.
    effective_amplitude = a0 / kappa**2
    mass_scale = 4.0 * np.sqrt(effective_amplitude) * np.sqrt(ell + 1.0) / np.sqrt(2.0)
    mass = mass_scale * mass_shape / max(mass_shape[-1], 1e-30)
    alpha = 1.0 - min(0.55, 0.8 * np.sqrt(effective_amplitude)) * np.exp(
        -r / max(2.0, ell + 1.0)
    )
    return np.vstack((mass, alpha, psi, dpsi))


def _omega_from_parameter(parameter: float) -> float:
    """Map an unconstrained BVP parameter to the bound-state range (0, 1)."""
    if parameter >= 0.0:
        exponential = np.exp(-parameter)
        value = 1.0 / (1.0 + exponential)
        return float(np.clip(value, 1e-12, 1.0 - 1e-12))
    exponential = np.exp(parameter)
    value = exponential / (1.0 + exponential)
    return float(np.clip(value, 1e-12, 1.0 - 1e-12))


def _parameter_from_omega(omega: float) -> float:
    clipped = np.clip(omega, 1e-8, 1.0 - 1e-8)
    return float(np.log(clipped / (1.0 - clipped)))


def solve_background(
    ell: int,
    a0: float,
    *,
    r_max: float = 80.0,
    points: int = 800,
    tolerance: float = 1e-7,
    max_nodes: int = 40000,
    omega_guess: float | None = None,
    seed: BackgroundSolution | None = None,
) -> BackgroundSolution:
    """Solve one nodeless equilibrium model as a nonlinear BVP.

    ``a0`` is the tabulated literature parameter a_ell^0; in the rescaled
    radial-field convention the raw center coefficient is ``a0/(2*ell+1)``.
    """
    if ell < 0 or int(ell) != ell:
        raise ValueError("ell must be a nonnegative integer")
    if a0 <= 0.0:
        raise ValueError("a0 must be positive")
    if r_max <= 5.0:
        raise ValueError("r_max must exceed 5")

    ell = int(ell)
    r0 = 1e-5
    r = np.geomspace(r0, r_max, points)
    kappa = 2 * ell + 1
    leading = a0 / kappa
    if omega_guess is None:
        omega_guess = float(np.clip(1.0 - 0.8 * a0 ** (2.0 / 3.0), 0.72, 0.995))

    if seed is None:
        y_guess = _initial_guess(r, ell, a0, omega_guess)
        parameter_guess = np.array([_parameter_from_omega(omega_guess)])
    else:
        y_guess = np.vstack(
            [np.interp(r, seed.r, field) for field in (seed.mass, seed.alpha, seed.psi, seed.dpsi)]
        )
        ratio = a0 / seed.a0
        y_guess[2:] *= ratio
        parameter_guess = np.array([_parameter_from_omega(seed.omega)])

    center_mass = _center_mass(r0, leading, ell)

    def fun(radius: np.ndarray, fields: np.ndarray, p: np.ndarray) -> np.ndarray:
        return _rhs(radius, fields, _omega_from_parameter(float(p[0])), ell)

    def bc(left: np.ndarray, right: np.ndarray, p: np.ndarray) -> np.ndarray:
        omega = _omega_from_parameter(float(p[0]))
        decay = np.sqrt(max(1.0 - omega**2, 1e-14))
        exterior_lapse = np.sqrt(max(1.0 - 2.0 * right[0] / r_max, 1e-14))
        power = 1.0 + right[0] * (1.0 - 2.0 * omega**2) / decay
        return np.array(
            [
                left[0] - center_mass,
                left[2] - leading * r0**ell,
                left[3] - ell * leading * r0 ** (ell - 1),
                right[1] - exterior_lapse,
                right[3] + (decay + power / r_max) * right[2],
            ]
        )

    result = solve_bvp(
        fun,
        bc,
        r,
        y_guess,
        p=parameter_guess,
        tol=tolerance,
        max_nodes=max_nodes,
        verbose=0,
    )
    if not result.success:
        raise RuntimeError(f"background BVP failed: {result.message}")

    dense_r = np.geomspace(r0, r_max, max(2400, 3 * points))
    fields = result.sol(dense_r)
    mass, alpha, psi, dpsi = fields
    omega = _omega_from_parameter(float(result.p[0]))
    rhs = _rhs(dense_r, fields, omega, ell)
    numerical_derivative = np.gradient(fields, dense_r, axis=1, edge_order=2)
    scales = np.maximum(np.max(np.abs(rhs), axis=1), 1.0)
    max_residual = float(np.max(np.abs(numerical_derivative - rhs) / scales[:, None]))

    gamma = 1.0 / np.sqrt(1.0 - 2.0 * mass / dense_r)
    charge_density = kappa * omega * psi**2 * gamma * dense_r**2 / alpha
    charge = float(np.trapezoid(charge_density, dense_r))
    target_mass = 0.99 * mass[-1]
    r99 = float(np.interp(target_mass, mass, dense_r))
    compactness = float(mass[-1] / r99)
    target_mass999 = 0.999 * mass[-1]
    r999 = float(np.interp(target_mass999, mass, dense_r))
    compactness999 = float(mass[-1] / r999)
    decay = np.sqrt(max(1.0 - omega**2, 1e-14))
    power = 1.0 + mass[-1] * (1.0 - 2.0 * omega**2) / decay
    tail_factor = decay + power / r_max
    # At a large outer radius both Robin terms can be below floating-point
    # relative accuracy. Normalize by the profile amplitude so this remains a
    # useful absolute boundary residual instead of a ratio of two underflows.
    tail_scale = max(float(np.max(np.abs(psi))), 1e-30)
    tail_residual = float(abs(dpsi[-1] + tail_factor * psi[-1]) / tail_scale)

    return BackgroundSolution(
        ell=ell,
        a0=float(a0),
        omega=omega,
        r=dense_r,
        mass=mass,
        alpha=alpha,
        psi=psi,
        dpsi=dpsi,
        noether_charge=charge,
        r99=r99,
        compactness99=compactness,
        r999=r999,
        compactness999=compactness999,
        max_ode_residual=max_residual,
        tail_residual=tail_residual,
        solver_status=result.status,
        solver_message=result.message,
    )


def solve_by_continuation(ell: int, a0: float, **kwargs: object) -> BackgroundSolution:
    """Reach a target along the nodeless branch from the validated 0.08 anchor."""
    anchor = 0.08
    solution = solve_background(ell, anchor, **kwargs)
    if np.isclose(a0, anchor, rtol=0.0, atol=1e-14):
        return solution
    steps = max(2, int(np.ceil(abs(a0 - anchor) / 0.005)) + 1)
    for amplitude in np.linspace(anchor, a0, steps)[1:]:
        solution = solve_background(ell, float(amplitude), seed=solution, **kwargs)
    return solution


def _row(solution: BackgroundSolution) -> dict[str, object]:
    return {
        "ell": solution.ell,
        "a0": solution.a0,
        "omega": solution.omega,
        "adm_mass": solution.adm_mass,
        "noether_charge": solution.noether_charge,
        "r99": solution.r99,
        "compactness99": solution.compactness99,
        "r999": solution.r999,
        "compactness999": solution.compactness999,
        "max_ode_residual": solution.max_ode_residual,
        "tail_residual": solution.tail_residual,
        "solver_status": solution.solver_status,
    }


def scan_sequence(ell: int, amplitudes: Iterable[float], **kwargs: object) -> list[BackgroundSolution]:
    requested = np.asarray(list(amplitudes), dtype=float)
    if requested.size == 0:
        return []
    anchor = solve_background(ell, 0.08, **kwargs)
    by_amplitude: dict[float, BackgroundSolution] = {}
    lower = sorted((float(value) for value in requested if value <= 0.08), reverse=True)
    upper = sorted(float(value) for value in requested if value > 0.08)
    seed = anchor
    for amplitude in lower:
        if np.isclose(amplitude, 0.08, atol=1e-14, rtol=0.0):
            by_amplitude[amplitude] = anchor
        else:
            seed = solve_background(ell, amplitude, seed=seed, **kwargs)
            by_amplitude[amplitude] = seed
    seed = anchor
    for amplitude in upper:
        seed = solve_background(ell, amplitude, seed=seed, **kwargs)
        by_amplitude[amplitude] = seed
    return [by_amplitude[float(value)] for value in requested]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ell", type=int, default=1)
    common.add_argument("--r-max", type=float, default=80.0)
    common.add_argument("--points", type=int, default=800)
    common.add_argument("--tolerance", type=float, default=1e-7)

    one = sub.add_parser("solve", parents=[common])
    one.add_argument("--a0", type=float, required=True)
    one.add_argument("--output", type=Path, required=True)

    scan = sub.add_parser("scan", parents=[common])
    scan.add_argument("--a0-min", type=float, required=True)
    scan.add_argument("--a0-max", type=float, required=True)
    scan.add_argument("--count", type=int, default=24)
    scan.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    kwargs = {"r_max": args.r_max, "points": args.points, "tolerance": args.tolerance}
    if args.command == "solve":
        solution = solve_by_continuation(args.ell, args.a0, **kwargs)
        solution.save(args.output)
        print(_row(solution))
        return

    amplitudes = np.linspace(args.a0_min, args.a0_max, args.count)
    solutions = scan_sequence(args.ell, amplitudes, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [_row(solution) for solution in solutions]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} models to {args.output}")


if __name__ == "__main__":
    main()
