"""Charged ell-boson-star equilibria in Einstein--Maxwell--Klein--Gordon theory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import solve_bvp

from background.ell_boson_star import (
    BackgroundSolution,
    _center_mass,
    _omega_from_parameter,
    _parameter_from_omega,
)


@dataclass
class ChargedBackgroundSolution:
    ell: int
    a0: float
    gauge_charge: float
    omega: float
    r: np.ndarray
    mass: np.ndarray
    alpha: np.ndarray
    psi: np.ndarray
    dpsi: np.ndarray
    electric_potential: np.ndarray
    electric_flux: np.ndarray
    noether_charge: float
    r99: float
    compactness99: float
    max_ode_residual: float
    tail_residual: float
    solver_status: int
    solver_message: str

    @property
    def adm_mass(self) -> float:
        # Outside the charge distribution m(r)=M_ADM-2 pi Q_E^2/r.
        return float(self.mass[-1] + 2.0 * np.pi * self.electric_flux[-1] ** 2 / self.r[-1])

    @property
    def total_electric_charge(self) -> float:
        return float(4.0 * np.pi * self.electric_flux[-1])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            ell=self.ell,
            a0=self.a0,
            gauge_charge=self.gauge_charge,
            omega=self.omega,
            r=self.r,
            mass=self.mass,
            alpha=self.alpha,
            psi=self.psi,
            dpsi=self.dpsi,
            electric_potential=self.electric_potential,
            electric_flux=self.electric_flux,
            noether_charge=self.noether_charge,
            r99=self.r99,
            compactness99=self.compactness99,
            max_ode_residual=self.max_ode_residual,
            tail_residual=self.tail_residual,
            solver_status=self.solver_status,
            solver_message=self.solver_message,
        )


def charged_rhs(
    radius: np.ndarray,
    fields: np.ndarray,
    omega: float,
    ell: int,
    gauge_charge: float,
) -> np.ndarray:
    mass, alpha, psi, dpsi, potential, electric_flux = fields
    multiplicity = 2 * ell + 1
    angular = ell * (ell + 1.0) / radius**2
    inv_gamma2 = 1.0 - 2.0 * mass / radius
    gamma2 = 1.0 / inv_gamma2
    gamma = np.sqrt(gamma2)
    local_frequency = omega - gauge_charge * potential

    mass_prime = 0.5 * multiplicity * radius**2 * (
        inv_gamma2 * dpsi**2
        + (local_frequency**2 / alpha**2 + 1.0 + angular) * psi**2
    ) + 2.0 * np.pi * electric_flux**2 / radius**2
    log_alpha_prime = gamma2 * (
        mass / radius**2
        + 0.5
        * multiplicity
        * radius
        * (
            inv_gamma2 * dpsi**2
            + (local_frequency**2 / alpha**2 - 1.0 - angular) * psi**2
        )
        - 2.0 * np.pi * electric_flux**2 / radius**3
    )
    alpha_prime = alpha * log_alpha_prime
    log_gamma_prime = gamma2 * (mass_prime / radius - mass / radius**2)
    dpsi_prime = -(
        2.0 / radius + log_alpha_prime - log_gamma_prime
    ) * dpsi - gamma2 * (
        local_frequency**2 / alpha**2 - 1.0 - angular
    ) * psi
    potential_prime = -alpha * gamma * electric_flux / radius**2
    flux_prime = (
        gauge_charge
        * multiplicity
        / (4.0 * np.pi)
        * gamma
        / alpha
        * radius**2
        * local_frequency
        * psi**2
    )
    return np.vstack(
        (
            mass_prime,
            alpha_prime,
            dpsi,
            dpsi_prime,
            potential_prime,
            flux_prime,
        )
    )


def _neutral_seed_fields(
    radius: np.ndarray, neutral: BackgroundSolution, gauge_charge: float
) -> np.ndarray:
    base = np.vstack(
        [
            np.interp(radius, neutral.r, field)
            for field in (neutral.mass, neutral.alpha, neutral.psi, neutral.dpsi)
        ]
    )
    gamma = 1.0 / np.sqrt(1.0 - 2.0 * base[0] / radius)
    density = (
        gauge_charge
        * (2 * neutral.ell + 1)
        / (4.0 * np.pi)
        * gamma
        / base[1]
        * radius**2
        * neutral.omega
        * base[2] ** 2
    )
    flux = cumulative_trapezoid(density, radius, initial=0.0)
    integrand = base[1] * gamma * flux / radius**2
    potential = flux[-1] / radius[-1] - cumulative_trapezoid(
        integrand[::-1], radius[::-1], initial=0.0
    )[::-1]
    return np.vstack((base, potential, flux))


def solve_charged_background(
    ell: int,
    a0: float,
    gauge_charge: float,
    *,
    r_max: float = 80.0,
    points: int = 900,
    tolerance: float = 2e-7,
    max_nodes: int = 60000,
    neutral_seed: BackgroundSolution | None = None,
    seed: ChargedBackgroundSolution | None = None,
) -> ChargedBackgroundSolution:
    if gauge_charge < 0.0:
        raise ValueError("gauge_charge must be nonnegative")
    if seed is None and neutral_seed is None:
        raise ValueError("a neutral_seed or charged seed is required")
    ell = int(ell)
    r0 = 1e-5
    radius = np.geomspace(r0, r_max, points)
    multiplicity = 2 * ell + 1
    leading = a0 / multiplicity
    if seed is None:
        assert neutral_seed is not None
        guess = _neutral_seed_fields(radius, neutral_seed, gauge_charge)
        omega_guess = neutral_seed.omega
    else:
        guess = np.vstack(
            [
                np.interp(radius, seed.r, field)
                for field in (
                    seed.mass,
                    seed.alpha,
                    seed.psi,
                    seed.dpsi,
                    seed.electric_potential,
                    seed.electric_flux,
                )
            ]
        )
        omega_guess = seed.omega
    parameter_guess = np.asarray((_parameter_from_omega(omega_guess),))
    center_mass = _center_mass(r0, leading, ell)

    def fun(r, y, p):
        return charged_rhs(
            r, y, _omega_from_parameter(float(p[0])), ell, gauge_charge
        )

    def bc(left, right, p):
        omega = _omega_from_parameter(float(p[0]))
        decay = np.sqrt(max(1.0 - omega**2, 1e-14))
        adm_mass = right[0] + 2.0 * np.pi * right[5] ** 2 / r_max
        exterior_f = 1.0 - 2.0 * adm_mass / r_max + 4.0 * np.pi * right[5] ** 2 / r_max**2
        power = 1.0 + (
            adm_mass * (1.0 - 2.0 * omega**2)
            + omega * gauge_charge * right[5]
        ) / decay
        return np.asarray(
            (
                left[0] - center_mass,
                left[2] - leading * r0**ell,
                left[3] - ell * leading * r0 ** (ell - 1),
                left[5],
                right[1] - np.sqrt(max(exterior_f, 1e-14)),
                right[3] + (decay + power / r_max) * right[2],
                right[4] - right[5] / r_max,
            )
        )

    result = solve_bvp(
        fun,
        bc,
        radius,
        guess,
        p=parameter_guess,
        tol=tolerance,
        max_nodes=max_nodes,
        verbose=0,
    )
    if not result.success:
        raise RuntimeError(f"charged background BVP failed: {result.message}")
    dense_r = np.geomspace(r0, r_max, max(2700, 3 * points))
    fields = result.sol(dense_r)
    mass, alpha, psi, dpsi, potential, electric_flux = fields
    omega = _omega_from_parameter(float(result.p[0]))
    rhs = charged_rhs(dense_r, fields, omega, ell, gauge_charge)
    numerical = np.gradient(fields, dense_r, axis=1, edge_order=2)
    scales = np.maximum(np.max(np.abs(rhs), axis=1), 1.0)
    max_residual = float(np.max(np.abs(numerical - rhs) / scales[:, None]))
    gamma = 1.0 / np.sqrt(1.0 - 2.0 * mass / dense_r)
    local_frequency = omega - gauge_charge * potential
    noether_density = (
        multiplicity * local_frequency * psi**2 * gamma * dense_r**2 / alpha
    )
    noether_charge = float(np.trapezoid(noether_density, dense_r))
    adm_mass_profile = mass + 2.0 * np.pi * electric_flux**2 / dense_r
    target = 0.99 * adm_mass_profile[-1]
    r99 = float(np.interp(target, adm_mass_profile, dense_r))
    adm_mass = float(adm_mass_profile[-1])
    decay = np.sqrt(max(1.0 - omega**2, 1e-14))
    power = 1.0 + (
        adm_mass * (1.0 - 2.0 * omega**2)
        + omega * gauge_charge * electric_flux[-1]
    ) / decay
    tail_scale = max(float(np.max(np.abs(psi))), 1e-30)
    tail_residual = float(
        abs(dpsi[-1] + (decay + power / r_max) * psi[-1]) / tail_scale
    )
    return ChargedBackgroundSolution(
        ell=ell,
        a0=float(a0),
        gauge_charge=float(gauge_charge),
        omega=omega,
        r=dense_r,
        mass=mass,
        alpha=alpha,
        psi=psi,
        dpsi=dpsi,
        electric_potential=potential,
        electric_flux=electric_flux,
        noether_charge=noether_charge,
        r99=r99,
        compactness99=adm_mass / r99,
        max_ode_residual=max_residual,
        tail_residual=tail_residual,
        solver_status=result.status,
        solver_message=result.message,
    )


def continue_in_charge(
    neutral: BackgroundSolution,
    target_charge: float,
    *,
    step: float = 0.25,
    **kwargs: object,
) -> ChargedBackgroundSolution:
    charges = np.linspace(
        0.0,
        target_charge,
        max(2, int(np.ceil(target_charge / step)) + 1),
    )
    current = solve_charged_background(
        neutral.ell,
        neutral.a0,
        float(charges[0]),
        neutral_seed=neutral,
        **kwargs,
    )
    for charge in charges[1:]:
        current = solve_charged_background(
            neutral.ell,
            neutral.a0,
            float(charge),
            seed=current,
            **kwargs,
        )
    return current
