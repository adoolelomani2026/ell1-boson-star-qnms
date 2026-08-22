"""Background interface and coefficients for the radial pulsation equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from background.ell_boson_star import BackgroundSolution, _rhs


@dataclass(frozen=True)
class BackgroundPoint:
    r: float
    mass: float
    alpha: float
    gamma: float
    psi: float
    dpsi: float
    ddpsi: float
    u: float
    du: float
    log_alpha_prime: float
    log_gamma_prime: float
    log_gamma_prime_derivative: float


class RadialBackground:
    """Read-only interpolated background with analytic derivative identities."""

    def __init__(self, solution: BackgroundSolution, representation: str = "hermite"):
        self.solution = solution
        self.ell = solution.ell
        self.kappa = 2 * solution.ell + 1
        self.omega = solution.omega
        self.alpha_c = float(solution.alpha[0])
        self.a0 = solution.a0
        self.adm_mass = solution.adm_mass
        self.representation = representation
        self.r_min = float(solution.r[0])
        self.r_max = float(solution.r[-1])
        # Interpolate the regular field u=psi/r^ell once, then obtain both psi
        # and psi' from that same interpolant. This avoids the derivative
        # inconsistency caused by independently interpolating psi and psi'.
        regular_u = solution.psi / solution.r**solution.ell
        self._mass_spline = PchipInterpolator(solution.r, solution.mass, extrapolate=False)
        self._alpha_spline = PchipInterpolator(solution.r, solution.alpha, extrapolate=False)
        regular_du = (
            solution.dpsi / solution.r**solution.ell
            - solution.ell * solution.psi / solution.r ** (solution.ell + 1)
        )
        if representation == "hermite":
            self._u_spline = CubicHermiteSpline(
                solution.r, regular_u, regular_du, extrapolate=False
            )
        elif representation == "pchip":
            self._u_spline = PchipInterpolator(solution.r, regular_u, extrapolate=False)
        else:
            raise ValueError("representation must be 'hermite' or 'pchip'")
        self._du_spline = self._u_spline.derivative()

    def arrays(self, radius: np.ndarray):
        radius = np.asarray(radius, dtype=float)
        if np.any(radius < self.r_min) or np.any(radius > self.r_max):
            raise ValueError("radius outside background domain")
        mass = self._mass_spline(radius)
        alpha = self._alpha_spline(radius)
        u = self._u_spline(radius)
        du = self._du_spline(radius)
        psi = radius**self.ell * u
        dpsi = self.ell * radius ** (self.ell - 1) * u + radius**self.ell * du
        fields = np.vstack((mass, alpha, psi, dpsi))
        derivatives = _rhs(radius, fields, self.omega, self.ell)
        mass_prime, alpha_prime, _, ddpsi = derivatives
        gamma2 = 1.0 / (1.0 - 2.0 * mass / radius)
        gamma = np.sqrt(gamma2)
        log_alpha_prime = alpha_prime / alpha
        log_gamma_prime = gamma2 * (mass_prime / radius - mass / radius**2)
        gamma_prime = gamma * log_gamma_prime
        angular = self.ell * (self.ell + 1)
        mu_l2 = 1.0 + angular / radius**2
        log_gamma_prime_derivative = (
            (gamma2 - 1.0) / (gamma * radius) * (gamma / radius - gamma_prime)
            + self.kappa
            * radius
            * (
                dpsi * ddpsi
                - (self.omega**2 * alpha_prime / alpha**3 + angular / radius**3)
                * gamma2
                * psi**2
                + (mu_l2 + self.omega**2 / alpha**2)
                * (gamma * gamma_prime * psi**2 + gamma2 * psi * dpsi)
            )
        )
        return (
            mass,
            alpha,
            gamma,
            psi,
            dpsi,
            ddpsi,
            u,
            du,
            log_alpha_prime,
            log_gamma_prime,
            log_gamma_prime_derivative,
        )

    def point(self, radius: float) -> BackgroundPoint:
        if not self.r_min <= radius <= self.r_max:
            raise ValueError(f"radius {radius} outside background domain")
        values = self.arrays(np.array([radius]))
        mass, alpha, gamma, psi, dpsi, ddpsi, u, du, log_alpha_prime, log_gamma_prime, log_gamma_prime_derivative = (
            float(value[0]) for value in values
        )
        return BackgroundPoint(
            r=radius,
            mass=mass,
            alpha=alpha,
            gamma=gamma,
            psi=psi,
            dpsi=dpsi,
            ddpsi=ddpsi,
            u=u,
            du=du,
            log_alpha_prime=log_alpha_prime,
            log_gamma_prime=log_gamma_prime,
            log_gamma_prime_derivative=log_gamma_prime_derivative,
        )


def pulsation_rhs(
    radius: float, state: np.ndarray, sigma2: float, background: RadialBackground
) -> np.ndarray:
    """Eqs. (A2a)-(A2b) / (70a)-(70b) of arXiv:2103.15012."""
    scalar_input = np.ndim(radius) == 0
    radii = np.atleast_1d(np.asarray(radius, dtype=float))
    values = np.asarray(state, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    phi, dphi, delta_l, ddelta_l = values
    _, alpha, gamma, psi, dpsi, _, _, _, la, lg, lg_derivative = background.arrays(radii)
    ell = background.ell
    kappa = background.kappa
    omega = background.omega
    gamma2 = gamma**2
    inv_gamma2 = 1.0 / gamma2
    mu_l2 = 1.0 + ell * (ell + 1) / radii**2
    psi_ratio = dpsi / psi

    phi_coefficient = 2.0 * gamma2 * (
        inv_gamma2 * psi_ratio**2
        + kappa * radii * mu_l2 * psi * dpsi
        + mu_l2
        + (2.0 * omega**2 - sigma2) / (2.0 * alpha**2)
    )
    l_in_phi = -2.0 * (
        1.0 / radii**2
        + 2.0 / radii * (psi_ratio - lg)
        + kappa * psi * dpsi * (la - lg + psi_ratio + 1.0 / radii)
        - kappa * gamma2 * psi**2 * (mu_l2 - omega**2 / alpha**2)
    )
    ddphi = (
        -(2.0 / radii + la - lg) * dphi
        - 2.0 / radii * ddelta_l
        + phi_coefficient * phi
        + l_in_phi * delta_l
    )

    dphi_in_l = -2.0 * (2.0 * psi_ratio - radii * gamma2 * mu_l2)
    dl_derivative = -(4.0 * psi_ratio + 3.0 * (la - lg))
    phi_in_l = -2.0 * gamma2 * (
        2.0 * inv_gamma2 * psi_ratio**2
        - radii * mu_l2 * (2.0 * psi_ratio + 2.0 * la + lg)
        + ell * (ell + 1) / radii**2
    )
    l_coefficient = 2.0 * (
        2.0 * kappa * dpsi**2
        - (psi_ratio - 1.0 / radii + la - lg) ** 2
        + 2.0 / radii**2
        - (4.0 * la / radii - lg / radii)
        + lg_derivative
        - gamma2 * (mu_l2 - (2.0 * omega**2 - sigma2) / (2.0 * alpha**2))
    )
    dddelta_l = (
        dphi_in_l * dphi
        + dl_derivative * ddelta_l
        + phi_in_l * phi
        + l_coefficient * delta_l
    )
    result = np.vstack((dphi, ddphi, ddelta_l, dddelta_l))
    return result[:, 0] if scalar_input else result
