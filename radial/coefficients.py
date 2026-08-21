"""Background interface and coefficients for the radial pulsation equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

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

    def __init__(self, solution: BackgroundSolution):
        self.solution = solution
        self.ell = solution.ell
        self.kappa = 2 * solution.ell + 1
        self.omega = solution.omega
        self.alpha_c = float(solution.alpha[0])
        self.a0 = solution.a0
        self.adm_mass = solution.adm_mass
        self.r_min = float(solution.r[0])
        self.r_max = float(solution.r[-1])
        self._splines = tuple(
            PchipInterpolator(solution.r, field, extrapolate=False)
            for field in (solution.mass, solution.alpha, solution.psi, solution.dpsi)
        )

    def point(self, radius: float) -> BackgroundPoint:
        if not self.r_min <= radius <= self.r_max:
            raise ValueError(f"radius {radius} outside background domain")
        mass, alpha, psi, dpsi = (float(spline(radius)) for spline in self._splines)
        fields = np.array([[mass], [alpha], [psi], [dpsi]])
        derivatives = _rhs(np.array([radius]), fields, self.omega, self.ell)[:, 0]
        mass_prime, alpha_prime, _, ddpsi = (float(value) for value in derivatives)
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
        return BackgroundPoint(
            r=radius,
            mass=mass,
            alpha=alpha,
            gamma=gamma,
            psi=psi,
            dpsi=dpsi,
            ddpsi=ddpsi,
            u=psi / radius**self.ell,
            du=dpsi / radius**self.ell - self.ell * psi / radius ** (self.ell + 1),
            log_alpha_prime=log_alpha_prime,
            log_gamma_prime=log_gamma_prime,
            log_gamma_prime_derivative=log_gamma_prime_derivative,
        )


def pulsation_rhs(
    radius: float, state: np.ndarray, sigma2: float, background: RadialBackground
) -> np.ndarray:
    """Eqs. (A2a)-(A2b) / (70a)-(70b) of arXiv:2103.15012."""
    phi, dphi, delta_l, ddelta_l = state
    point = background.point(radius)
    ell = background.ell
    kappa = background.kappa
    omega = background.omega
    alpha = point.alpha
    gamma2 = point.gamma**2
    inv_gamma2 = 1.0 / gamma2
    mu_l2 = 1.0 + ell * (ell + 1) / radius**2
    psi_ratio = point.dpsi / point.psi
    la = point.log_alpha_prime
    lg = point.log_gamma_prime

    phi_coefficient = 2.0 * gamma2 * (
        inv_gamma2 * psi_ratio**2
        + kappa * radius * mu_l2 * point.psi * point.dpsi
        + mu_l2
        + (2.0 * omega**2 - sigma2) / (2.0 * alpha**2)
    )
    l_in_phi = -2.0 * (
        1.0 / radius**2
        + 2.0 / radius * (psi_ratio - lg)
        + kappa * point.psi * point.dpsi * (la - lg + psi_ratio + 1.0 / radius)
        - kappa * gamma2 * point.psi**2 * (mu_l2 - omega**2 / alpha**2)
    )
    ddphi = (
        -(2.0 / radius + la - lg) * dphi
        - 2.0 / radius * ddelta_l
        + phi_coefficient * phi
        + l_in_phi * delta_l
    )

    dphi_in_l = -2.0 * (2.0 * psi_ratio - radius * gamma2 * mu_l2)
    dl_derivative = -(4.0 * psi_ratio + 3.0 * (la - lg))
    phi_in_l = -2.0 * gamma2 * (
        2.0 * inv_gamma2 * psi_ratio**2
        - radius * mu_l2 * (2.0 * psi_ratio + 2.0 * la + lg)
        + ell * (ell + 1) / radius**2
    )
    l_coefficient = 2.0 * (
        2.0 * kappa * point.dpsi**2
        - (psi_ratio - 1.0 / radius + la - lg) ** 2
        + 2.0 / radius**2
        - (4.0 * la / radius - lg / radius)
        + point.log_gamma_prime_derivative
        - gamma2 * (mu_l2 - (2.0 * omega**2 - sigma2) / (2.0 * alpha**2))
    )
    dddelta_l = (
        dphi_in_l * dphi
        + dl_derivative * ddelta_l
        + phi_in_l * phi
        + l_coefficient * delta_l
    )
    return np.array((dphi, ddphi, ddelta_l, dddelta_l))
