"""Independent coefficient blocks for the radial generalized eigenproblem.

This module intentionally does not call ``pulsation_rhs``. Keeping a second,
explicit transcription of the published equations allows matrix-assembly tests
to detect common implementation errors between the BVP and spectral solvers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coefficients import RadialBackground


@dataclass(frozen=True)
class RadialCoefficientBlocks:
    phi_prime_in_phi_equation: np.ndarray
    delta_l_prime_in_phi_equation: np.ndarray
    phi_in_phi_equation: np.ndarray
    delta_l_in_phi_equation: np.ndarray
    sigma2_phi_in_phi_equation: np.ndarray
    phi_prime_in_l_equation: np.ndarray
    delta_l_prime_in_l_equation: np.ndarray
    phi_in_l_equation: np.ndarray
    delta_l_in_l_equation: np.ndarray
    sigma2_delta_l_in_l_equation: np.ndarray


def independent_coefficient_blocks(
    radius: np.ndarray, background: RadialBackground
) -> RadialCoefficientBlocks:
    """Evaluate a direct, independently transcribed Appendix coefficient set."""
    radius = np.asarray(radius, dtype=float)
    (
        _,
        alpha,
        gamma,
        psi,
        dpsi,
        _,
        _,
        _,
        log_alpha_prime,
        log_gamma_prime,
        log_gamma_prime_derivative,
    ) = background.arrays(radius)
    ell = background.ell
    kappa = background.kappa
    omega = background.omega
    gamma2 = gamma**2
    inv_gamma2 = 1.0 / gamma2
    mu_l2 = 1.0 + ell * (ell + 1) / radius**2
    psi_ratio = dpsi / psi

    phi_prime_in_phi = -(2.0 / radius + log_alpha_prime - log_gamma_prime)
    delta_l_prime_in_phi = -2.0 / radius
    phi_in_phi = 2.0 * gamma2 * (
        inv_gamma2 * psi_ratio**2
        + kappa * radius * mu_l2 * psi * dpsi
        + mu_l2
        + omega**2 / alpha**2
    )
    delta_l_in_phi = -2.0 * (
        1.0 / radius**2
        + 2.0 / radius * (psi_ratio - log_gamma_prime)
        + kappa
        * psi
        * dpsi
        * (log_alpha_prime - log_gamma_prime + psi_ratio + 1.0 / radius)
        - kappa * gamma2 * psi**2 * (mu_l2 - omega**2 / alpha**2)
    )

    phi_prime_in_l = -2.0 * (2.0 * psi_ratio - radius * gamma2 * mu_l2)
    delta_l_prime_in_l = -(
        4.0 * psi_ratio + 3.0 * (log_alpha_prime - log_gamma_prime)
    )
    phi_in_l = -2.0 * gamma2 * (
        2.0 * inv_gamma2 * psi_ratio**2
        - radius
        * mu_l2
        * (2.0 * psi_ratio + 2.0 * log_alpha_prime + log_gamma_prime)
        + ell * (ell + 1) / radius**2
    )
    delta_l_in_l = 2.0 * (
        2.0 * kappa * dpsi**2
        - (psi_ratio - 1.0 / radius + log_alpha_prime - log_gamma_prime) ** 2
        + 2.0 / radius**2
        - (4.0 * log_alpha_prime / radius - log_gamma_prime / radius)
        + log_gamma_prime_derivative
        - gamma2 * (mu_l2 - omega**2 / alpha**2)
    )
    sigma_coefficient = -gamma2 / alpha**2
    return RadialCoefficientBlocks(
        phi_prime_in_phi,
        delta_l_prime_in_phi,
        phi_in_phi,
        delta_l_in_phi,
        sigma_coefficient,
        phi_prime_in_l,
        delta_l_prime_in_l,
        phi_in_l,
        delta_l_in_l,
        sigma_coefficient,
    )


def independent_center_d0(background: RadialBackground) -> float:
    """Center coefficient at sigma2=0, independently of ``center_d``."""
    appendix_a0 = background.a0 / background.kappa
    return (
        1.0
        + background.omega**2 / background.alpha_c**2
        + 6.0 * background.kappa * appendix_a0**2
    ) / 5.0
