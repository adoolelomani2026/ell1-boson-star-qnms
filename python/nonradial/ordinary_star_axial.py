"""Odd-parity QNMs of an ordinary (ell=0 background) mini boson star.

This is an independent control for the stellar open-boundary machinery.  The
scalar field has even parity on a spherical background, so the axial channel
reduces to the matter-filled Regge--Wheeler equation of Macedo et al.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from radial.coefficients import RadialBackground


def stellar_regge_wheeler_potential(
    radius: float, background: RadialBackground, *, multipole: int = 2
) -> float:
    """Return the ordinary-star axial potential in repository normalization.

    For the free spherical scalar, ``-kappa(p_rad-rho)/2 = psi**2`` after
    converting the physical field to the repository's rescaled field.
    """

    if background.ell != 0:
        raise ValueError("ordinary axial control requires an ell=0 background")
    point = background.point(radius)
    return float(
        point.alpha**2
        * (
            multipole * (multipole + 1.0) / radius**2
            - 6.0 * point.mass / radius**3
            + point.psi**2
        )
    )


def stellar_outgoing_log_derivative(
    frequency: complex,
    radius: float,
    *,
    mass: float,
    multipole: int = 2,
    leaver_terms: int = 500,
) -> complex:
    """Return outgoing ``d ln(Psi)/dr`` from the R2-centered recurrence."""

    if radius <= 2.0 * mass:
        raise ValueError("radius must exceed 2M")
    if leaver_terms < 30:
        raise ValueError("leaver_terms must be at least 30")
    alpha = np.zeros(leaver_terms + 2, dtype=complex)
    beta = np.zeros_like(alpha)
    gamma = np.zeros_like(alpha)
    delta = np.zeros_like(alpha)
    for n in range(1, leaver_terms + 2):
        alpha[n] = n * (n + 1) * (radius - 2.0 * mass)
        # These signs follow by direct substitution into the RW equation for
        # exp(+i sigma r)(r-2M)^(2 i M sigma) sum a_n z^n.  Some OCR copies of
        # the published formula lose the overall signs of beta and gamma.
        beta[n] = 2.0 * n * (
            3.0 * mass * n - radius * n
            + 1.0j * radius**2 * frequency
        )
        gamma[n] = -(
            6.0 * mass * ((n - 1) * n - 1)
            + (1 + multipole - n) * (multipole + n) * radius
        )
        delta[n] = 2.0 * mass * (3 - n) * (1 + n)

    reduced_beta = beta.copy()
    reduced_gamma = gamma.copy()
    for n in range(2, leaver_terms + 2):
        elimination = delta[n] / reduced_gamma[n - 1]
        reduced_beta[n] -= elimination * alpha[n - 1]
        reduced_gamma[n] -= elimination * reduced_beta[n - 1]

    ratio = 0.0j
    for n in range(leaver_terms, 0, -1):
        ratio = -reduced_gamma[n] / (reduced_beta[n] + alpha[n] * ratio)
    prefactor_log_derivative = (
        1.0j * frequency * radius / (radius - 2.0 * mass)
    )
    return complex(prefactor_log_derivative + ratio / radius)


def stellar_axial_mismatch(
    frequency: complex,
    background: RadialBackground,
    *,
    multipole: int = 2,
    r_start: float = 2.0e-4,
    r_match: float | None = None,
    leaver_terms: int = 500,
    rtol: float = 2.0e-10,
    atol: float = 2.0e-12,
) -> complex:
    """Match the regular stellar solution to a Leaver-minimal exterior.

    The returned Wronskian-like residual is invariant under normalization of
    the interior wavefunction.  The exterior replacement is assessed by
    varying ``r_match`` in the benchmark experiment.
    """

    if r_match is None:
        r_match = 1.4 * background.solution.r99
    if not background.r_min <= r_start < r_match <= background.r_max:
        raise ValueError("require r_min <= r_start < r_match <= r_max")
    center = background.point(r_start)
    exponent = multipole + 1
    initial = np.array(
        (r_start**exponent, exponent * r_start ** (exponent - 1)),
        dtype=complex,
    )

    def rhs(radius: float, state: np.ndarray) -> np.ndarray:
        point = background.point(radius)
        f = point.alpha / point.gamma
        f_log_prime = point.log_alpha_prime - point.log_gamma_prime
        potential = stellar_regge_wheeler_potential(
            radius, background, multipole=multipole
        )
        wave, derivative = state
        return np.array(
            (
                derivative,
                -f_log_prime * derivative
                - (frequency**2 - potential) * wave / f**2,
            ),
            dtype=complex,
        )

    result = solve_ivp(
        rhs,
        (r_start, r_match),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"stellar Regge--Wheeler integration failed: {result.message}")
    wave, derivative = result.y[:, -1]
    if abs(wave) < 1.0e-300:
        raise FloatingPointError("interior wave vanished at the matching radius")

    exterior = stellar_outgoing_log_derivative(
        frequency,
        r_match,
        mass=background.adm_mass,
        multipole=multipole,
        leaver_terms=leaver_terms,
    )
    return complex(derivative / wave - exterior)
