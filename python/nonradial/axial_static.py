"""Stationary J=L=2 odd-parity response of a neutral ell=1 boson star."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from nonradial.axial_ekg import (
    G_NEWTON,
    L,
    MU,
    background_axial_scalar,
)
from radial.coefficients import RadialBackground


C_VECTOR = 1.0 / (2.0 * np.sqrt(np.pi))


def static_rhs(
    radius: float, state: np.ndarray, background: RadialBackground
) -> np.ndarray:
    """Return derivatives for (h0,h0',d,d'), d=u_plus-u_minus.

    The stationary regular sector has h1=0 and u_plus+u_minus=0.  The tA
    Einstein equation and the difference of the two KG equations then close
    without the 1/sigma reduction used by the dynamical system.
    """

    values = np.asarray(state, dtype=complex)
    if values.ndim not in (1, 2) or values.shape[0] != 4:
        raise ValueError("state must have shape (4,) or (4,n)")
    h0, dh0, difference, ddifference = values
    point, field, _, alpha_second, k0 = background_axial_scalar(
        background, radius
    )
    alpha = point.alpha
    gamma = point.gamma
    alpha2 = alpha**2
    gamma2 = gamma**2
    la = point.log_alpha_prime
    lg = point.log_gamma_prime
    omega = background.omega

    d2h0 = (
        (la + lg) * dh0
        + 2.0 * (alpha_second / alpha - la * lg) * h0
        - 4.0 * lg * h0 / radius
        + (4.0 * gamma2 + 2.0) * h0 / radius**2
        + 16.0
        * np.pi
        * G_NEWTON
        * gamma2
        * C_VECTOR
        * field
        * omega
        * difference
        + 8.0 * np.pi * G_NEWTON * gamma2 * k0 * h0
    )
    first_derivative = 2.0 / radius + la - lg
    potential = omega**2 / alpha2 - MU**2 - L * (L + 1.0) / radius**2
    d2difference = (
        -first_derivative * ddifference
        - gamma2 * potential * difference
        - 4.0
        * C_VECTOR
        * omega
        * gamma2
        * field
        * h0
        / (alpha2 * radius**2)
    )
    return np.asarray((dh0, d2h0, ddifference, d2difference), dtype=complex)


def static_center_basis(radius: float) -> np.ndarray:
    """Leading regular metric-led and scalar-led stationary solutions."""

    basis = np.zeros((4, 2), dtype=complex)
    basis[:, 0] = (radius**3, 3.0 * radius**2, 0.0, 0.0)
    basis[:, 1] = (0.0, 0.0, radius**2, 2.0 * radius)
    return basis


def integrate_static_regular_basis(
    background: RadialBackground,
    *,
    r_start: float | None = None,
    r_end: float = 14.0,
    rtol: float = 2e-10,
    atol: float = 2e-12,
) -> np.ndarray:
    left = max(background.r_min, 2e-4) if r_start is None else r_start
    initial = static_center_basis(left).reshape(-1, order="F")

    def rhs(radius, flattened):
        matrix = flattened.reshape((4, 2), order="F")
        return static_rhs(radius, matrix, background).reshape(-1, order="F")

    result = solve_ivp(
        rhs,
        (left, r_end),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"static interior integration failed: {result.message}")
    return result.y[:, -1].reshape((4, 2), order="F")


def _vacuum_metric_basis(
    radius: float, mass: float, *, r_far: float = 500.0
) -> tuple[np.ndarray, np.ndarray]:
    """Exact growing and asymptotically normalized decaying vacuum columns."""

    growing = np.asarray(
        (radius**2 * (radius - 2.0 * mass), 3.0 * radius**2 - 4.0 * mass * radius),
        dtype=complex,
    )
    powers = np.arange(2, 8, dtype=float)
    coefficients = np.asarray(
        (
            1.0,
            4.0 * mass / 3.0,
            40.0 * mass**2 / 21.0,
            20.0 * mass**3 / 7.0,
            40.0 * mass**4 / 9.0,
            64.0 * mass**5 / 9.0,
        )
    )
    value = np.sum(coefficients * r_far ** (-powers))
    derivative = np.sum(-powers * coefficients * r_far ** (-powers - 1.0))

    def vacuum_rhs(r, state):
        coefficient = (6.0 * r - 4.0 * mass) / (r**2 * (r - 2.0 * mass))
        return np.asarray((state[1], coefficient * state[0]), dtype=complex)

    result = solve_ivp(
        vacuum_rhs,
        (r_far, radius),
        np.asarray((value, derivative), dtype=complex),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not result.success:
        raise RuntimeError(f"static vacuum metric integration failed: {result.message}")
    decaying = result.y[:, -1]
    physical_coefficient_per_normalized_amplitude = np.asarray(
        (1.0 / growing[0], 1.0 / decaying[0]), dtype=complex
    )
    normalized = np.column_stack(
        (growing / growing[0], decaying / decaying[0])
    )
    return normalized, physical_coefficient_per_normalized_amplitude


def _vacuum_scalar_log_derivative(
    radius: float, omega: float, mass: float, *, r_far: float = 500.0
) -> complex:
    decay = np.sqrt(MU**2 - omega**2)
    f_far = 1.0 - 2.0 * mass / r_far
    rstar_far = r_far + 2.0 * mass * np.log(r_far / (2.0 * mass) - 1.0)
    k = 1j * decay
    z = k * rstar_far
    polynomial = 1.0 + 3.0j / z - 3.0 / z**2
    polynomial_prime = -3.0j / z**2 + 6.0 / z**3
    q = k * (1.0j - 1.0 / z + polynomial_prime / polynomial) / f_far

    def riccati_rhs(r, value):
        f = 1.0 - 2.0 * mass / r
        la = mass / (r**2 * f)
        first = 2.0 / r + 2.0 * la
        potential = (omega**2 / f**2) - (MU**2 + 6.0 / r**2) / f
        return -value**2 - first * value - potential

    result = solve_ivp(
        riccati_rhs,
        (r_far, radius),
        np.asarray((q,), dtype=complex),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not result.success:
        raise RuntimeError(f"static vacuum scalar integration failed: {result.message}")
    return complex(result.y[0, -1])


def integrate_static_exterior_basis(
    background: RadialBackground,
    *,
    r_match: float = 14.0,
    r_end: float = 35.0,
    rtol: float = 2e-10,
    atol: float = 2e-12,
) -> tuple[np.ndarray, np.ndarray]:
    metric, metric_normalizations = _vacuum_metric_basis(
        r_end, background.adm_mass
    )
    scalar_q = _vacuum_scalar_log_derivative(
        r_end, background.omega, background.adm_mass
    )
    initial = np.zeros((4, 3), dtype=complex)
    initial[:2, :2] = metric
    initial[2:, 2] = (1.0, scalar_q)

    def rhs(radius, flattened):
        matrix = flattened.reshape((4, 3), order="F")
        return static_rhs(radius, matrix, background).reshape(-1, order="F")

    result = solve_ivp(
        rhs,
        (r_end, r_match),
        initial.reshape(-1, order="F"),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"static exterior integration failed: {result.message}")
    return (
        result.y[:, -1].reshape((4, 3), order="F"),
        metric_normalizations,
    )


def static_tidal_response(
    background: RadialBackground,
    *,
    r_match: float = 14.0,
    r_end: float = 35.0,
    rtol: float = 2e-10,
    atol: float = 2e-12,
) -> dict[str, float]:
    """Match the regular interior to tidal, response, and scalar exteriors."""

    interior = integrate_static_regular_basis(
        background, r_end=r_match, rtol=rtol, atol=atol
    )
    exterior, metric_normalizations = integrate_static_exterior_basis(
        background,
        r_match=r_match,
        r_end=r_end,
        rtol=rtol,
        atol=atol,
    )
    match = np.column_stack((interior, -exterior))
    row_norms = np.maximum(np.linalg.norm(match, axis=1), 1e-300)
    row_scaled = match / row_norms[:, None]
    column_norms = np.maximum(np.linalg.norm(row_scaled, axis=0), 1e-300)
    normalized = row_scaled / column_norms
    _, singular_values, vh = np.linalg.svd(normalized)
    coefficients = vh.conj().T[:, -1] / column_norms
    coefficients /= max(np.linalg.norm(coefficients[:2]), 1e-300)
    exterior_coefficients = coefficients[2:]
    matching_residual = np.linalg.norm(match @ coefficients) / max(
        np.linalg.norm(interior @ coefficients[:2]), 1e-300
    )
    growing = exterior_coefficients[0] * metric_normalizations[0]
    response = exterior_coefficients[1] * metric_normalizations[1]
    response_ratio = response / growing
    return {
        "response_B_over_A": float(response_ratio.real),
        "response_imaginary_residual": float(abs(response_ratio.imag)),
        "dimensionless_B_over_A_M5": float(response_ratio.real / background.adm_mass**5),
        "scalar_tail_over_tidal_A": float(
            abs(exterior_coefficients[2] / growing)
        ),
        "normalized_matching_residual": float(matching_residual),
        "normalized_minimum_singular_value": float(singular_values[-1]),
    }
