"""Independent method-of-lines form of the neutral axial EKG equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_ekg import KAPPA_V, L, MU


@dataclass(frozen=True)
class AxialGridBackground:
    r: np.ndarray
    alpha: np.ndarray
    gamma: np.ndarray
    log_alpha_prime: np.ndarray
    log_gamma_prime: np.ndarray
    alpha_second: np.ndarray
    field: np.ndarray
    dfield: np.ndarray
    k0: np.ndarray
    omega: float
    mass: float


def grid_background(solution: BackgroundSolution, radii: np.ndarray) -> AxialGridBackground:
    """Interpolate the star and continue it by exact Schwarzschild vacuum."""

    from radial.coefficients import RadialBackground

    r = np.asarray(radii, dtype=float)
    if np.any(r <= 0) or np.any(np.diff(r) <= 0):
        raise ValueError("radii must be strictly increasing and positive")
    radial = RadialBackground(solution, geometry_representation="hermite")
    inside = r <= radial.r_max
    alpha = np.empty_like(r)
    gamma = np.empty_like(r)
    la = np.empty_like(r)
    lg = np.empty_like(r)
    alpha_second = np.empty_like(r)
    field = np.zeros_like(r)
    dfield = np.zeros_like(r)
    k0 = np.zeros_like(r)
    if np.any(inside):
        values = radial.arrays(r[inside])
        _, a, g, psi, dpsi, _, _, _, lai, lgi, _ = values
        f = np.sqrt(3.0) * psi
        df = np.sqrt(3.0) * dpsi
        alpha[inside] = a
        gamma[inside] = g
        la[inside] = lai
        lg[inside] = lgi
        alpha_second[inside] = radial.equilibrium_lapse_second_derivative(r[inside])
        field[inside] = f
        dfield[inside] = df
        k0[inside] = (
            -solution.omega**2 * f**2 / a**2
            + df**2 / g**2
            + 2.0 * f**2 / r[inside] ** 2
            + MU**2 * f**2
        ) / (4.0 * np.pi)
    if np.any(~inside):
        ro = r[~inside]
        schwarzschild = 1.0 - 2.0 * solution.adm_mass / ro
        ao = np.sqrt(schwarzschild)
        lao = solution.adm_mass / (ro**2 * schwarzschild)
        alpha[~inside] = ao
        gamma[~inside] = 1.0 / ao
        la[~inside] = lao
        lg[~inside] = -lao
        alpha_second[~inside] = -2.0 * solution.adm_mass / (ro**3 * ao) - (
            solution.adm_mass**2 / (ro**4 * ao**3)
        )
    return AxialGridBackground(
        r=r,
        alpha=alpha,
        gamma=gamma,
        log_alpha_prime=la,
        log_gamma_prime=lg,
        alpha_second=alpha_second,
        field=field,
        dfield=dfield,
        k0=k0,
        omega=solution.omega,
        mass=solution.adm_mass,
    )


def derivative_4(values: np.ndarray, spacing: float) -> np.ndarray:
    """Fourth-order centered first derivative with one-sided edge stencils."""

    values = np.asarray(values)
    result = np.empty_like(values)
    result[2:-2] = (
        values[:-4] - 8.0 * values[1:-3] + 8.0 * values[3:-1] - values[4:]
    ) / (12.0 * spacing)
    result[0] = (-25 * values[0] + 48 * values[1] - 36 * values[2] + 16 * values[3] - 3 * values[4]) / (12 * spacing)
    result[1] = (-3 * values[0] - 10 * values[1] + 18 * values[2] - 6 * values[3] + values[4]) / (12 * spacing)
    result[-2] = -(-3 * values[-1] - 10 * values[-2] + 18 * values[-3] - 6 * values[-4] + values[-5]) / (12 * spacing)
    result[-1] = -(-25 * values[-1] + 48 * values[-2] - 36 * values[-3] + 16 * values[-4] - 3 * values[-5]) / (12 * spacing)
    return result


def second_derivative_4(values: np.ndarray, spacing: float) -> np.ndarray:
    """Fourth-order centered second derivative; second order at two edges."""

    values = np.asarray(values)
    result = np.empty_like(values)
    result[2:-2] = (
        -values[:-4]
        + 16.0 * values[1:-3]
        - 30.0 * values[2:-2]
        + 16.0 * values[3:-1]
        - values[4:]
    ) / (12.0 * spacing**2)
    result[:2] = np.gradient(np.gradient(values, spacing, edge_order=2), spacing, edge_order=2)[:2]
    result[-2:] = np.gradient(np.gradient(values, spacing, edge_order=2), spacing, edge_order=2)[-2:]
    return result


def ko_dissipation(values: np.ndarray, spacing: float, strength: float) -> np.ndarray:
    """Sixth-derivative Kreiss--Oliger filter for fourth-order differences."""

    values = np.asarray(values)
    result = np.zeros_like(values)
    result[..., 3:-3] = strength / (64.0 * spacing) * (
        values[..., :-6]
        - 6.0 * values[..., 1:-5]
        + 15.0 * values[..., 2:-4]
        - 20.0 * values[..., 3:-3]
        + 15.0 * values[..., 4:-2]
        - 6.0 * values[..., 5:-1]
        + values[..., 6:]
    )
    return result


def evolution_rhs(
    state: np.ndarray,
    background: AxialGridBackground,
    spacing: float,
    *,
    sponge: np.ndarray | None = None,
    dissipation: float = 0.5,
) -> np.ndarray:
    """Return d/dt(h0,h1,k,p,pi,q,chi) on a uniform radial grid."""

    y = np.asarray(state, dtype=complex)
    if y.shape != (7, background.r.size):
        raise ValueError("state must have shape (7, number_of_radii)")
    h0, h1, k, p, pi, q, chi = y
    r = background.r
    alpha = background.alpha
    gamma = background.gamma
    alpha2 = alpha**2
    gamma2 = gamma**2
    la = background.log_alpha_prime
    lg = background.log_gamma_prime
    field = background.field
    dfield = background.dfield
    c = 1.0 / (2.0 * np.sqrt(np.pi))

    h1_r = derivative_4(h1, spacing)
    p_r = derivative_4(p, spacing)
    q_r = derivative_4(q, spacing)
    p_rr = second_derivative_4(p, spacing)
    q_rr = second_derivative_4(q, spacing)
    h0_t = alpha2 / gamma2 * (
        h1_r + (la - lg) * h1 - 8.0j * np.pi * c * gamma2 * field * (p + q)
    )
    h0_t_r = derivative_4(h0_t, spacing)
    geometric_h1 = (
        background.alpha_second / (alpha * gamma2)
        - la * lg / gamma2
        - lg / (r * gamma2)
        + la / (r * gamma2)
        + 2.0 / r**2
    )
    radial_bilinear = dfield * (p + q) - field * (p_r + q_r)
    stress_r = -0.5j * c * radial_bilinear - 0.5 * background.k0 * h1
    k_t = (
        h0_t_r
        - 2.0 * h0_t / r
        - 2.0 * alpha2 * geometric_h1 * h1
        + 16.0 * np.pi * alpha2 * stress_r
    )
    first_derivative = 2.0 / r + la - lg
    radial_metric = field * h1_r + (2.0 * dfield + field * (la - lg)) * h1
    common_p = (
        alpha2 / gamma2 * (p_rr + first_derivative * p_r)
        + (background.omega**2 - alpha2 * (MU**2 + L * (L + 1.0) / r**2)) * p
    )
    common_q = (
        alpha2 / gamma2 * (q_rr + first_derivative * q_r)
        + (background.omega**2 - alpha2 * (MU**2 + L * (L + 1.0) / r**2)) * q
    )
    p_metric = KAPPA_V / r**2 * (
        field * (2.0j * background.omega * h0 + h0_t)
        - alpha2 / gamma2 * radial_metric
    )
    q_metric = KAPPA_V / r**2 * (
        field * (h0_t - 2.0j * background.omega * h0)
        - alpha2 / gamma2 * radial_metric
    )
    pi_t = common_p + p_metric - 2.0j * background.omega * pi
    chi_t = common_q + q_metric + 2.0j * background.omega * chi
    result = np.array((h0_t, k, k_t, pi, pi_t, chi, chi_t), dtype=complex)
    if dissipation:
        result += ko_dissipation(y, spacing, dissipation)
    if sponge is not None:
        damping = np.asarray(sponge, dtype=float)
        if damping.shape != r.shape:
            raise ValueError("sponge must match the radial grid")
        result -= damping[None, :] * y
    return result


def rk4_step(state, dt, rhs):
    """One explicit fourth-order Runge--Kutta step."""

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
