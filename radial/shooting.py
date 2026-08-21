"""Two-parameter outward shooting for radial ell=1 pulsation modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from background.ell_boson_star import BackgroundSolution

from .center_series import center_series
from .coefficients import RadialBackground, pulsation_rhs


@dataclass(frozen=True)
class RadialMode:
    sigma2: float
    center_c: float
    epsilon: float
    r_max: float
    residual: np.ndarray
    residual_scale: np.ndarray
    r: np.ndarray
    state: np.ndarray
    physical_scalar: np.ndarray
    delta_lambda: np.ndarray
    node_count: int
    ivp_success: bool
    root_success: bool
    root_message: str


def integrate_mode(
    background: RadialBackground,
    sigma2: float,
    center_c: float,
    *,
    epsilon: float,
    r_max: float,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float | np.ndarray = 1e-12,
    dense_output: bool = False,
):
    y0 = center_series(epsilon, sigma2, center_c, background)
    return solve_ivp(
        lambda radius, state: pulsation_rhs(radius, state, sigma2, background),
        (epsilon, r_max),
        y0,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
    )


def outer_residual(state: np.ndarray, radius: float, background: RadialBackground) -> np.ndarray:
    point = background.point(radius)
    phi, _, delta_l, _ = state
    first = np.sqrt(point.gamma / point.alpha) * radius * point.psi * phi
    second = (
        np.sqrt(point.alpha / point.gamma)
        * (point.psi * delta_l - radius * point.dpsi * phi)
        / background.omega
    )
    return np.array((first, second))


def shooting_residual(
    scaled: np.ndarray,
    background: RadialBackground,
    *,
    epsilon: float,
    r_max: float,
    method: str,
    rtol: float,
    atol: float,
) -> np.ndarray:
    sigma2 = 1e-4 * float(scaled[0])
    center_c = 1e-2 * float(scaled[1])
    integration = integrate_mode(
        background,
        sigma2,
        center_c,
        epsilon=epsilon,
        r_max=r_max,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not integration.success:
        return np.array((1e6, 1e6))
    # Scaling removes the common small background-field amplitude without
    # changing the zeros of the published boundary conditions.
    residual = outer_residual(integration.y[:, -1], r_max, background)
    scale = max(abs(background.point(r_max).psi), 1e-30)
    return residual / scale


def eliminate_center_c(
    sigma2: float,
    background: RadialBackground,
    *,
    epsilon: float,
    r_max: float,
    method: str,
    rtol: float,
    atol: float,
    center_step: float = 1e-2,
    dense_output: bool = False,
):
    """Use ODE linearity to impose the first outer condition exactly.

    Integrations at two center coefficients form an affine basis. This avoids
    asking a two-dimensional root finder to resolve two almost-collinear,
    exponentially amplified residuals.
    """
    zero = integrate_mode(
        background,
        sigma2,
        0.0,
        epsilon=epsilon,
        r_max=r_max,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
    )
    stepped = integrate_mode(
        background,
        sigma2,
        center_step,
        epsilon=epsilon,
        r_max=r_max,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
    )
    if not zero.success or not stepped.success:
        raise RuntimeError("radial basis integration failed")
    residual_zero = outer_residual(zero.y[:, -1], r_max, background)
    residual_step = outer_residual(stepped.y[:, -1], r_max, background)
    residual_derivative = (residual_step - residual_zero) / center_step
    center_c = -residual_zero[0] / residual_derivative[0]
    residual = residual_zero + center_c * residual_derivative
    return center_c, residual, zero, stepped


def _node_count(values: np.ndarray) -> int:
    # The published outer conditions drive the physical perturbation to zero;
    # cancellation of exponentially large numerical variables can create a
    # tiny sign flip in that tail. Count only resolved interior nodes.
    threshold = 1e-4 * np.max(np.abs(values))
    significant = values[np.abs(values) > threshold]
    if significant.size < 2:
        return 0
    return int(np.count_nonzero(significant[1:] * significant[:-1] < 0.0))


def solve_radial_mode(
    solution: BackgroundSolution,
    *,
    sigma2_guess: float = 2.40e-4,
    center_c_guess: float = -2.78e-2,
    epsilon: float = 1e-5,
    r_max: float = 25.0,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    sigma2_bracket: tuple[float, float] = (2e-4, 3e-4),
) -> RadialMode:
    background = RadialBackground(solution)
    del center_c_guess  # center_c is eliminated exactly from the linear basis.

    def reduced(sigma2: float) -> float:
        _, residual, _, _ = eliminate_center_c(
            sigma2,
            background,
            epsilon=epsilon,
            r_max=r_max,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        return float(residual[1])

    lower, upper = sigma2_bracket
    if not lower < sigma2_guess < upper:
        raise ValueError("sigma2_guess must lie inside sigma2_bracket")
    sigma2, root_result = brentq(
        reduced,
        lower,
        upper,
        # Tighter values chase roundoff after cancellation of exponentially
        # dominant basis solutions; 5e-10 is below the published precision.
        xtol=5e-10,
        rtol=1e-10,
        maxiter=50,
        full_output=True,
        disp=False,
    )
    center_c, residual, zero, stepped = eliminate_center_c(
        sigma2,
        background,
        epsilon=epsilon,
        r_max=r_max,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    zero_residual = outer_residual(zero.y[:, -1], r_max, background)
    stepped_residual = outer_residual(stepped.y[:, -1], r_max, background)
    residual_scale = np.maximum(np.maximum(np.abs(zero_residual), np.abs(stepped_residual)), 1.0)
    radii = np.geomspace(epsilon, r_max, 3000)
    assert zero.sol is not None and stepped.sol is not None
    zero_state = zero.sol(radii)
    state_derivative = (stepped.sol(radii) - zero_state) / 1e-2
    state = zero_state + center_c * state_derivative
    psi = np.array([background.point(float(radius)).psi for radius in radii])
    physical_scalar = psi * state[0]
    delta_lambda = 2.0 * background.kappa * psi**2 * state[2]
    return RadialMode(
        sigma2=sigma2,
        center_c=center_c,
        epsilon=epsilon,
        r_max=r_max,
        residual=residual,
        residual_scale=residual_scale,
        r=radii,
        state=state,
        physical_scalar=physical_scalar,
        delta_lambda=delta_lambda,
        node_count=_node_count(delta_lambda),
        ivp_success=zero.success and stepped.success,
        root_success=bool(root_result.converged),
        root_message=f"brentq converged in {root_result.iterations} iterations",
    )
