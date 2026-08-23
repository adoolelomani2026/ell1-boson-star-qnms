"""Nonlinear two-point BVP certification for radial pulsation eigenmodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

from background.ell_boson_star import BackgroundSolution
from .center_series import center_d, center_series
from .coefficients import RadialBackground, pulsation_rhs
from .mode_tracking import node_count
from .shooting import integrate_mode, outer_residual


@dataclass(frozen=True)
class BvpRadialMode:
    sigma2: float
    center_c: float
    epsilon: float
    r_max: float
    r: np.ndarray
    state: np.ndarray
    physical_scalar: np.ndarray
    delta_lambda: np.ndarray
    physical_boundary_residual: np.ndarray
    max_scipy_interval_rms_relative_residual: float
    max_dense_pointwise_relative_residual: float
    node_count: int
    success: bool
    message: str
    nodes_used: int


def solve_radial_bvp(
    solution: BackgroundSolution,
    *,
    sigma2_guess: float = 2.40e-4,
    center_c_guess: float = -2.78e-2,
    epsilon: float = 1e-3,
    r_max: float = 25.0,
    points: int = 500,
    tolerance: float = 1e-6,
    max_nodes: int = 30000,
    background_representation: str = "hermite",
) -> BvpRadialMode:
    """Solve the mode globally while imposing both physical outer conditions."""
    background = RadialBackground(solution, representation=background_representation)
    mesh = np.geomspace(epsilon, r_max, points)

    # An outward IVP supplies only an initial Newton guess. Certification comes
    # from the global collocation equations and directly imposed endpoint BCs.
    initial_ivp = integrate_mode(
        background,
        sigma2_guess,
        center_c_guess,
        epsilon=epsilon,
        r_max=r_max,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )
    if initial_ivp.success and initial_ivp.sol is not None:
        guess = initial_ivp.sol(mesh)
    else:
        start = center_series(epsilon, sigma2_guess, center_c_guess, background)
        guess = np.repeat(start[:, None], points, axis=1)

    def equations(radius: np.ndarray, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        return pulsation_rhs(radius, state, float(parameter[0]), background)

    def boundary(left: np.ndarray, right: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        sigma2 = float(parameter[0])
        d_coefficient = center_d(sigma2, background)
        outer = outer_residual(right, r_max, background)
        return np.array(
            (
                left[0] - 0.5 * epsilon * left[1] - 1.0,
                left[2] - 0.5 * epsilon * left[3] - 1.0,
                left[3] - 2.0 * d_coefficient * epsilon,
                outer[0],
                outer[1],
            )
        )

    result = solve_bvp(
        equations,
        boundary,
        mesh,
        guess,
        p=np.array([sigma2_guess]),
        tol=tolerance,
        max_nodes=max_nodes,
        verbose=0,
    )
    radii = np.geomspace(epsilon, r_max, 3000)
    state = result.sol(radii)
    state_derivative = result.sol(radii, 1)
    equation_values = pulsation_rhs(radii, state, float(result.p[0]), background)
    pointwise_relative = np.abs(state_derivative - equation_values) / (1.0 + np.abs(equation_values))
    _, _, _, psi, _, _, _, _, _, _, _ = background.arrays(radii)
    physical_scalar = psi * state[0]
    delta_lambda = 2.0 * background.kappa * psi**2 * state[2]
    physical_residual = outer_residual(state[:, -1], r_max, background)
    center_c = float(state[1, 0] / (2.0 * epsilon))
    max_rms_residual = (
        float(np.max(result.rms_residuals)) if result.rms_residuals.size else float("nan")
    )
    max_pointwise_residual = float(np.max(pointwise_relative))
    return BvpRadialMode(
        sigma2=float(result.p[0]),
        center_c=center_c,
        epsilon=epsilon,
        r_max=r_max,
        r=radii,
        state=state,
        physical_scalar=physical_scalar,
        delta_lambda=delta_lambda,
        physical_boundary_residual=physical_residual,
        max_scipy_interval_rms_relative_residual=max_rms_residual,
        max_dense_pointwise_relative_residual=max_pointwise_residual,
        node_count=node_count(radii, delta_lambda),
        success=bool(result.success),
        message=result.message,
        nodes_used=result.x.size,
    )
