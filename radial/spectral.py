"""Independent Chebyshev generalized-eigenvalue radial solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig

from background.ell_boson_star import BackgroundSolution
from .center_series import center_d
from .coefficients import RadialBackground, pulsation_rhs
from .shooting import _node_count


@dataclass(frozen=True)
class SpectralRadialMode:
    sigma2: float
    r: np.ndarray
    delta_varphi: np.ndarray
    delta_l: np.ndarray
    physical_scalar: np.ndarray
    delta_lambda: np.ndarray
    node_count: int
    generalized_residual: float


def chebyshev_lobatto(points: int, r_min: float, r_max: float):
    if points < 8:
        raise ValueError("at least 8 Chebyshev-Lobatto points are required")
    index = np.arange(points)
    x = np.cos(np.pi * index / (points - 1))
    weights = np.ones(points)
    weights[[0, -1]] = 2.0
    weights *= (-1.0) ** index
    differences = x[:, None] - x[None, :]
    derivative = (weights[:, None] / weights[None, :]) / (differences + np.eye(points))
    derivative -= np.diag(np.sum(derivative, axis=1))
    # Reverse to ascending radius and map [-1,1] to [r_min,r_max].
    x = x[::-1]
    derivative = derivative[::-1, ::-1]
    radius = 0.5 * (r_max + r_min) + 0.5 * (r_max - r_min) * x
    derivative *= 2.0 / (r_max - r_min)
    return radius, derivative, derivative @ derivative


def assemble_generalized_problem(
    solution: BackgroundSolution,
    *,
    points: int = 80,
    epsilon: float = 1e-3,
    r_max: float = 40.0,
    background_representation: str = "hermite",
):
    background = RadialBackground(solution, representation=background_representation)
    radius, d1, d2 = chebyshev_lobatto(points, epsilon, r_max)
    zeros = np.zeros((4, points))

    def response(component: int, sigma2: float = 0.0):
        state = zeros.copy()
        state[component] = 1.0
        return pulsation_rhs(radius, state, sigma2, background)

    phi = response(0)
    dphi = response(1)
    delta_l = response(2)
    ddelta_l = response(3)
    phi_lambda = response(0, 1.0)[1] - phi[1]
    l_lambda = response(2, 1.0)[3] - delta_l[3]

    size = 2 * points
    matrix_a = np.zeros((size, size))
    matrix_b = np.zeros((size, size))
    interior = range(1, points - 1)
    for i in interior:
        # delta-varphi equation: D2 phi - RHS(lambda=0) = lambda q phi.
        matrix_a[i, :points] = d2[i] - dphi[1, i] * d1[i]
        matrix_a[i, i] -= phi[1, i]
        matrix_a[i, points:] = -ddelta_l[1, i] * d1[i]
        matrix_a[i, points + i] -= delta_l[1, i]
        matrix_b[i, i] = phi_lambda[i]

        # delta-L equation.
        row = points + i
        matrix_a[row, :points] = -dphi[3, i] * d1[i]
        matrix_a[row, i] -= phi[3, i]
        matrix_a[row, points:] = d2[i] - ddelta_l[3, i] * d1[i]
        matrix_a[row, points + i] -= delta_l[3, i]
        matrix_b[row, points + i] = l_lambda[i]

    # Regular center conditions, homogeneous in the arbitrary mode amplitude.
    center_value = np.zeros(points)
    center_value[0] = 1.0
    leading_value = center_value - 0.5 * epsilon * d1[0]
    matrix_a[0, :points] = -leading_value
    matrix_a[0, points:] = leading_value

    d0 = center_d(0.0, background)
    matrix_a[points, :points] = -2.0 * epsilon * d0 * leading_value
    matrix_a[points, points:] = d1[0]
    matrix_b[points, :points] = (
        -epsilon / (5.0 * background.alpha_c**2) * leading_value
    )

    # Published finite-radius outer conditions reduce to Dirichlet conditions
    # for both numerical variables because psi and omega are nonzero there.
    matrix_a[points - 1, points - 1] = 1.0
    matrix_a[2 * points - 1, 2 * points - 1] = 1.0
    return matrix_a, matrix_b, radius, background


def solve_radial_spectrum(
    solution: BackgroundSolution,
    *,
    points: int = 80,
    epsilon: float = 1e-3,
    r_max: float = 40.0,
    background_representation: str = "hermite",
    sigma2_min: float = -0.02,
    sigma2_max: float = 0.1,
    imaginary_tolerance: float = 1e-7,
) -> list[SpectralRadialMode]:
    matrix_a, matrix_b, radius, background = assemble_generalized_problem(
        solution,
        points=points,
        epsilon=epsilon,
        r_max=r_max,
        background_representation=background_representation,
    )
    eigenvalues, eigenvectors = eig(matrix_a, matrix_b, check_finite=True)
    modes: list[SpectralRadialMode] = []
    for index, value in enumerate(eigenvalues):
        if not np.isfinite(value) or abs(value.imag) > imaginary_tolerance:
            continue
        sigma2 = float(value.real)
        if not sigma2_min <= sigma2 <= sigma2_max:
            continue
        vector = eigenvectors[:, index]
        pivot = int(np.argmax(np.abs(vector)))
        vector = vector * np.exp(-1j * np.angle(vector[pivot]))
        if np.max(np.abs(vector.imag)) > 1e-6 * np.max(np.abs(vector.real)):
            continue
        vector = vector.real
        vector /= max(np.max(np.abs(vector)), 1e-300)
        phi = vector[:points]
        delta_l = vector[points:]
        _, _, _, psi, _, _, _, _, _, _, _ = background.arrays(radius)
        physical_scalar = psi * phi
        delta_lambda = 2.0 * background.kappa * psi**2 * delta_l
        residual_vector = matrix_a @ vector - sigma2 * (matrix_b @ vector)
        denominator = (
            np.linalg.norm(matrix_a @ vector)
            + abs(sigma2) * np.linalg.norm(matrix_b @ vector)
            + 1e-300
        )
        modes.append(
            SpectralRadialMode(
                sigma2=sigma2,
                r=radius,
                delta_varphi=phi,
                delta_l=delta_l,
                physical_scalar=physical_scalar,
                delta_lambda=delta_lambda,
                node_count=_node_count(delta_lambda),
                generalized_residual=float(np.linalg.norm(residual_vector) / denominator),
            )
        )
    return sorted(modes, key=lambda mode: mode.sigma2)

