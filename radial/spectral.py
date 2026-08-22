"""Independent Chebyshev generalized-eigenvalue radial solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig

from background.ell_boson_star import BackgroundSolution
from .coefficients import RadialBackground
from .independent_operator import independent_center_d0, independent_coefficient_blocks
from .mode_tracking import node_count


@dataclass(frozen=True)
class SpectralRadialMode:
    sigma2: float
    r: np.ndarray
    delta_varphi: np.ndarray
    delta_l: np.ndarray
    physical_scalar: np.ndarray
    delta_lambda: np.ndarray
    node_count: int
    scaled_generalized_residual: float
    unscaled_generalized_residual: float
    eigenvalue_condition_number: float

    @property
    def generalized_residual(self) -> float:
        """Backward-compatible name for the unscaled pencil residual."""
        return self.unscaled_generalized_residual


def equilibrate_pencil(matrix_a: np.ndarray, matrix_b: np.ndarray):
    """Two-sided max-norm equilibration of a generalized eigenvalue pencil."""
    combined = np.maximum(np.abs(matrix_a), np.abs(matrix_b))
    row_norm = np.max(combined, axis=1)
    row_scale = 1.0 / np.maximum(row_norm, np.finfo(float).tiny)
    row_a = row_scale[:, None] * matrix_a
    row_b = row_scale[:, None] * matrix_b
    column_norm = np.max(np.maximum(np.abs(row_a), np.abs(row_b)), axis=0)
    column_scale = 1.0 / np.maximum(column_norm, np.finfo(float).tiny)
    return (
        row_a * column_scale[None, :],
        row_b * column_scale[None, :],
        row_scale,
        column_scale,
    )


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
    coefficients = independent_coefficient_blocks(radius, background)

    size = 2 * points
    matrix_a = np.zeros((size, size))
    matrix_b = np.zeros((size, size))
    interior = range(1, points - 1)
    for i in interior:
        # delta-varphi equation: D2 phi - RHS(lambda=0) = lambda q phi.
        matrix_a[i, :points] = (
            d2[i] - coefficients.phi_prime_in_phi_equation[i] * d1[i]
        )
        matrix_a[i, i] -= coefficients.phi_in_phi_equation[i]
        matrix_a[i, points:] = (
            -coefficients.delta_l_prime_in_phi_equation[i] * d1[i]
        )
        matrix_a[i, points + i] -= coefficients.delta_l_in_phi_equation[i]
        matrix_b[i, i] = coefficients.sigma2_phi_in_phi_equation[i]

        # delta-L equation.
        row = points + i
        matrix_a[row, :points] = -coefficients.phi_prime_in_l_equation[i] * d1[i]
        matrix_a[row, i] -= coefficients.phi_in_l_equation[i]
        matrix_a[row, points:] = (
            d2[i] - coefficients.delta_l_prime_in_l_equation[i] * d1[i]
        )
        matrix_a[row, points + i] -= coefficients.delta_l_in_l_equation[i]
        matrix_b[row, points + i] = coefficients.sigma2_delta_l_in_l_equation[i]

    # Regular center conditions, homogeneous in the arbitrary mode amplitude.
    center_value = np.zeros(points)
    center_value[0] = 1.0
    leading_value = center_value - 0.5 * epsilon * d1[0]
    matrix_a[0, :points] = -leading_value
    matrix_a[0, points:] = leading_value

    d0 = independent_center_d0(background)
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
    if background_representation != "hermite":
        raise ValueError(
            "the spectral certification solver requires the C1 Hermite background; "
            "PCHIP is permitted only for local-BVP uncertainty checks"
        )
    matrix_a, matrix_b, radius, background = assemble_generalized_problem(
        solution,
        points=points,
        epsilon=epsilon,
        r_max=r_max,
        background_representation=background_representation,
    )
    scaled_a, scaled_b, _, column_scale = equilibrate_pencil(matrix_a, matrix_b)
    eigenvalues, left_vectors, right_vectors = eig(
        scaled_a, scaled_b, left=True, right=True, check_finite=True
    )
    modes: list[SpectralRadialMode] = []
    for index, value in enumerate(eigenvalues):
        if not np.isfinite(value) or abs(value.imag) > imaginary_tolerance:
            continue
        sigma2 = float(value.real)
        if not sigma2_min <= sigma2 <= sigma2_max:
            continue
        scaled_vector = right_vectors[:, index]
        vector = column_scale * scaled_vector
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
        unscaled_residual_vector = matrix_a @ vector - sigma2 * (matrix_b @ vector)
        unscaled_denominator = (
            np.linalg.norm(matrix_a @ vector)
            + abs(sigma2) * np.linalg.norm(matrix_b @ vector)
            + 1e-300
        )
        scaled_residual_vector = scaled_a @ scaled_vector - sigma2 * (scaled_b @ scaled_vector)
        scaled_denominator = (
            np.linalg.norm(scaled_a @ scaled_vector)
            + abs(sigma2) * np.linalg.norm(scaled_b @ scaled_vector)
            + 1e-300
        )
        left = left_vectors[:, index]
        pairing = abs(np.vdot(left, scaled_b @ scaled_vector))
        condition_number = float(
            np.linalg.norm(left) * np.linalg.norm(scaled_vector) / max(pairing, 1e-300)
        )
        modes.append(
            SpectralRadialMode(
                sigma2=sigma2,
                r=radius,
                delta_varphi=phi,
                delta_l=delta_l,
                physical_scalar=physical_scalar,
                delta_lambda=delta_lambda,
                node_count=node_count(radius, delta_lambda),
                scaled_generalized_residual=float(
                    np.linalg.norm(scaled_residual_vector) / scaled_denominator
                ),
                unscaled_generalized_residual=float(
                    np.linalg.norm(unscaled_residual_vector) / unscaled_denominator
                ),
                eigenvalue_condition_number=condition_number,
            )
        )
    return sorted(modes, key=lambda mode: mode.sigma2)
