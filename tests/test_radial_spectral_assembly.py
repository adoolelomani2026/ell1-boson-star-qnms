import numpy as np

from radial.coefficients import pulsation_rhs
from radial.center_series import center_d
from radial.spectral import (
    assemble_generalized_problem,
    chebyshev_lobatto,
    equilibrate_pencil,
)


def test_generalized_matrix_matches_direct_operator(ell1_background_a008):
    points = 48
    sigma2 = 3.7e-4
    matrix_a, matrix_b, radius, background = assemble_generalized_problem(
        ell1_background_a008, points=points, r_max=40.0
    )
    _, d1, d2 = chebyshev_lobatto(points, 1e-3, 40.0)
    phi = 1.0 + 0.03 * radius + 0.002 * radius**2
    delta_l = 0.8 - 0.02 * radius + 0.001 * radius**2
    vector = np.concatenate((phi, delta_l))
    state = np.vstack((phi, d1 @ phi, delta_l, d1 @ delta_l))
    direct = pulsation_rhs(radius, state, sigma2, background)
    residual = matrix_a @ vector - sigma2 * (matrix_b @ vector)
    np.testing.assert_allclose(residual[1 : points - 1], (d2 @ phi - direct[1])[1:-1], atol=2e-10)
    np.testing.assert_allclose(
        residual[points + 1 : 2 * points - 1],
        (d2 @ delta_l - direct[3])[1:-1],
        atol=2e-10,
    )
    epsilon = radius[0]
    leading_phi = phi[0] - 0.5 * epsilon * (d1 @ phi)[0]
    leading_l = delta_l[0] - 0.5 * epsilon * (d1 @ delta_l)[0]
    np.testing.assert_allclose(residual[0], leading_l - leading_phi, atol=2e-13)
    np.testing.assert_allclose(
        residual[points],
        (d1 @ delta_l)[0]
        - 2.0 * epsilon * center_d(sigma2, background) * leading_phi,
        atol=2e-13,
    )
    np.testing.assert_allclose(residual[points - 1], phi[-1], atol=0.0)
    np.testing.assert_allclose(residual[-1], delta_l[-1], atol=0.0)


def test_equilibration_reduces_pencil_dynamic_range(ell1_background_a008):
    matrix_a, matrix_b, _, _ = assemble_generalized_problem(
        ell1_background_a008, points=80, r_max=40.0
    )
    scaled_a, scaled_b, row_scale, column_scale = equilibrate_pencil(matrix_a, matrix_b)
    combined = np.maximum(np.abs(scaled_a), np.abs(scaled_b))
    nonzero_rows = np.max(combined, axis=1) > 0.0
    nonzero_columns = np.max(combined, axis=0) > 0.0
    assert np.ptp(np.max(combined, axis=1)[nonzero_rows]) < 1e-12
    assert np.ptp(np.max(combined, axis=0)[nonzero_columns]) < 1e-12
    assert np.all(row_scale > 0.0) and np.all(column_scale > 0.0)
