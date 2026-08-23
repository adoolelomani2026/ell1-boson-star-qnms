from types import SimpleNamespace

import numpy as np

from nonradial.axial_ekg import (
    axial_rhs,
    center_basis,
    exterior_basis,
    exterior_channel_wavenumbers,
    exterior_log_derivatives,
    scattering_amplitudes,
    scattering_matrix,
)


def test_third_compound_generator_matches_minor_derivative():
    from nonradial.axial_ekg import _plucker_coordinates, _third_compound_generator

    rng = np.random.default_rng(20260823)
    generator = rng.normal(size=(6, 6))
    basis = rng.normal(size=(6, 3))
    expected = _third_compound_generator(generator) @ _plucker_coordinates(basis)
    step = 1.0e-7
    observed = (
        _plucker_coordinates(basis + step * generator @ basis)
        - _plucker_coordinates(basis - step * generator @ basis)
    ) / (2.0 * step)
    np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=2e-8)


class FlatVacuumBackground:
    omega = 0.8
    ell = 1
    adm_mass = 0.0
    r_min = 1e-5
    r_max = 35.0

    def point(self, radius):
        return SimpleNamespace(
            r=radius,
            alpha=1.0,
            gamma=1.0,
            psi=0.0,
            dpsi=0.0,
            log_alpha_prime=0.0,
            log_gamma_prime=0.0,
        )

    def lapse_second_derivative(self, radius):
        return np.asarray(0.0)


def test_flat_vacuum_metric_equations_reduce_to_regge_wheeler_pair():
    radius = 3.7
    sigma = 0.42 - 0.03j
    h0 = 0.7 + 0.2j
    h1 = -0.4 + 0.1j
    state = np.array((h0, h1, 0, 0, 0, 0), dtype=complex)

    result = axial_rhs(radius, state, sigma, FlatVacuumBackground())

    expected_h1_prime = -1j * sigma * h0
    expected_h0_prime = 2 * h0 / radius + (
        sigma**2 - 4 / radius**2
    ) * h1 / (1j * sigma)
    assert np.allclose(result[1], expected_h1_prime, rtol=0, atol=1e-13)
    assert np.allclose(result[0], expected_h0_prime, rtol=0, atol=1e-13)


def test_flat_vacuum_scalar_sidebands_decouple():
    radius = 4.1
    sigma = 0.2 - 0.01j
    up, vp, um, vm = 0.3, -0.2, -0.1, 0.4
    state = np.array((0, 0, up, vp, um, vm), dtype=complex)

    result = axial_rhs(radius, state, sigma, FlatVacuumBackground())
    omega = FlatVacuumBackground.omega
    expected_plus = -2 * vp / radius - (
        (omega - sigma) ** 2 - 1 - 6 / radius**2
    ) * up
    expected_minus = -2 * vm / radius - (
        (omega + sigma) ** 2 - 1 - 6 / radius**2
    ) * um
    assert np.allclose(result[3], expected_plus, rtol=0, atol=1e-13)
    assert np.allclose(result[5], expected_minus, rtol=0, atol=1e-13)


def test_center_basis_contains_three_regular_seed_columns():
    radius = 1e-3
    sigma = 0.3 - 0.02j
    basis = center_basis(radius, sigma, FlatVacuumBackground())

    assert basis.shape == (6, 3)
    assert np.linalg.matrix_rank(basis) == 3
    assert np.isclose(basis[3, 1] / basis[2, 1], 2 / radius)
    assert np.isclose(basis[5, 2] / basis[4, 2], 2 / radius)
    assert np.isclose(
        basis[1, 0] / radius**4,
        -1j * sigma / 4,
    )


def test_closed_scalar_channels_choose_decaying_sheet():
    _, k_plus, k_minus = exterior_channel_wavenumbers(
        0.1 - 0.01j, FlatVacuumBackground.omega
    )

    assert k_plus.imag >= 0
    assert k_minus.imag >= 0


def test_flat_exterior_riccati_matches_spherical_hankel_solution():
    radius = 20.0
    sigma = 0.25 - 0.02j
    background = FlatVacuumBackground()
    x_gravity, q_plus, q_minus = exterior_log_derivatives(
        radius, sigma, background.omega, 0.0, r_far=100.0
    )
    _, k_plus, k_minus = exterior_channel_wavenumbers(sigma, background.omega)

    def hankel_log_derivative(k):
        z = k * radius
        polynomial = 1 + 3j / z - 3 / z**2
        polynomial_prime = -3j / z**2 + 6 / z**3
        return k * (1j - 1 / z + polynomial_prime / polynomial)

    q_h1 = 2 / radius + hankel_log_derivative(sigma)
    assert np.allclose(x_gravity, 1j * q_h1 / sigma, rtol=2e-4, atol=2e-6)
    assert np.allclose(q_plus, hankel_log_derivative(k_plus), rtol=2e-4, atol=2e-6)
    assert np.allclose(q_minus, hankel_log_derivative(k_minus), rtol=2e-4, atol=2e-6)


def test_flat_regular_center_scattering_has_unit_gravitational_reflection():
    result = scattering_amplitudes(
        0.08,
        FlatVacuumBackground(),
        incident_channel="gravity",
        r_match=14.0,
        r_end=35.0,
        rtol=1e-10,
        atol=1e-12,
    )

    assert np.isclose(abs(result["gravity_out"]), 1.0, rtol=2e-7, atol=2e-7)
    assert abs(result["plus_decay"]) < 1e-12
    assert abs(result["minus_out"]) < 1e-12
    assert result["linear_residual"] < 1e-12


def test_flat_two_channel_scattering_matrix_is_diagonal_and_unit_modulus():
    matrix, residual = scattering_matrix(
        0.30,
        FlatVacuumBackground(),
        r_match=14.0,
        r_end=35.0,
        rtol=1e-10,
        atol=1e-12,
    )

    assert np.allclose(matrix - np.diag(np.diag(matrix)), 0.0, atol=1e-12)
    assert np.allclose(np.abs(np.diag(matrix)), 1.0, rtol=2e-7, atol=2e-7)
    assert residual < 1e-12
