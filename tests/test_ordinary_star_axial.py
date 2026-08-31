import numpy as np

from background.ell_boson_star import solve_background
from nonradial.ordinary_star_axial import (
    stellar_outgoing_log_derivative,
    stellar_regge_wheeler_potential,
)
from radial.coefficients import RadialBackground


def test_ordinary_stellar_axial_potential_has_flat_center_limit():
    background = RadialBackground(
        solve_background(0, 0.12, r_max=40.0, points=350, tolerance=2.0e-6)
    )
    radius = 3.0e-4
    potential = stellar_regge_wheeler_potential(radius, background)
    expected = background.alpha_c**2 * 6.0 / radius**2
    assert np.isclose(potential / expected, 1.0, rtol=2.0e-4)


def test_ordinary_stellar_axial_potential_approaches_schwarzschild_tail():
    background = RadialBackground(
        solve_background(0, 0.12, r_max=50.0, points=400, tolerance=2.0e-6)
    )
    radius = 45.0
    point = background.point(radius)
    expected = point.alpha**2 * (6.0 / radius**2 - 6.0 * point.mass / radius**3)
    potential = stellar_regge_wheeler_potential(radius, background)
    assert abs(potential - expected) < 1.0e-14


def test_r2_centered_leaver_derivative_has_outgoing_asymptotic_limit():
    frequency = 0.277 - 0.388j
    mass = 0.633
    radius = 1000.0
    coordinate_log_derivative = stellar_outgoing_log_derivative(
        frequency, radius, mass=mass, leaver_terms=400
    )
    tortoise_log_derivative = (1.0 - 2.0 * mass / radius) * coordinate_log_derivative
    assert abs(tortoise_log_derivative - 1.0j * frequency) < 8.0e-6
