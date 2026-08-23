import numpy as np

from nonradial.axial_time_domain import derivative_4, ko_dissipation, second_derivative_4


def test_fourth_order_first_derivative_is_exact_for_quartic_polynomial_inside():
    r = np.linspace(0.1, 5.0, 101)
    values = r**4 - 2 * r**3 + 0.7 * r
    exact = 4 * r**3 - 6 * r**2 + 0.7

    numerical = derivative_4(values, r[1] - r[0])

    assert np.max(np.abs(numerical[2:-2] - exact[2:-2])) < 2e-10


def test_fourth_order_second_derivative_is_exact_for_quartic_polynomial_inside():
    r = np.linspace(0.1, 5.0, 101)
    values = r**4 - 2 * r**3 + 0.7 * r
    exact = 12 * r**2 - 12 * r

    numerical = second_derivative_4(values, r[1] - r[0])

    assert np.max(np.abs(numerical[2:-2] - exact[2:-2])) < 2e-9


def test_derivative_operators_preserve_complex_linearity():
    r = np.linspace(0.2, 3.0, 80)
    spacing = r[1] - r[0]
    values = np.exp(0.3j * r)

    assert np.allclose(
        derivative_4((2 - 3j) * values, spacing),
        (2 - 3j) * derivative_4(values, spacing),
    )


def test_ko_filter_annihilates_degree_five_polynomial_interior():
    r = np.linspace(0.1, 5.0, 101)
    values = r**5 - 2 * r**3 + r

    filtered = ko_dissipation(values, r[1] - r[0], 0.12)

    assert np.max(np.abs(filtered[3:-3])) < 2e-10
