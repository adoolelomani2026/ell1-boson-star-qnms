import pytest

from background.ell_boson_star import solve_by_continuation
from radial.center_series import center_d, center_series
from radial.coefficients import RadialBackground


@pytest.fixture(scope="module")
def radial_background():
    return RadialBackground(solve_by_continuation(1, 0.08))


def test_operator_consistent_amplitude_and_center_data(radial_background):
    expected = (
        1.0
        + (2.0 * radial_background.omega**2 - 2.4e-4)
        / (2.0 * radial_background.alpha_c**2)
        + 6.0 * 3.0 * (0.08 / 3.0) ** 2
    ) / 5.0
    assert center_d(2.4e-4, radial_background) == pytest.approx(expected)
    values = center_series(1e-5, 2.4e-4, -2.78e-2, radial_background)
    assert values[0] == pytest.approx(1.0 - 2.78e-2 * 1e-10)
    assert values[1] == pytest.approx(-5.56e-7)
    assert values[2] == pytest.approx(1.0 + expected * 1e-10)
