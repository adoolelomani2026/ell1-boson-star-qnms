from types import SimpleNamespace

import numpy as np

from nonradial.axial_static import static_rhs, static_tidal_response


class FlatVacuumBackground:
    omega = 0.8
    ell = 1
    adm_mass = 0.2
    r_min = 1e-5
    r_max = 35.0

    def point(self, radius):
        # Schwarzschild rather than globally flat so the exterior response
        # basis is exact; the center test below uses a separate local object.
        f = 1.0 - 2.0 * self.adm_mass / radius
        alpha = np.sqrt(f)
        la = self.adm_mass / (radius**2 * f)
        return SimpleNamespace(
            alpha=alpha,
            gamma=1.0 / alpha,
            psi=0.0,
            dpsi=0.0,
            log_alpha_prime=la,
            log_gamma_prime=-la,
        )

    def lapse_second_derivative(self, radius):
        f = 1.0 - 2.0 * self.adm_mass / radius
        alpha = np.sqrt(f)
        return np.asarray(
            -2.0 * self.adm_mass / (radius**3 * alpha)
            - self.adm_mass**2 / (radius**4 * alpha**3)
        )


class LocalFlatBackground(FlatVacuumBackground):
    adm_mass = 0.0


def test_static_flat_metric_equation_has_expected_power_laws():
    background = LocalFlatBackground()
    radius = 2.7
    for power in (3.0, -2.0):
        h0 = radius**power
        state = np.asarray((h0, power * radius ** (power - 1.0), 0.0, 0.0))
        rhs = static_rhs(radius, state, background)
        assert np.isclose(rhs[1], power * (power - 1.0) * radius ** (power - 2.0))


def test_static_scalar_difference_decouples_in_vacuum():
    background = LocalFlatBackground()
    radius = 3.1
    state = np.asarray((0.0, 0.0, 0.4, -0.2))
    rhs = static_rhs(radius, state, background)
    expected = 0.4 * (1.0 + 6.0 / radius**2 - background.omega**2) + 0.4 / radius
    assert np.isclose(rhs[3], expected)
