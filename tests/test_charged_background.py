import numpy as np

from background.charged_ell_boson_star import charged_rhs
from background.ell_boson_star import _rhs


def test_charged_rhs_reduces_exactly_to_neutral_scalar_gravity_system():
    radius = np.geomspace(0.2, 8.0, 20)
    mass = 0.02 * radius**3 / (1.0 + radius**3)
    alpha = 0.9 + 0.01 * radius
    psi = 0.03 * radius * np.exp(-0.4 * radius)
    dpsi = psi * (1.0 / radius - 0.4)
    fields = np.vstack((mass, alpha, psi, dpsi, np.zeros_like(radius), np.zeros_like(radius)))
    charged = charged_rhs(radius, fields, 0.84, 1, 0.0)
    neutral = _rhs(radius, fields[:4], 0.84, 1)

    assert np.allclose(charged[:4], neutral, rtol=0.0, atol=1e-14)
    assert np.allclose(charged[4:], 0.0, rtol=0.0, atol=1e-14)


def test_electric_flux_sources_and_rn_mass_tail_have_consistent_signs():
    radius = np.asarray((2.0, 3.0))
    fields = np.vstack(
        (
            np.full(2, 0.1),
            np.full(2, 0.95),
            np.full(2, 0.02),
            np.zeros(2),
            np.full(2, 0.01),
            np.full(2, 0.03),
        )
    )
    rhs = charged_rhs(radius, fields, 0.85, 1, 0.5)

    assert np.all(rhs[4] < 0.0)
    assert np.all(rhs[5] > 0.0)
    assert np.all(rhs[0] > 2.0 * np.pi * fields[5] ** 2 / radius**2)
