import numpy as np
import pytest

from radial.spectral import chebyshev_lobatto, solve_radial_spectrum


def test_chebyshev_derivative_exact_for_low_polynomials():
    radius, d1, d2 = chebyshev_lobatto(20, 1e-3, 40.0)
    assert np.max(np.abs(d1 @ radius - 1.0)) < 1e-10
    assert np.max(np.abs(d2 @ radius**2 - 2.0)) < 2e-9


def test_spectral_ground_and_first_overtone(radial_spectrum_a008):
    ground = next(mode for mode in radial_spectrum_a008 if mode.node_count == 0)
    overtone = next(mode for mode in radial_spectrum_a008 if mode.node_count == 1)
    assert ground.sigma2 == pytest.approx(2.40043e-4, abs=2e-9)
    assert overtone.sigma2 == pytest.approx(0.0907**2, rel=5e-4)
    assert np.sqrt(overtone.sigma2) == pytest.approx(0.0907, rel=3e-4)
    assert ground.generalized_residual < 1e-6
    assert overtone.generalized_residual < 1e-7


@pytest.mark.slow
def test_spectral_resolution_convergence(ell1_background_a008, radial_spectrum_a008):
    spectrum60 = solve_radial_spectrum(
        ell1_background_a008,
        points=60,
        r_max=40.0,
        sigma2_min=-1e-3,
        sigma2_max=0.02,
    )
    ground60 = next(mode for mode in spectrum60 if mode.node_count == 0)
    ground80 = next(mode for mode in radial_spectrum_a008 if mode.node_count == 0)
    assert ground80.sigma2 == pytest.approx(ground60.sigma2, abs=4e-9)
