import numpy as np
import pytest

from radial.mode_tracking import node_count, track_mode_by_overlap
from radial.spectral import chebyshev_lobatto, equilibrate_pencil, solve_radial_spectrum


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
    assert ground.scaled_generalized_residual < ground.unscaled_generalized_residual
    assert ground.eigenvalue_condition_number > 1.0


def test_spectral_rejects_nonsmooth_pchip_background(ell1_background_a008):
    with pytest.raises(ValueError, match="requires the C1 Hermite background"):
        solve_radial_spectrum(
            ell1_background_a008, points=40, background_representation="pchip"
        )


def test_node_counter_retains_low_amplitude_lobe():
    radius = np.linspace(0.0, 1.0, 1001)
    values = (radius - 0.2) * (radius - 0.5) * (radius - 0.8)
    values[radius > 0.65] *= 1e-3
    assert node_count(radius, values) == 3


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


@pytest.mark.slow
def test_broader_spectral_envelope_and_higher_mode_nodes(ell1_background_a008):
    references = []
    for points in (50, 80, 120, 160):
        spectrum = solve_radial_spectrum(
            ell1_background_a008,
            points=points,
            r_max=40.0,
            sigma2_min=-1e-3,
            sigma2_max=0.02,
        )
        ground = next(mode for mode in spectrum if mode.node_count == 0)
        references.append(ground.sigma2)
        if points == 120:
            assert [mode.node_count for mode in spectrum[:4]] == [0, 1, 2, 3]
    assert np.ptp(references) < 7e-9


def test_overlap_tracks_overtone_across_resolutions(ell1_background_a008):
    spectrum60 = solve_radial_spectrum(
        ell1_background_a008, points=60, r_max=40.0, sigma2_max=0.02
    )
    spectrum80 = solve_radial_spectrum(
        ell1_background_a008, points=80, r_max=40.0, sigma2_max=0.02
    )
    reference = next(mode for mode in spectrum60 if mode.node_count == 1)
    tracked, overlap = track_mode_by_overlap(reference, spectrum80)
    assert tracked.node_count == 1
    assert overlap > 0.99998
