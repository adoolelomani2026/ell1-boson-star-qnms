import pytest

from radial.bvp import solve_radial_bvp
from radial.mode_tracking import eigenfunction_overlap


@pytest.mark.slow
def test_bvp_and_spectral_ground_eigenfunctions_agree(
    ell1_background_a008, radial_spectrum_a008
):
    bvp = solve_radial_bvp(
        ell1_background_a008,
        r_max=40.0,
        points=300,
        tolerance=2e-6,
    )
    spectral = next(mode for mode in radial_spectrum_a008 if mode.node_count == 0)
    scalar_overlap = eigenfunction_overlap(
        bvp.r, bvp.physical_scalar, spectral.r, spectral.physical_scalar
    )
    metric_overlap = eigenfunction_overlap(
        bvp.r, bvp.delta_lambda, spectral.r, spectral.delta_lambda
    )
    assert scalar_overlap > 0.999999
    assert metric_overlap > 0.99999
