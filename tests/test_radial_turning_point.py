import pytest

from background.ell_boson_star import solve_by_continuation
from radial.bvp import solve_radial_bvp


@pytest.mark.slow
def test_bvp_stability_crossing_executes_solver(radial_bvp_a008):
    near_turning_background = solve_by_continuation(1, 0.10)
    unstable_background = solve_by_continuation(1, 0.105)
    near_turning = solve_radial_bvp(
        near_turning_background,
        sigma2_guess=4e-6,
        center_c_guess=-3.33e-2,
        points=300,
        tolerance=5e-6,
    )
    unstable = solve_radial_bvp(
        unstable_background,
        sigma2_guess=-7.1e-5,
        center_c_guess=-3.47e-2,
        points=300,
        tolerance=5e-6,
    )
    assert radial_bvp_a008.sigma2 > 0.0
    assert near_turning.success and abs(near_turning.sigma2) < 5e-6
    assert unstable.success and unstable.sigma2 < 0.0
    assert unstable.sigma2 == pytest.approx(-7.11e-5, rel=0.01)
