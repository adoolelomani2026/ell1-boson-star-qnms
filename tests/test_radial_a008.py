import pytest

from background.ell_boson_star import solve_by_continuation
from radial.shooting import solve_radial_mode


@pytest.fixture(scope="module")
def radial_ground_mode():
    background = solve_by_continuation(1, 0.08)
    return solve_radial_mode(background, r_max=25.0, rtol=1e-8, atol=1e-10)


def test_radial_ground_mode_benchmark(radial_ground_mode):
    # Published values are rounded and reported with roughly 0.1% convergence.
    assert radial_ground_mode.root_success
    assert radial_ground_mode.sigma2 == pytest.approx(2.40e-4, rel=5e-3)
    assert radial_ground_mode.center_c == pytest.approx(-2.78e-2, rel=5e-3)
    assert radial_ground_mode.node_count == 0


def test_published_outer_conditions(radial_ground_mode):
    relative = abs(radial_ground_mode.residual) / radial_ground_mode.residual_scale
    assert relative[0] < 1e-10
    assert relative[1] < 1e-6
