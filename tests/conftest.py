import pytest

from background.ell_boson_star import solve_by_continuation
from radial.bvp import solve_radial_bvp


@pytest.fixture(scope="session")
def ell1_background_a008():
    return solve_by_continuation(1, 0.08)


@pytest.fixture(scope="session")
def radial_bvp_a008(ell1_background_a008):
    return solve_radial_bvp(
        ell1_background_a008,
        r_max=25.0,
        points=300,
        tolerance=5e-6,
    )
