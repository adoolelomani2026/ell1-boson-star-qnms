import pytest

from background.ell_boson_star import solve_by_continuation
from radial.bvp import solve_radial_bvp
from radial.spectral import solve_radial_spectrum


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


@pytest.fixture(scope="session")
def radial_spectrum_a008(ell1_background_a008):
    return solve_radial_spectrum(
        ell1_background_a008,
        points=80,
        r_max=40.0,
        sigma2_min=-1e-3,
        sigma2_max=0.02,
    )
