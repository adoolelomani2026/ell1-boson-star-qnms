import pytest

from radial.bvp import solve_radial_bvp


@pytest.mark.slow
def test_bvp_domain_plateau_executes_solver(ell1_background_a008, radial_bvp_a008):
    mode30 = solve_radial_bvp(
        ell1_background_a008,
        r_max=30.0,
        points=300,
        tolerance=5e-6,
    )
    assert mode30.success
    assert mode30.sigma2 == pytest.approx(radial_bvp_a008.sigma2, rel=5e-4)
    assert mode30.node_count == 0
