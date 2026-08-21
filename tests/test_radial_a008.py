import numpy as np
import pytest


def test_radial_ground_mode_benchmark(radial_bvp_a008):
    assert radial_bvp_a008.success
    assert radial_bvp_a008.sigma2 == pytest.approx(2.40e-4, rel=5e-3)
    assert radial_bvp_a008.center_c == pytest.approx(-2.78e-2, rel=5e-3)
    assert radial_bvp_a008.node_count == 0
    assert radial_bvp_a008.max_collocation_residual < 5.1e-6


def test_physical_outer_conditions_are_directly_satisfied(radial_bvp_a008):
    assert np.max(np.abs(radial_bvp_a008.physical_boundary_residual)) < 1e-12
