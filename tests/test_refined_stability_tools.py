import numpy as np
import pytest

from experiments.refined_stability_scan import zero_crossing_linear


def test_zero_crossing_linear_recovers_affine_root():
    x = np.array([0.09, 0.10, 0.11])
    y = 2.5 * (x - 0.103)
    root, lower, upper = zero_crossing_linear(x, y)
    assert root == pytest.approx(0.103)
    assert (lower, upper) == pytest.approx((0.10, 0.11))


def test_zero_crossing_requires_unique_bracket():
    with pytest.raises(ValueError):
        zero_crossing_linear(np.arange(3.0), np.ones(3))
    with pytest.raises(ValueError):
        zero_crossing_linear(
            np.arange(4.0), np.array([-1.0, 1.0, -1.0, 1.0])
        )
