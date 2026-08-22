import numpy as np

from radial.coefficients import RadialBackground


def test_hermite_reproduces_stored_field_derivative(ell1_background_a008):
    background = RadialBackground(ell1_background_a008, representation="hermite")
    indices = np.linspace(0, ell1_background_a008.r.size - 1, 101, dtype=int)
    radius = ell1_background_a008.r[indices]
    values = background.arrays(radius)
    np.testing.assert_allclose(values[3], ell1_background_a008.psi[indices], rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(values[4], ell1_background_a008.dpsi[indices], rtol=2e-12, atol=2e-14)
