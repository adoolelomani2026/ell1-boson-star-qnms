from pathlib import Path

import numpy as np

from background.ell_boson_star import BackgroundSolution
from radial.coefficients import RadialBackground


ROOT = Path(__file__).resolve().parents[1]


def test_ekg_lapse_second_derivative_agrees_with_hermite_interior():
    solution = BackgroundSolution.load(ROOT / "reports" / "axial" / "background_ell1_a008_axial.npz")
    background = RadialBackground(solution, geometry_representation="hermite")
    radii = np.geomspace(0.01, 35.0, 200)
    interpolated = background.lapse_second_derivative(radii)
    reconstructed = background.equilibrium_lapse_second_derivative(radii)
    assert np.max(np.abs(interpolated - reconstructed)) < 2e-6
    assert np.median(np.abs(interpolated - reconstructed)) < 2e-8
