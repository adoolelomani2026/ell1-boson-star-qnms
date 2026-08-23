import numpy as np

import sympy as sp

from nonradial.axial_projection import axial_projection, symbolic_m0_projection


def test_ell1_j2_l2_is_density_silent_but_axially_active():
    result = axial_projection(magnetic_m=0)

    assert result.density_norm < 1e-12
    assert abs(result.vector_coefficient) > 1e-3
    assert abs(result.tensor_coefficient) > 1e-3


def test_axial_projection_magnitudes_are_independent_of_m():
    results = [axial_projection(magnetic_m=m) for m in range(-2, 3)]
    vector_magnitudes = np.array([abs(item.vector_coefficient) for item in results])
    tensor_magnitudes = np.array([abs(item.tensor_coefficient) for item in results])

    assert np.ptp(vector_magnitudes) < 2e-12
    assert np.ptp(tensor_magnitudes) < 2e-12


def test_axial_projection_is_quadrature_converged():
    coarse = axial_projection(magnetic_m=2, n_theta=24, n_phi=32)
    fine = axial_projection(magnetic_m=2, n_theta=48, n_phi=64)

    assert abs(coarse.vector_coefficient - fine.vector_coefficient) < 2e-12
    assert abs(coarse.tensor_coefficient - fine.tensor_coefficient) < 2e-12


def test_ell0_control_has_no_axial_matter_projection():
    result = axial_projection(
        ell=0, total_j=2, orbital_l=2, magnetic_m=0
    )

    assert abs(result.vector_coefficient) < 1e-13
    assert abs(result.tensor_coefficient) < 1e-13


def test_m0_symbolic_projection_is_exact():
    result = symbolic_m0_projection()

    assert result["density"] == 0
    assert result["polar_vector_component"] == 0
    assert sp.simplify(result["vector_coefficient"] + sp.I / (2 * sp.sqrt(sp.pi))) == 0
    assert sp.simplify(result["tensor_coefficient"] - sp.I / sp.sqrt(sp.pi)) == 0
