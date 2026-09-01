from types import SimpleNamespace

import numpy as np

from nonradial.axial_spectrum import (
    count_modes_adaptive,
    quadtree_census,
    rectangular_contour,
    refine_scaled_root,
    winding_number,
)
from nonradial.riemann_sheet import SidebandSheet


def test_rectangular_contour_is_counter_clockwise_and_has_no_duplicate_corners():
    contour = rectangular_contour((1.0, 3.0), (-2.0, -1.0), 5)
    assert contour.shape == (20,)
    assert len(np.unique(contour)) == len(contour)
    assert contour[0] == 1.0 - 2.0j


def test_winding_number_counts_known_linear_zero():
    contour = rectangular_contour((-2.0, 2.0), (-2.0, 2.0), 16)
    count, maximum_increment = winding_number(contour - (0.3 - 0.4j))
    assert count == 1
    assert maximum_increment < np.pi


def test_winding_number_excludes_external_zero():
    contour = rectangular_contour((-2.0, 2.0), (-2.0, 2.0), 16)
    count, _ = winding_number(contour - (3.0 + 0.0j))
    assert count == 0


def test_phase_resolution_detects_an_underresolved_contour():
    values = np.exp(1j * np.linspace(0.0, 1.5 * np.pi, 4))
    count, maximum_increment = winding_number(values)
    assert count == 1
    assert maximum_increment >= 0.5 * np.pi


def test_adaptive_count_resolves_a_near_boundary_linear_zero():
    def determinant(sigma, _background, **_options):
        return sigma - (0.98 + 0.01j)

    result = count_modes_adaptive(
        object(),
        real_bounds=(-1.0, 1.0),
        imaginary_bounds=(-1.0, 1.0),
        initial_points_per_edge=4,
        determinant=determinant,
    )
    assert result["winding_number"] == 1
    assert result["phase_resolution_pass"]
    assert result["count_stability_pass"]
    assert len(result["winding_history"]) >= 2
    assert result["final_contour_points"] > 16


def test_adaptive_count_rejects_invalid_stability_requirements():
    with np.testing.assert_raises(ValueError):
        count_modes_adaptive(
            object(),
            real_bounds=(-1.0, 1.0),
            imaginary_bounds=(-1.0, 1.0),
            determinant=lambda sigma, _background: sigma,
            required_stable_refinements=0,
        )


def test_scaled_root_refinement_does_not_assume_holomorphic_normalization():
    root = 0.31 - 0.27j

    def determinant(sigma, _background, **_options):
        return (sigma - root) * (1.0 + 0.4 * abs(sigma))

    result = refine_scaled_root(
        0.29 - 0.25j,
        object(),
        determinant=determinant,
        derivative_step=1.0e-5,
        maximum_iterations=12,
    )
    assert result.converged
    assert abs(result.pole - root) < 1.0e-10


def test_sideband_sheet_declares_branch_points_and_cut_intersection():
    sheet = SidebandSheet.physical_lower_half_plane()
    points = sheet.branch_points(0.8)
    np.testing.assert_allclose(points["plus"], (-0.2, 1.8))
    np.testing.assert_allclose(points["minus"], (-1.8, 0.2))
    assert sheet.cell_intersects_cut((-0.1, 0.1), (-0.2, 0.2), 0.8)
    assert not sheet.cell_intersects_cut((-0.1, 0.1), (-0.2, -0.01), 0.8)


def test_quadtree_census_assigns_every_polynomial_zero():
    roots = (-0.63 - 0.42j, 0.31 - 0.67j, 0.72 - 0.22j)

    def determinant(sigma, _background, **_options):
        return np.prod([sigma - root for root in roots])

    census = quadtree_census(
        SimpleNamespace(omega=0.8),
        real_bounds=(-1.0, 1.0),
        imaginary_bounds=(-1.0, -0.01),
        sheet=SidebandSheet.physical_lower_half_plane(),
        determinant=determinant,
        maximum_depth=6,
    )
    assert census.complete
    assert census.counted_zeros == len(roots)
    assert len(census.assigned_poles) == len(roots)
    for expected in roots:
        assert min(abs(observed - expected) for observed in census.assigned_poles) < 1e-9


def test_quadtree_contour_moment_handles_off_center_single_root():
    root = 0.97 - 0.015j
    census = quadtree_census(
        SimpleNamespace(omega=0.8),
        real_bounds=(-1.0, 1.0),
        imaginary_bounds=(-1.0, -0.001),
        sheet=SidebandSheet.physical_lower_half_plane(),
        determinant=lambda sigma, _background, **_options: sigma - root,
        initial_points_per_edge=4,
        maximum_depth=2,
    )
    assert census.complete
    assert abs(census.assigned_poles[0] - root) < 1e-9


def test_quadtree_census_explicitly_excludes_cut_cells():
    census = quadtree_census(
        SimpleNamespace(omega=0.8),
        real_bounds=(-0.1, 0.1),
        imaginary_bounds=(-0.1, 0.1),
        sheet=SidebandSheet.physical_lower_half_plane(),
        determinant=lambda sigma, _background, **_options: sigma - (2.0 + 0.0j),
    )
    assert not census.complete
    assert len(census.excluded_cut_cells) == 1
    assert not census.leaves
