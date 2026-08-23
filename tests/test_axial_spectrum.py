import numpy as np

from nonradial.axial_spectrum import count_modes_adaptive, rectangular_contour, winding_number


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
    assert result["final_contour_points"] > 16
