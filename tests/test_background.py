import numpy as np
import pytest

from background.ell_boson_star import solve_by_continuation
from background.radius_audit import audit_radii


@pytest.fixture(scope="module")
def literature_model():
    return solve_by_continuation(1, 0.08, r_max=80.0, points=800, tolerance=1e-7)


def test_literature_frequency(literature_model):
    # Table II of arXiv:2103.15012 gives omega=0.8519 at a_1^0=0.080.
    assert literature_model.omega == pytest.approx(0.8519, abs=8e-4)


def test_background_is_regular_and_horizonless(literature_model):
    solution = literature_model
    assert np.all(np.diff(solution.mass) >= -1e-10)
    assert np.max(2.0 * solution.mass / solution.r) < 1.0
    assert np.all(solution.psi > -1e-9)
    exterior_lapse = np.sqrt(1.0 - 2.0 * solution.adm_mass / solution.r[-1])
    assert solution.alpha[-1] == pytest.approx(exterior_lapse, abs=2e-8)


def test_saved_diagnostics_are_small(literature_model):
    assert literature_model.max_ode_residual < 2e-3
    assert literature_model.tail_residual < 2e-5


def test_published_radius_tracks_999_percent_mass():
    maximum_model = solve_by_continuation(1, 0.10)
    audit = audit_radii(maximum_model)
    assert audit.eta_at_published_radius == pytest.approx(0.999, abs=2e-4)
    assert audit.r99_mass == pytest.approx(10.18, abs=0.03)
    assert audit.r999_mass == pytest.approx(12.70, abs=0.04)
    assert abs(audit.r99_mass_finite - audit.r99_mass_extrapolated) < 1e-5
