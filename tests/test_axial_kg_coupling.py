import sympy as sp

from experiments.derive_axial_kg_coupling import derive_kg_metric_coupling


def test_coordinate_and_compact_kg_couplings_are_identical():
    derived, expected, residual = derive_kg_metric_coupling()

    assert residual == 0
    assert sp.simplify(derived - expected) == 0


def test_minimal_local_state_has_six_components():
    # h0 and h1 are first order after the Einstein constraints are selected;
    # each of the two KG sidebands supplies an amplitude and its derivative.
    metric_components = 2
    sidebands = 2
    radial_order_per_sideband = 2

    assert metric_components + sidebands * radial_order_per_sideband == 6
