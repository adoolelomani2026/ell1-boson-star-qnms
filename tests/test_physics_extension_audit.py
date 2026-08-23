import numpy as np
import pytest

from experiments.physics_extension_audit import (
    FINE_STRUCTURE,
    PLANCK_MASS_EV,
    generate_audit,
    physical_scales,
)


def test_physical_scaling_relations_are_exact():
    low = physical_scales(
        1e-10,
        dimensionless_mass=1.171271161113,
        dimensionless_charge=1.212966,
        dimensionless_r99=11.16733443,
    )
    high = physical_scales(
        1e-9,
        dimensionless_mass=1.171271161113,
        dimensionless_charge=1.212966,
        dimensionless_r99=11.16733443,
    )
    assert high.mass_solar == pytest.approx(low.mass_solar / 10)
    assert high.r99_km == pytest.approx(low.r99_km / 10)
    assert high.particle_number == pytest.approx(low.particle_number / 100)
    assert high.planck_suppression == pytest.approx(low.planck_suppression * 100)


def test_electromagnetic_balance_definition():
    scale = physical_scales(
        1.0,
        dimensionless_mass=1.0,
        dimensionless_charge=1.0,
        dimensionless_r99=1.0,
    )
    ratio = 1.0 / PLANCK_MASS_EV
    assert FINE_STRUCTURE * scale.charge_balance_in_units_of_e**2 == pytest.approx(
        ratio**2
    )


def test_quantum_audit_is_monotone_and_classical_over_scan():
    rows = generate_audit(np.geomspace(1e-22, 1e9, 32))
    assert all(a.mass_solar > b.mass_solar for a, b in zip(rows, rows[1:]))
    assert all(a.r99_km > b.r99_km for a, b in zip(rows, rows[1:]))
    assert all(
        a.planck_suppression < b.planck_suppression for a, b in zip(rows, rows[1:])
    )
    assert max(row.planck_suppression for row in rows) < 1e-36
    assert max(row.finite_n_fractional_scale for row in rows) < 1e-18


def test_invalid_scale_inputs_are_rejected():
    with pytest.raises(ValueError):
        physical_scales(
            0.0,
            dimensionless_mass=1.0,
            dimensionless_charge=1.0,
            dimensionless_r99=1.0,
        )
