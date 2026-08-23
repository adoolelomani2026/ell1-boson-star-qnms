import numpy as np

from nonradial.schwarzschild_rw import (
    leaver_continued_fraction,
    outgoing_log_derivative,
    regge_wheeler_potential,
)


def test_regge_wheeler_potential_vanishes_at_horizon_and_infinity():
    assert abs(regge_wheeler_potential(2.0)) < 1.0e-15
    assert regge_wheeler_potential(1.0e8) < 1.0e-14


def test_regge_wheeler_potential_is_positive_outside_horizon():
    radii = np.linspace(2.0001, 100.0, 500)
    assert np.all([regge_wheeler_potential(radius) > 0.0 for radius in radii])


def test_leaver_residual_is_small_at_fundamental_mode():
    frequency = 0.37367168 - 0.08896232j
    assert abs(leaver_continued_fraction(frequency, terms=300)) < 2.0e-7


def test_outgoing_log_derivative_is_series_converged():
    frequency = 0.37367168 - 0.08896232j
    low = outgoing_log_derivative(frequency, 35.0, terms=300)
    high = outgoing_log_derivative(frequency, 35.0, terms=600)
    assert abs(low - high) < 1.0e-10
