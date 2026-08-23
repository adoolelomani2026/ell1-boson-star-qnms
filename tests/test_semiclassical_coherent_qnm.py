import pytest

from experiments.semiclassical_coherent_qnm import coherent_qnm_statistics


def test_coherent_qnm_dispersion_and_bias_have_expected_n_scaling():
    kwargs = dict(
        sigma=0.05 - 1e-6j,
        first_derivative=0.4 - 1e-5j,
        second_derivative=-0.2 + 2e-5j,
        charge=1.2,
        charge_first_derivative=0.6,
        charge_second_derivative=-10.0,
    )
    low = coherent_qnm_statistics(1e20, **kwargs)
    high = coherent_qnm_statistics(4e20, **kwargs)

    assert high["qnm_absolute_standard_deviation"] == pytest.approx(
        low["qnm_absolute_standard_deviation"] / 2.0
    )
    assert high["mean_qnm_fractional_shift_magnitude"] == pytest.approx(
        low["mean_qnm_fractional_shift_magnitude"] / 4.0
    )


def test_coherent_qnm_statistics_reject_nonpositive_occupation():
    with pytest.raises(ValueError):
        coherent_qnm_statistics(
            0.0,
            sigma=0.05 + 0j,
            first_derivative=0.4 + 0j,
            second_derivative=0j,
            charge=1.2,
            charge_first_derivative=0.6,
            charge_second_derivative=-10.0,
        )


def test_occupation_noise_uses_charge_sequence_jacobian():
    result = coherent_qnm_statistics(
        100.0,
        sigma=0.05 - 1e-6j,
        first_derivative=0.4 - 1e-5j,
        second_derivative=-2.0 - 1e-4j,
        charge=1.2,
        charge_first_derivative=0.6,
        charge_second_derivative=-10.0,
    )
    assert result["delta_a1_0_times_sqrt_N"] == pytest.approx(2.0)
