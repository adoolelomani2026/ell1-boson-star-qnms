"""Regular-center data for the radial pulsation variables."""

from __future__ import annotations

import numpy as np

from .coefficients import RadialBackground


def center_d(sigma2: float, background: RadialBackground) -> float:
    if background.ell != 1:
        raise NotImplementedError("the current center specialization is for ell=1")
    # The coefficient in the Appendix expansion is the actual leading
    # coefficient of the radial field used by the ODE: psi ~ a0 r^ell.  In this
    # repository's validated background normalization that is a_ell^0/kappa.
    # Using kappa*a_ell^0 produces an O(1) center-operator residual.
    appendix_a0 = background.a0 / background.kappa
    return (
        1.0
        + (2.0 * background.omega**2 - sigma2) / (2.0 * background.alpha_c**2)
        + 6.0 * background.kappa * appendix_a0**2
    ) / 5.0


def center_series(
    epsilon: float, sigma2: float, center_c: float, background: RadialBackground
) -> np.ndarray:
    d_coefficient = center_d(sigma2, background)
    return np.array(
        (
            1.0 + center_c * epsilon**2,
            2.0 * center_c * epsilon,
            1.0 + d_coefficient * epsilon**2,
            2.0 * d_coefficient * epsilon,
        )
    )
