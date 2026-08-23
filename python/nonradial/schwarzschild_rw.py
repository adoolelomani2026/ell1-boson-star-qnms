"""Independent Schwarzschild Regge--Wheeler quasinormal-mode control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SchwarzschildMode:
    seed: complex
    frequency: complex
    residual: float
    success: bool
    evaluations: int


def regge_wheeler_potential(radius: float, *, mass: float = 1.0, ell: int = 2) -> float:
    """Odd-parity Schwarzschild potential in geometric units."""

    f = 1.0 - 2.0 * mass / radius
    return f * (ell * (ell + 1.0) / radius**2 - 6.0 * mass / radius**3)


def _horizon_log_derivative(frequency: complex, mass: float, epsilon: float) -> complex:
    """Ingoing d(ln Psi)/dr_* through first order at the horizon."""

    coefficient = (3.0 / (8.0 * mass**3)) / (
        1.0 / (2.0 * mass) - 2.0j * frequency
    )
    return -1.0j * frequency + coefficient * epsilon


def _infinity_log_derivative(frequency: complex, mass: float, radius: float) -> complex:
    """Outgoing d(ln Psi)/dr_* through O(r^-3) for ell=2."""

    a2 = -3.0j / frequency
    a3 = -3.0 / frequency**2 + 9.0j * mass / frequency
    return 1.0j * frequency + a2 / radius**2 + a3 / radius**3


def logarithmic_mismatch(
    frequency: complex,
    *,
    mass: float = 1.0,
    ell: int = 2,
    r_match: float = 3.2,
    r_far: float = 180.0,
    horizon_epsilon: float = 1.0e-5,
    rtol: float = 2.0e-10,
    atol: float = 2.0e-12,
) -> complex:
    """Match ingoing and outgoing Riccati solutions at one radius."""

    if ell != 2:
        raise NotImplementedError("the certified control currently implements ell=2")
    if r_match <= 2.0 * mass or r_far <= r_match:
        raise ValueError("require 2M < r_match < r_far")

    def rhs(radius: float, state: np.ndarray) -> np.ndarray:
        f = 1.0 - 2.0 * mass / radius
        potential = regge_wheeler_potential(radius, mass=mass, ell=ell)
        return np.asarray(-(state[0] ** 2 + frequency**2 - potential) / f).reshape(1)

    left = 2.0 * mass + horizon_epsilon
    ingoing = solve_ivp(
        rhs,
        (left, r_match),
        [_horizon_log_derivative(frequency, mass, horizon_epsilon)],
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    outgoing = solve_ivp(
        rhs,
        (r_far, r_match),
        [_infinity_log_derivative(frequency, mass, r_far)],
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not ingoing.success or not outgoing.success:
        raise RuntimeError("Regge--Wheeler Riccati integration failed")
    return complex(ingoing.y[0, -1] - outgoing.y[0, -1])


def leaver_continued_fraction(
    frequency: complex,
    *,
    mass: float = 1.0,
    ell: int = 2,
    spin: int = 2,
    terms: int = 300,
) -> complex:
    """Leaver characteristic function for Schwarzschild perturbations.

    The recurrence uses Leaver's horizon-radius convention ``2M=1`` and
    Laplace frequency ``s=-2 i M omega``.  Backward evaluation selects the
    minimal series solution and avoids the exponentially ill-conditioned
    real-axis outgoing integration used only as a diagnostic above.
    """

    if terms < 20:
        raise ValueError("terms must be at least 20")
    s = -2.0j * mass * frequency

    def alpha(n: int) -> complex:
        return n**2 + (2.0 * s + 2.0) * n + 2.0 * s + 1.0

    def beta(n: int) -> complex:
        return -(
            2.0 * n**2
            + (8.0 * s + 2.0) * n
            + 8.0 * s**2
            + 4.0 * s
            + ell * (ell + 1.0)
            - spin**2
            + 1.0
        )

    def gamma(n: int) -> complex:
        return n**2 + 4.0 * s * n + 4.0 * s**2 - spin**2

    tail = beta(terms)
    for n in range(terms - 1, 0, -1):
        tail = beta(n) - alpha(n) * gamma(n + 1) / tail
    return complex(beta(0) - alpha(0) * gamma(1) / tail)


def outgoing_log_derivative(
    frequency: complex,
    radius: float,
    *,
    mass: float = 1.0,
    ell: int = 2,
    spin: int = 2,
    terms: int = 500,
) -> complex:
    """Return outgoing ``d ln(Psi)/dr_*`` from a Miller-minimal series."""

    if radius <= 2.0 * mass:
        raise ValueError("radius must lie outside the horizon")
    s = -2.0j * mass * frequency

    def alpha(n: int) -> complex:
        return n**2 + (2.0 * s + 2.0) * n + 2.0 * s + 1.0

    def beta(n: int) -> complex:
        return -(
            2.0 * n**2
            + (8.0 * s + 2.0) * n
            + 8.0 * s**2
            + 4.0 * s
            + ell * (ell + 1.0)
            - spin**2
            + 1.0
        )

    def gamma(n: int) -> complex:
        return n**2 + 4.0 * s * n + 4.0 * s**2 - spin**2

    coefficients = np.zeros(terms + 2, dtype=complex)
    coefficients[terms] = 1.0
    for n in range(terms, 0, -1):
        coefficients[n - 1] = -(
            alpha(n) * coefficients[n + 1] + beta(n) * coefficients[n]
        ) / gamma(n)
        if abs(coefficients[n - 1]) > 1.0e100:
            coefficients[n - 1 :] *= 1.0e-100
    coefficients /= coefficients[0]
    x = 1.0 - 2.0 * mass / radius
    powers = x ** np.arange(terms + 1)
    series = np.dot(coefficients[: terms + 1], powers)
    series_x = np.dot(
        np.arange(1, terms + 1) * coefficients[1 : terms + 1],
        powers[:terms],
    )
    log_derivative_r = (
        1.0j * frequency
        - 2.0j * mass * frequency / (radius - 2.0 * mass)
        + 4.0j * mass * frequency / radius
        + (series_x / series) * (2.0 * mass / radius**2)
    )
    return complex((1.0 - 2.0 * mass / radius) * log_derivative_r)


def discover_fundamental_mode(
    *,
    real_bounds: tuple[float, float] = (0.34, 0.41),
    imaginary_bounds: tuple[float, float] = (-0.115, -0.065),
    scan_shape: tuple[int, int] = (6, 5),
    **options,
) -> tuple[SchwarzschildMode, list[dict[str, float]]]:
    """Grid-discover and refine the fundamental ell=2 axial QNM."""

    nx, ny = scan_shape
    rows: list[dict[str, float]] = []
    for imaginary in np.linspace(*imaginary_bounds, ny):
        for real in np.linspace(*real_bounds, nx):
            value = leaver_continued_fraction(complex(real, imaginary), **options)
            rows.append(
                {
                    "frequency_real": float(real),
                    "frequency_imag": float(imaginary),
                    "mismatch_abs": float(abs(value)),
                }
            )
    best = min(rows, key=lambda row: row["mismatch_abs"])
    seed = complex(best["frequency_real"], best["frequency_imag"])
    scale = np.array(
        (
            (real_bounds[1] - real_bounds[0]) / (nx - 1),
            (imaginary_bounds[1] - imaginary_bounds[0]) / (ny - 1),
        )
    )
    reference = max(abs(leaver_continued_fraction(seed, **options)), 1.0e-300)

    def equations(offsets: np.ndarray) -> tuple[float, float]:
        coordinates = np.array((seed.real, seed.imag)) + scale * offsets
        value = leaver_continued_fraction(complex(*coordinates), **options) / reference
        return value.real, value.imag

    lower = np.array(
        (
            (real_bounds[0] - seed.real) / scale[0],
            (imaginary_bounds[0] - seed.imag) / scale[1],
        )
    )
    upper = np.array(
        (
            (real_bounds[1] - seed.real) / scale[0],
            (imaginary_bounds[1] - seed.imag) / scale[1],
        )
    )
    solution = least_squares(
        equations,
        (0.0, 0.0),
        bounds=(lower, upper),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    coordinates = np.array((seed.real, seed.imag)) + scale * solution.x
    frequency = complex(*coordinates)
    residual = abs(leaver_continued_fraction(frequency, **options)) / reference
    return (
        SchwarzschildMode(
            seed=seed,
            frequency=frequency,
            residual=float(residual),
            success=bool(solution.success),
            evaluations=int(solution.nfev),
        ),
        rows,
    )
