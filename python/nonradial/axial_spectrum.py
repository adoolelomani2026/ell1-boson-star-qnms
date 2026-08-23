"""Search, refine, and count neutral axial EKG quasinormal modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from nonradial.axial_ekg import (
    matching_evans_determinant,
    matching_raw_determinant,
    matching_singular_value,
)
from radial.coefficients import RadialBackground


@dataclass(frozen=True)
class AxialModeSearch:
    seed: complex
    pole: complex
    scan_minimum: float
    raw_residual_relative: float
    solver_success: bool
    evaluations: int


@dataclass(frozen=True)
class ComplexRootRefinement:
    pole: complex
    relative_residual: float
    converged: bool
    iterations: int
    evaluations: int


def refine_analytic_root(
    seed: complex,
    background: RadialBackground,
    *,
    matching_options: dict[str, float] | None = None,
    determinant: Callable[..., complex] = matching_raw_determinant,
    derivative_step: float = 2.0e-7,
    step_tolerance: float = 2.0e-12,
    residual_tolerance: float = 2.0e-5,
    maximum_iterations: int = 6,
) -> ComplexRootRefinement:
    """Refine a simple zero by complex Newton iteration.

    A centered derivative on the real-frequency direction is valid because the
    production determinant is holomorphic on the fixed physical sheet.
    """

    options = dict(matching_options or {})
    # A continuation seed may already lie extremely close to the next root,
    # so its determinant is not a stable normalization.  Use a fixed nearby
    # offset as the local reference scale as well.
    reference = max(
        abs(determinant(seed, background, **options)),
        abs(determinant(seed + 10.0 * derivative_step, background, **options)),
        1.0e-300,
    )
    pole = complex(seed)
    evaluations = 2
    step = complex(np.inf)
    for iteration in range(1, maximum_iterations + 1):
        value = determinant(pole, background, **options)
        plus = determinant(pole + derivative_step, background, **options)
        minus = determinant(pole - derivative_step, background, **options)
        evaluations += 3
        derivative = (plus - minus) / (2.0 * derivative_step)
        if derivative == 0.0 or not np.isfinite(abs(derivative)):
            break
        step = value / derivative
        pole -= step
        if abs(step) < step_tolerance:
            break
    residual = abs(determinant(pole, background, **options)) / reference
    evaluations += 1
    converged = bool(abs(step) < step_tolerance and residual < residual_tolerance)
    return ComplexRootRefinement(
        pole=pole,
        relative_residual=float(residual),
        converged=converged,
        iterations=iteration,
        evaluations=evaluations,
    )


def rectangular_contour(
    real_bounds: tuple[float, float],
    imaginary_bounds: tuple[float, float],
    points_per_edge: int,
) -> np.ndarray:
    """Counter-clockwise rectangular contour without repeated corners."""

    if points_per_edge < 3:
        raise ValueError("points_per_edge must be at least three")
    x0, x1 = real_bounds
    y0, y1 = imaginary_bounds
    if not x0 < x1 or not y0 < y1:
        raise ValueError("bounds must be strictly increasing")
    n = points_per_edge
    return np.concatenate(
        (
            np.linspace(x0, x1, n, endpoint=False) + 1j * y0,
            x1 + 1j * np.linspace(y0, y1, n, endpoint=False),
            np.linspace(x1, x0, n, endpoint=False) + 1j * y1,
            x0 + 1j * np.linspace(y1, y0, n, endpoint=False),
        )
    )


def winding_number(values: np.ndarray) -> tuple[int, float]:
    """Return phase winding and the largest resolved phase increment."""

    samples = np.asarray(values, dtype=complex)
    if samples.ndim != 1 or len(samples) < 4 or np.any(samples == 0):
        raise ValueError("nonzero one-dimensional contour samples are required")
    closed = np.concatenate((samples, samples[:1]))
    phase = np.unwrap(np.angle(closed))
    increments = np.diff(phase)
    winding = int(np.rint(np.sum(increments) / (2.0 * np.pi)))
    return winding, float(np.max(np.abs(increments)))


def discover_mode(
    background: RadialBackground,
    *,
    real_bounds: tuple[float, float],
    imaginary_bounds: tuple[float, float],
    scan_shape: tuple[int, int] = (7, 5),
    matching_options: dict[str, float] | None = None,
) -> tuple[AxialModeSearch, list[dict[str, float]]]:
    """Discover a pole from a declared grid, then refine the raw determinant.

    No stored frequency is read.  The least normalized singular value chooses
    the initial seed; the unnormalized, locally holomorphic matching
    determinant supplies the two real equations for refinement.
    """

    options = dict(matching_options or {})
    nx, ny = scan_shape
    if nx < 3 or ny < 3:
        raise ValueError("scan_shape entries must be at least three")
    reals = np.linspace(*real_bounds, nx)
    imaginaries = np.linspace(*imaginary_bounds, ny)
    scan_options = dict(options)
    scan_options["rtol"] = max(float(options.get("rtol", 2e-6)), 2e-6)
    scan_options["atol"] = max(float(options.get("atol", 2e-8)), 2e-8)
    rows: list[dict[str, float]] = []
    for imaginary in imaginaries:
        for real in reals:
            sigma = complex(real, imaginary)
            value = matching_singular_value(
                sigma,
                background,
                **scan_options,
            )
            rows.append(
                {"sigma_real": real, "sigma_imag": imaginary, "singular_value": value}
            )
    best = min(rows, key=lambda row: row["singular_value"])
    seed = complex(best["sigma_real"], best["sigma_imag"])
    refinement = refine_analytic_root(
        seed,
        background,
        matching_options=options,
        determinant=matching_evans_determinant,
        derivative_step=(real_bounds[1] - real_bounds[0]) / 6000.0,
    )
    pole = refinement.pole
    raw_relative = refinement.relative_residual
    result = AxialModeSearch(
        seed=seed,
        pole=pole,
        scan_minimum=float(best["singular_value"]),
        raw_residual_relative=float(raw_relative),
        solver_success=refinement.converged,
        evaluations=refinement.evaluations,
    )
    return result, rows


def count_modes(
    background: RadialBackground,
    *,
    real_bounds: tuple[float, float],
    imaginary_bounds: tuple[float, float],
    points_per_edge: int = 12,
    matching_options: dict[str, float] | None = None,
    determinant: Callable[..., complex] = matching_evans_determinant,
) -> dict[str, float | int]:
    """Count zeros inside a contour from the raw determinant phase."""

    contour = rectangular_contour(real_bounds, imaginary_bounds, points_per_edge)
    options = dict(matching_options or {})
    values = np.array([determinant(z, background, **options) for z in contour])
    count, maximum_increment = winding_number(values)
    return {
        "winding_number": count,
        "points_per_edge": points_per_edge,
        "minimum_boundary_determinant": float(np.min(np.abs(values))),
        "maximum_phase_increment": maximum_increment,
        # A stable rounded integer is not enough to certify the contour.  The
        # publication workflow also requires every resolved phase step to be
        # smaller than pi/2 so a missed wrap is unlikely.
        "phase_resolution_pass": bool(maximum_increment < 0.5 * np.pi),
    }


def count_modes_adaptive(
    background: RadialBackground,
    *,
    real_bounds: tuple[float, float],
    imaginary_bounds: tuple[float, float],
    initial_points_per_edge: int = 8,
    maximum_phase_step: float = 0.5 * np.pi,
    maximum_refinements: int = 8,
    matching_options: dict[str, float] | None = None,
    determinant: Callable[..., complex] = matching_evans_determinant,
) -> dict[str, float | int | bool]:
    """Adaptively resolve a determinant contour before counting its zeros.

    Only segments whose principal phase increment violates the declared limit
    are bisected.  This is substantially cheaper than uniformly oversampling a
    contour whose rapid phase turn is localized near one boundary point.
    """

    points = list(
        rectangular_contour(real_bounds, imaginary_bounds, initial_points_per_edge)
    )
    options = dict(matching_options or {})
    cache: dict[complex, complex] = {}

    def value(point: complex) -> complex:
        key = complex(point)
        if key not in cache:
            cache[key] = determinant(key, background, **options)
        return cache[key]

    initial_values = np.asarray([value(point) for point in points])
    initial_increments = np.angle(np.roll(initial_values, -1) / initial_values)
    initial_winding = int(
        np.rint(np.sum(initial_increments) / (2.0 * np.pi))
    )
    initial_maximum_increment = float(np.max(np.abs(initial_increments)))
    refinements = 0
    for _ in range(maximum_refinements + 1):
        insert_after: dict[int, complex] = {}
        for index, left in enumerate(points):
            right = points[(index + 1) % len(points)]
            increment = abs(np.angle(value(right) / value(left)))
            if increment >= maximum_phase_step:
                insert_after[index] = 0.5 * (left + right)
        if not insert_after:
            break
        if refinements >= maximum_refinements:
            break
        refined: list[complex] = []
        for index, point in enumerate(points):
            refined.append(point)
            if index in insert_after:
                refined.append(insert_after[index])
        points = refined
        refinements += 1

    values = np.asarray([value(point) for point in points])
    ratios = np.roll(values, -1) / values
    increments = np.angle(ratios)
    maximum_increment = float(np.max(np.abs(increments)))
    winding = int(np.rint(np.sum(increments) / (2.0 * np.pi)))
    return {
        "winding_number": winding,
        "initial_winding_number": initial_winding,
        "initial_points_per_edge": initial_points_per_edge,
        "initial_maximum_phase_increment": initial_maximum_increment,
        "final_contour_points": len(points),
        "refinement_levels": refinements,
        "minimum_boundary_determinant": float(np.min(np.abs(values))),
        "maximum_phase_increment": maximum_increment,
        "phase_resolution_pass": bool(maximum_increment < maximum_phase_step),
    }
