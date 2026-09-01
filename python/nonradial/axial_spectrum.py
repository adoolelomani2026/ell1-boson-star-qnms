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
from nonradial.riemann_sheet import SidebandSheet


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


@dataclass(frozen=True)
class SpectrumCell:
    """One rectangular cell in a declared complex-frequency census."""

    real_bounds: tuple[float, float]
    imaginary_bounds: tuple[float, float]
    depth: int = 0

    @property
    def center(self) -> complex:
        return complex(
            0.5 * sum(self.real_bounds), 0.5 * sum(self.imaginary_bounds)
        )

    def contains(self, point: complex, padding: float = 0.0) -> bool:
        return (
            self.real_bounds[0] - padding <= point.real <= self.real_bounds[1] + padding
            and self.imaginary_bounds[0] - padding <= point.imag <= self.imaginary_bounds[1] + padding
        )

    def split(self) -> tuple["SpectrumCell", ...]:
        xm = 0.5 * sum(self.real_bounds)
        ym = 0.5 * sum(self.imaginary_bounds)
        x0, x1 = self.real_bounds
        y0, y1 = self.imaginary_bounds
        depth = self.depth + 1
        return (
            SpectrumCell((x0, xm), (y0, ym), depth),
            SpectrumCell((xm, x1), (y0, ym), depth),
            SpectrumCell((xm, x1), (ym, y1), depth),
            SpectrumCell((x0, xm), (ym, y1), depth),
        )


@dataclass(frozen=True)
class CensusLeaf:
    cell: SpectrumCell
    winding_number: int
    maximum_phase_increment: float
    minimum_boundary_determinant: float
    pole: complex | None
    relative_residual: float | None
    status: str


@dataclass(frozen=True)
class QuadtreeCensus:
    sheet_name: str
    leaves: tuple[CensusLeaf, ...]
    excluded_cut_cells: tuple[SpectrumCell, ...]
    determinant_evaluations: int

    @property
    def counted_zeros(self) -> int:
        return sum(leaf.winding_number for leaf in self.leaves)

    @property
    def assigned_poles(self) -> tuple[complex, ...]:
        return tuple(leaf.pole for leaf in self.leaves if leaf.pole is not None)

    @property
    def complete(self) -> bool:
        return not self.excluded_cut_cells and all(
            leaf.status in {"empty", "assigned"} for leaf in self.leaves
        )


class CachedDeterminant:
    """Shared determinant cache for neighboring quadtree contours."""

    def __init__(self, determinant: Callable[[complex], complex]):
        self.determinant = determinant
        self.cache: dict[complex, complex] = {}

    def __call__(self, point: complex) -> complex:
        key = complex(point)
        if key not in self.cache:
            value = complex(self.determinant(key))
            if not np.isfinite(value.real) or not np.isfinite(value.imag):
                raise ValueError(f"invalid determinant sample at {key!r}")
            self.cache[key] = value
        return self.cache[key]

    @property
    def evaluations(self) -> int:
        return len(self.cache)


def _count_cached_cell(
    evaluator: CachedDeterminant,
    cell: SpectrumCell,
    *,
    initial_points_per_edge: int,
    maximum_phase_step: float,
    maximum_refinements: int,
    minimum_uniform_refinements: int,
) -> tuple[int, float, float, bool, complex | None]:
    """Adaptively count one cell while sharing samples with all other cells."""

    points = list(
        rectangular_contour(
            cell.real_bounds, cell.imaginary_bounds, initial_points_per_edge
        )
    )
    for refinement in range(maximum_refinements + 1):
        values = np.asarray([evaluator(point) for point in points])
        if np.any(values == 0.0):
            raise ValueError("a determinant zero lies on a census boundary")
        increments = np.angle(np.roll(values, -1) / values)
        if refinement < minimum_uniform_refinements:
            violating = np.arange(len(points))
        else:
            violating = np.flatnonzero(np.abs(increments) >= maximum_phase_step)
        if len(violating) == 0:
            break
        if refinement == maximum_refinements:
            return (
                int(np.rint(np.sum(increments) / (2.0 * np.pi))),
                float(np.max(np.abs(increments))),
                float(np.min(np.abs(values))),
                False,
                None,
            )
        insert_after = set(int(index) for index in violating)
        refined: list[complex] = []
        for index, point in enumerate(points):
            refined.append(point)
            if index in insert_after:
                refined.append(0.5 * (point + points[(index + 1) % len(points)]))
        points = refined
    winding = int(np.rint(np.sum(increments) / (2.0 * np.pi)))
    moment_seed = None
    if winding != 0:
        contour = np.asarray(points, dtype=complex)
        ratios = np.roll(values, -1) / values
        log_increments = np.log(np.abs(ratios)) + 1.0j * np.angle(ratios)
        midpoints = 0.5 * (contour + np.roll(contour, -1))
        moment_seed = complex(
            np.sum(midpoints * log_increments) / (2.0j * np.pi * winding)
        )
    return (
        winding,
        float(np.max(np.abs(increments))),
        float(np.min(np.abs(values))),
        True,
        moment_seed,
    )


def _refine_cached_cell_zero(
    evaluator: CachedDeterminant,
    cell: SpectrumCell,
    boundary_scale: float,
    moment_seed: complex | None = None,
    *,
    maximum_iterations: int = 16,
    relative_tolerance: float = 1.0e-7,
) -> tuple[complex | None, float | None]:
    """Assign a winding-one cell to a converged simple determinant zero."""

    x0, x1 = cell.real_bounds
    y0, y1 = cell.imaginary_bounds
    width = x1 - x0
    height = y1 - y0
    derivative_step = max(min(width, height) * 1.0e-3, 1.0e-7)
    diagonal = abs(complex(width, height))
    seeds = (
        *(tuple() if moment_seed is None else (moment_seed,)),
        cell.center,
        complex(x0 + 0.25 * width, y0 + 0.25 * height),
        complex(x0 + 0.75 * width, y0 + 0.25 * height),
        complex(x0 + 0.75 * width, y0 + 0.75 * height),
        complex(x0 + 0.25 * width, y0 + 0.75 * height),
    )
    best: tuple[complex, float] | None = None
    for seed in seeds:
        pole = seed
        for _ in range(maximum_iterations):
            value = evaluator(pole)
            derivative_x = (
                evaluator(pole + derivative_step)
                - evaluator(pole - derivative_step)
            ) / (2.0 * derivative_step)
            derivative_y = (
                evaluator(pole + 1j * derivative_step)
                - evaluator(pole - 1j * derivative_step)
            ) / (2.0 * derivative_step)
            jacobian = np.asarray(
                (
                    (derivative_x.real, derivative_y.real),
                    (derivative_x.imag, derivative_y.imag),
                )
            )
            if not np.all(np.isfinite(jacobian)) or abs(np.linalg.det(jacobian)) < 1e-30:
                break
            step_vector = np.linalg.solve(
                jacobian, np.asarray((value.real, value.imag))
            )
            step = complex(step_vector[0], step_vector[1])
            if abs(step) > 0.75 * diagonal:
                step *= 0.75 * diagonal / abs(step)
            pole -= step
            if abs(step) < 1.0e-11 * max(1.0, abs(pole)):
                break
        residual = abs(evaluator(pole)) / max(boundary_scale, 1.0e-300)
        if cell.contains(pole, padding=1.0e-10 * diagonal):
            if best is None or residual < best[1]:
                best = (complex(pole), float(residual))
    if best is None or best[1] >= relative_tolerance:
        return None, None if best is None else best[1]
    return best


def quadtree_census(
    background: RadialBackground,
    *,
    real_bounds: tuple[float, float],
    imaginary_bounds: tuple[float, float],
    sheet: SidebandSheet,
    matching_options: dict[str, object] | None = None,
    determinant: Callable[..., complex] = matching_evans_determinant,
    initial_points_per_edge: int = 4,
    maximum_phase_step: float = 0.25 * np.pi,
    maximum_contour_refinements: int = 10,
    minimum_uniform_refinements: int = 1,
    maximum_depth: int = 8,
) -> QuadtreeCensus:
    """Recursively census a cut-free rectangle until every pole is assigned.

    Cells intersecting a declared sideband cut are explicitly excluded and
    make the result incomplete.  A later keyhole contour can replace those
    exclusions without changing the sheet or census data model.
    """

    options = dict(matching_options or {})
    if minimum_uniform_refinements < 0:
        raise ValueError("minimum_uniform_refinements must be nonnegative")
    if minimum_uniform_refinements > maximum_contour_refinements:
        raise ValueError("uniform refinements exceed contour refinement cap")
    options["sheet"] = sheet
    evaluator = CachedDeterminant(
        lambda sigma: determinant(sigma, background, **options)
    )
    pending = [SpectrumCell(real_bounds, imaginary_bounds)]
    leaves: list[CensusLeaf] = []
    excluded: list[SpectrumCell] = []
    while pending:
        cell = pending.pop()
        if sheet.cell_intersects_cut(
            cell.real_bounds, cell.imaginary_bounds, background.omega
        ):
            excluded.append(cell)
            continue
        winding, phase_step, boundary_minimum, resolved, moment_seed = _count_cached_cell(
            evaluator,
            cell,
            initial_points_per_edge=initial_points_per_edge,
            maximum_phase_step=maximum_phase_step,
            maximum_refinements=maximum_contour_refinements,
            minimum_uniform_refinements=minimum_uniform_refinements,
        )
        if not resolved:
            leaves.append(
                CensusLeaf(
                    cell, winding, phase_step, boundary_minimum, None, None,
                    "unresolved-phase",
                )
            )
        elif winding == 0:
            leaves.append(
                CensusLeaf(
                    cell, winding, phase_step, boundary_minimum, None, None,
                    "empty",
                )
            )
        elif winding == 1:
            pole, residual = _refine_cached_cell_zero(
                evaluator, cell, boundary_minimum, moment_seed
            )
            if pole is not None:
                leaves.append(
                    CensusLeaf(
                        cell, winding, phase_step, boundary_minimum, pole,
                        residual, "assigned",
                    )
                )
            elif cell.depth < maximum_depth:
                pending.extend(reversed(cell.split()))
            else:
                leaves.append(
                    CensusLeaf(
                        cell, winding, phase_step, boundary_minimum, None,
                        residual, "unassigned",
                    )
                )
        elif cell.depth < maximum_depth:
            pending.extend(reversed(cell.split()))
        else:
            leaves.append(
                CensusLeaf(
                    cell, winding, phase_step, boundary_minimum, None, None,
                    "multiple-or-meromorphic",
                )
            )
    return QuadtreeCensus(
        sheet_name=sheet.name,
        leaves=tuple(leaves),
        excluded_cut_cells=tuple(excluded),
        determinant_evaluations=evaluator.evaluations,
    )


def refine_scaled_root(
    seed: complex,
    background: RadialBackground,
    *,
    matching_options: dict[str, object] | None = None,
    determinant: Callable[..., complex] = matching_raw_determinant,
    derivative_step: float = 2.0e-7,
    step_tolerance: float = 2.0e-12,
    residual_tolerance: float = 2.0e-5,
    maximum_iterations: int = 6,
) -> ComplexRootRefinement:
    """Refine a simple zero as two real equations in ``(Re sigma, Im sigma)``.

    Exterior integrations are rescaled by positive real magnitudes to avoid
    overflow.  Those factors preserve zeros and contour phase but need not be
    holomorphic, so a two-dimensional finite-difference Jacobian is used
    instead of assuming the Cauchy--Riemann equations.
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
    step = np.asarray((np.inf, np.inf), dtype=float)
    for iteration in range(1, maximum_iterations + 1):
        value = determinant(pole, background, **options)
        x_plus = determinant(pole + derivative_step, background, **options)
        x_minus = determinant(pole - derivative_step, background, **options)
        y_plus = determinant(pole + 1j * derivative_step, background, **options)
        y_minus = determinant(pole - 1j * derivative_step, background, **options)
        evaluations += 5
        derivative_x = (x_plus - x_minus) / (2.0 * derivative_step)
        derivative_y = (y_plus - y_minus) / (2.0 * derivative_step)
        jacobian = np.asarray(
            (
                (derivative_x.real, derivative_y.real),
                (derivative_x.imag, derivative_y.imag),
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(jacobian)) or abs(np.linalg.det(jacobian)) < 1.0e-30:
            break
        step = np.linalg.solve(jacobian, np.asarray((value.real, value.imag)))
        maximum_step = 0.1 * max(1.0, abs(pole))
        if np.linalg.norm(step) > maximum_step:
            step *= maximum_step / np.linalg.norm(step)
        pole -= complex(step[0], step[1])
        if np.linalg.norm(step) < step_tolerance:
            break
    residual = abs(determinant(pole, background, **options)) / reference
    evaluations += 1
    converged = bool(np.linalg.norm(step) < step_tolerance and residual < residual_tolerance)
    return ComplexRootRefinement(
        pole=pole,
        relative_residual=float(residual),
        converged=converged,
        iterations=iteration,
        evaluations=evaluations,
    )


def refine_analytic_root(*args, **kwargs) -> ComplexRootRefinement:
    """Backward-compatible name for :func:`refine_scaled_root`.

    The implementation deliberately makes no analyticity assumption.
    """

    return refine_scaled_root(*args, **kwargs)


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
    matching_options: dict[str, object] | None = None,
) -> tuple[AxialModeSearch, list[dict[str, float]]]:
    """Discover a pole from a declared grid, then refine the raw determinant.

    No stored frequency is read.  The least normalized singular value chooses
    the initial seed; the phase-preserving matching determinant supplies the
    two real equations for refinement.
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
    refinement = refine_scaled_root(
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
    matching_options: dict[str, object] | None = None,
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
    minimum_uniform_refinements: int = 1,
    required_stable_refinements: int = 1,
    matching_options: dict[str, object] | None = None,
    determinant: Callable[..., complex] = matching_evans_determinant,
) -> dict[str, float | int | bool]:
    """Adaptively resolve a determinant contour before counting its zeros.

    Exactly ``minimum_uniform_refinements`` initial levels bisect every segment.
    Thereafter only segments violating the phase bound are bisected.  The
    uniform checks prevent an entire 2-pi turn between two endpoints from
    masquerading as a small principal increment, while the fixed number of
    global levels bounds the cost near thresholds.  If the requested count
    stability is not reached, the result explicitly fails acceptance.
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
    if minimum_uniform_refinements < 0 or required_stable_refinements < 1:
        raise ValueError("invalid refinement-stability requirements")
    if minimum_uniform_refinements > maximum_refinements:
        raise ValueError("uniform refinements cannot exceed maximum refinements")
    refinements = 0
    winding_history = [initial_winding]
    for _ in range(maximum_refinements + 1):
        insert_after: dict[int, complex] = {}
        for index, left in enumerate(points):
            right = points[(index + 1) % len(points)]
            increment = abs(np.angle(value(right) / value(left)))
            if (
                refinements < minimum_uniform_refinements
                or increment >= maximum_phase_step
            ):
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
        level_values = np.asarray([value(point) for point in points])
        level_increments = np.angle(np.roll(level_values, -1) / level_values)
        winding_history.append(
            int(np.rint(np.sum(level_increments) / (2.0 * np.pi)))
        )

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
        "winding_history": winding_history,
        "count_stability_pass": bool(
            len(winding_history) > required_stable_refinements
            and len(set(winding_history[-(required_stable_refinements + 1) :])) == 1
        ),
        "minimum_boundary_determinant": float(np.min(np.abs(values))),
        "maximum_phase_increment": maximum_increment,
        "phase_resolution_pass": bool(maximum_increment < maximum_phase_step),
    }
