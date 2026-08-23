"""Resolved node finding and eigenfunction-overlap mode tracking."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


def resolved_zero_locations(
    radius: np.ndarray,
    values: np.ndarray,
    *,
    relative_floor: float = 1e-11,
) -> np.ndarray:
    """Locate resolved interior zeros by shape-preserving interpolation.

    The physical endpoint zeros are excluded. Only samples indistinguishable
    from floating-point/interpolation noise are removed; unlike the historical
    ``1e-4`` amplitude cut, this retains low-amplitude lobes of higher modes.
    """
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)
    if radius.ndim != 1 or values.shape != radius.shape or radius.size < 4:
        raise ValueError("radius and values must be matching one-dimensional arrays")
    order = np.argsort(radius)
    radius = radius[order]
    values = values[order]
    if np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius samples must be distinct")
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale == 0.0:
        return np.empty(0)
    floor = max(relative_floor * scale, 64.0 * np.finfo(float).eps * scale)
    interpolant = PchipInterpolator(radius, values, extrapolate=False)
    roots: list[float] = []
    # Skip intervals touching either imposed physical endpoint.
    for left in range(1, radius.size - 2):
        right = left + 1
        yl, yr = values[left], values[right]
        if max(abs(yl), abs(yr)) <= floor or yl * yr > 0.0:
            continue
        if abs(yl) <= floor:
            root = radius[left]
        elif abs(yr) <= floor:
            root = radius[right]
        else:
            root = brentq(interpolant, radius[left], radius[right])
        if not roots or abs(root - roots[-1]) > 1e-8 * max(1.0, radius[-1]):
            roots.append(float(root))
    return np.asarray(roots)


def node_count(radius: np.ndarray, values: np.ndarray) -> int:
    return int(resolved_zero_locations(radius, values).size)


def eigenfunction_overlap(
    reference_radius: np.ndarray,
    reference_values: np.ndarray,
    candidate_radius: np.ndarray,
    candidate_values: np.ndarray,
) -> float:
    """Return the absolute normalized L2 overlap on the common radial domain."""
    reference_radius = np.asarray(reference_radius, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    order = np.argsort(candidate_radius)
    candidate_radius = np.asarray(candidate_radius, dtype=float)[order]
    candidate_values = np.asarray(candidate_values, dtype=float)[order]
    lo = max(float(reference_radius[0]), float(candidate_radius[0]))
    hi = min(float(reference_radius[-1]), float(candidate_radius[-1]))
    mask = (reference_radius >= lo) & (reference_radius <= hi)
    radius = reference_radius[mask]
    left = reference_values[mask]
    right = PchipInterpolator(candidate_radius, candidate_values)(radius)
    numerator = np.trapezoid(left * right, radius)
    denominator = np.sqrt(
        np.trapezoid(left**2, radius) * np.trapezoid(right**2, radius)
    )
    return float(abs(numerator) / max(float(denominator), 1e-300))


def track_mode_by_overlap(reference, candidates, *, field: str = "delta_lambda"):
    """Select a continued mode by maximum physical-eigenfunction overlap."""
    if not candidates:
        raise ValueError("at least one candidate mode is required")
    overlaps = [
        eigenfunction_overlap(
            reference.r,
            getattr(reference, field),
            candidate.r,
            getattr(candidate, field),
        )
        for candidate in candidates
    ]
    index = int(np.argmax(overlaps))
    return candidates[index], overlaps[index]
