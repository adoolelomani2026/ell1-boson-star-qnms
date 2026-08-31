"""Explicit massive-sideband Riemann sheets for nonradial spectra.

The square roots are represented as products of principal roots,

    k(q) = sqrt(q-mu) sqrt(q+mu),

which places the finite cut on ``q in [-mu,mu]``.  A pair of signs then
selects one of the four sheets of the two-sideband problem.  This avoids the
nonanalytic pointwise rule ``Im(k)>=0`` inside argument-principle contours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SidebandSheet:
    """Sheet signs relative to the finite-cut product square root."""

    plus_sign: int
    minus_sign: int
    name: str = "declared"

    def __post_init__(self) -> None:
        if self.plus_sign not in (-1, 1) or self.minus_sign not in (-1, 1):
            raise ValueError("sideband sheet signs must be +1 or -1")

    @classmethod
    def physical_lower_half_plane(cls) -> "SidebandSheet":
        """Outgoing/decaying sheet analytically continued below real sigma."""

        return cls(+1, -1, "physical-lower")

    @classmethod
    def physical_upper_half_plane(cls) -> "SidebandSheet":
        """Conjugate physical sheet analytically continued above real sigma."""

        return cls(-1, +1, "physical-upper")

    @staticmethod
    def _finite_cut_root(channel_frequency: complex, mass: float) -> complex:
        q = complex(channel_frequency)
        return complex(np.sqrt(q - mass + 0.0j) * np.sqrt(q + mass + 0.0j))

    def wavenumbers(
        self, sigma: complex, omega: float, mass: float = 1.0
    ) -> tuple[complex, complex]:
        """Return ``(k_plus,k_minus)`` on this declared sheet."""

        k_plus = self.plus_sign * self._finite_cut_root(omega - sigma, mass)
        k_minus = self.minus_sign * self._finite_cut_root(omega + sigma, mass)
        return complex(k_plus), complex(k_minus)

    @staticmethod
    def branch_points(omega: float, mass: float = 1.0) -> dict[str, tuple[float, float]]:
        """Return the two real branch-point pairs in the sigma plane."""

        return {
            "plus": (omega - mass, omega + mass),
            "minus": (-omega - mass, -omega + mass),
        }

    def cell_intersects_cut(
        self,
        real_bounds: tuple[float, float],
        imaginary_bounds: tuple[float, float],
        omega: float,
        mass: float = 1.0,
    ) -> bool:
        """Whether a closed rectangular cell intersects either finite cut."""

        x0, x1 = real_bounds
        y0, y1 = imaginary_bounds
        if not (y0 <= 0.0 <= y1):
            return False
        return any(x0 <= right and left <= x1 for left, right in self.branch_points(omega, mass).values())

