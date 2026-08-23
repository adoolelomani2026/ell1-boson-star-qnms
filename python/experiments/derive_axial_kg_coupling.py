"""Independently derive the odd metric coupling in the KG equation.

The background scalar is one representative member of the internal ell=1
triplet.  The M=0 axial metric harmonic is sufficient because rotational
covariance fixes the radial operator for every M.  This script checks the
coordinate calculation against the compact hand-derived expression.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sympy as sp

from experiments.derive_axial_metric_operator import _christoffel


def derive_kg_metric_coupling() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    t, r, theta, phi = sp.symbols("t r theta phi", real=True)
    sigma, omega = sp.symbols("sigma omega", complex=True)
    coords = (t, r, theta, phi)
    alpha = sp.Function("alpha")(r)
    gamma = sp.Function("gamma")(r)
    field = sp.Function("F")(r)
    h0 = sp.Function("h0")(r)
    h1 = sp.Function("h1")(r)

    background = sp.diag(-alpha**2, gamma**2, r**2, r**2 * sp.sin(theta) ** 2)
    inverse = background.inv()
    x_phi = sp.sin(theta) ** 2 * sp.cos(theta)
    time_factor = sp.exp(-sp.I * sigma * t)
    perturbation = sp.zeros(4)
    perturbation[0, 3] = perturbation[3, 0] = h0 * x_phi * time_factor
    perturbation[1, 3] = perturbation[3, 1] = h1 * x_phi * time_factor
    delta_inverse = -inverse * perturbation * inverse

    gamma0 = _christoffel(background, inverse, coords)
    delta_gamma = [[[sp.S.Zero for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b in range(4):
            for c in range(4):
                inverse_term = sum(
                    delta_inverse[a, d]
                    * (
                        sp.diff(background[d, c], coords[b])
                        + sp.diff(background[d, b], coords[c])
                        - sp.diff(background[b, c], coords[d])
                    )
                    for d in range(4)
                )
                metric_term = sum(
                    inverse[a, d]
                    * (
                        sp.diff(perturbation[d, c], coords[b])
                        + sp.diff(perturbation[d, b], coords[c])
                        - sp.diff(perturbation[b, c], coords[d])
                    )
                    for d in range(4)
                )
                delta_gamma[a][b][c] = sp.simplify((inverse_term + metric_term) / 2)

    # A single complex ell=1 angular function is enough to expose X^A D_A B.
    angular_background = sp.sin(theta) * sp.exp(sp.I * phi)
    scalar_background = sp.exp(sp.I * omega * t) * field * angular_background
    derivatives = [sp.diff(scalar_background, coordinate) for coordinate in coords]

    delta_box = sp.S.Zero
    for mu in range(4):
        for nu in range(4):
            covariant_hessian = sp.diff(scalar_background, coords[mu], coords[nu])
            covariant_hessian -= sum(gamma0[lam][mu][nu] * derivatives[lam] for lam in range(4))
            delta_box += delta_inverse[mu, nu] * covariant_hessian
            delta_box -= inverse[mu, nu] * sum(
                delta_gamma[lam][mu][nu] * derivatives[lam] for lam in range(4)
            )

    angular_advection = x_phi * sp.diff(angular_background, phi) / sp.sin(theta) ** 2
    common = sp.exp(sp.I * (omega - sigma) * t) * angular_advection / r**2
    derived = sp.simplify(delta_box / common)
    expected = (
        sp.I * field * (2 * omega - sigma) * h0 / alpha**2
        - (
            field * sp.diff(h1, r)
            + (
                2 * sp.diff(field, r)
                + field * (sp.diff(alpha, r) / alpha - sp.diff(gamma, r) / gamma)
            )
            * h1
        )
        / gamma**2
    )
    residual = sp.simplify(sp.expand(derived - expected))
    return derived, expected, residual


def main() -> None:
    derived, expected, residual = derive_kg_metric_coupling()
    if residual != 0:
        raise RuntimeError(f"KG coupling check failed: {residual}")
    output = ROOT / "symbolic" / "axial_kg_coupling_symbolic.txt"
    output.write_text(
        "derived_delta_box_coefficient\n"
        + sp.sstr(derived)
        + "\n\nexpected_coefficient\n"
        + sp.sstr(expected)
        + "\n\nresidual\n0\n",
        encoding="utf-8",
    )
    print("KG metric-coupling identity: PASS")
    print(sp.pretty(derived))


if __name__ == "__main__":
    main()
