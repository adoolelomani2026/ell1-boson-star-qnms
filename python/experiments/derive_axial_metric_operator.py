"""Symbolically derive the J=2 odd metric operator on a spherical background.

The calculation is deliberately independent of the published component
equations.  It linearizes the Einstein tensor in a bookkeeping parameter for
the M=0 Regge--Wheeler-gauge metric.  Rotational covariance then fixes the same
radial operator for every M.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sympy as sp


def _christoffel(metric: sp.Matrix, inverse: sp.Matrix, coords: tuple[sp.Symbol, ...]):
    dimension = len(coords)
    result = [
        [[sp.S.Zero for _ in range(dimension)] for _ in range(dimension)]
        for _ in range(dimension)
    ]
    for a in range(dimension):
        for b in range(dimension):
            for c in range(dimension):
                result[a][b][c] = sp.simplify(
                    sum(
                        inverse[a, d]
                        * (
                            sp.diff(metric[d, c], coords[b])
                            + sp.diff(metric[d, b], coords[c])
                            - sp.diff(metric[b, c], coords[d])
                        )
                        for d in range(dimension)
                    )
                    / 2
                )
    return result


def derive_axial_metric_operator() -> dict[str, sp.Expr]:
    t, r, theta, phi = sp.symbols("t r theta phi", real=True)
    sigma = sp.symbols("sigma", complex=True)
    coords = (t, r, theta, phi)
    lapse = sp.Function("alpha")(r)
    radial_metric = sp.Function("gamma")(r)
    h0 = sp.Function("h0")(r)
    h1 = sp.Function("h1")(r)
    time_factor = sp.exp(-sp.I * sigma * t)

    background = sp.diag(
        -lapse**2,
        radial_metric**2,
        r**2,
        r**2 * sp.sin(theta) ** 2,
    )
    inverse = sp.simplify(background.inv())

    # The overall normalization of X_phi cancels from every projected ratio.
    # X_phi proportional to sin(theta)^2 cos(theta) for J=2, M=0.
    x_phi = sp.sin(theta) ** 2 * sp.cos(theta)
    perturbation = sp.zeros(4)
    perturbation[0, 3] = perturbation[3, 0] = h0 * x_phi * time_factor
    perturbation[1, 3] = perturbation[3, 1] = h1 * x_phi * time_factor
    delta_inverse = -inverse * perturbation * inverse

    gamma0 = _christoffel(background, inverse, coords)
    dimension = 4
    delta_gamma = [
        [[sp.S.Zero for _ in range(dimension)] for _ in range(dimension)]
        for _ in range(dimension)
    ]
    for a in range(dimension):
        for b in range(dimension):
            for c in range(dimension):
                inverse_term = sum(
                    delta_inverse[a, d]
                    * (
                        sp.diff(background[d, c], coords[b])
                        + sp.diff(background[d, b], coords[c])
                        - sp.diff(background[b, c], coords[d])
                    )
                    for d in range(dimension)
                )
                metric_term = sum(
                    inverse[a, d]
                    * (
                        sp.diff(perturbation[d, c], coords[b])
                        + sp.diff(perturbation[d, b], coords[c])
                        - sp.diff(perturbation[b, c], coords[d])
                    )
                    for d in range(dimension)
                )
                delta_gamma[a][b][c] = sp.simplify((inverse_term + metric_term) / 2)

    ricci0 = sp.MutableDenseMatrix(4, 4, [0] * 16)
    delta_ricci = sp.MutableDenseMatrix(4, 4, [0] * 16)
    for a in range(dimension):
        for b in range(dimension):
            ricci0[a, b] = sp.simplify(
                sum(
                    sp.diff(gamma0[c][a][b], coords[c])
                    - sp.diff(gamma0[c][a][c], coords[b])
                    + sum(
                        gamma0[c][a][b] * gamma0[d][c][d]
                        - gamma0[c][a][d] * gamma0[d][b][c]
                        for d in range(dimension)
                    )
                    for c in range(dimension)
                )
            )
            delta_ricci[a, b] = sp.simplify(
                sum(
                    sp.diff(delta_gamma[c][a][b], coords[c])
                    - sp.diff(delta_gamma[c][a][c], coords[b])
                    + sum(
                        delta_gamma[c][a][b] * gamma0[d][c][d]
                        + gamma0[c][a][b] * delta_gamma[d][c][d]
                        - delta_gamma[c][a][d] * gamma0[d][b][c]
                        - gamma0[c][a][d] * delta_gamma[d][b][c]
                        for d in range(dimension)
                    )
                    for c in range(dimension)
                )
            )

    scalar0 = sp.simplify(sum(inverse[a, b] * ricci0[a, b] for a in range(4) for b in range(4)))
    delta_scalar = sp.simplify(
        sum(
            delta_inverse[a, b] * ricci0[a, b]
            + inverse[a, b] * delta_ricci[a, b]
            for a in range(4)
            for b in range(4)
        )
    )

    delta_einstein = sp.MutableDenseMatrix(4, 4, [0] * 16)
    for a in range(4):
        for b in range(4):
            delta_einstein[a, b] = sp.simplify(
                delta_ricci[a, b]
                - perturbation[a, b] * scalar0 / 2
                - background[a, b] * delta_scalar / 2
            )

    x_theta_phi = sp.simplify(
        (
            sp.diff(x_phi, theta)
            - 2 * sp.cot(theta) * x_phi
        )
        / 2
    )
    projected = {
        "delta_G_tA": sp.simplify(
            delta_einstein[0, 3] / (x_phi * time_factor)
        ),
        "delta_G_rA": sp.simplify(
            delta_einstein[1, 3] / (x_phi * time_factor)
        ),
        "delta_G_AB": sp.simplify(
            delta_einstein[2, 3] / (x_theta_phi * time_factor)
        ),
    }
    return projected


def main() -> None:
    equations = derive_axial_metric_operator()
    output = ROOT / "symbolic" / "axial_metric_operator_symbolic.txt"
    lines = []
    for name, expression in equations.items():
        lines.extend((name, sp.sstr(expression), ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    for name, expression in equations.items():
        print(f"{name} =")
        print(sp.pretty(expression))
        print()


if __name__ == "__main__":
    main()
