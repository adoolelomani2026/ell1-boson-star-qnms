"""Stationary odd-parity tidal-response checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background.ell_boson_star import BackgroundSolution
from nonradial.axial_static import static_tidal_response
from radial.coefficients import RadialBackground


def main() -> None:
    solution = BackgroundSolution.load(ROOT / "reports" / "axial" / "background_ell1_a008_axial_refined.npz")
    background = RadialBackground(solution, geometry_representation="hermite")
    domains = []
    for r_match, r_end in (
        (10.0, 35.0),
        (12.0, 35.0),
        (14.0, 35.0),
        (16.0, 35.0),
        (14.0, 30.0),
        (14.0, 40.0),
    ):
        row = {"r_match": r_match, "r_end": r_end}
        row.update(
            static_tidal_response(
                background,
                r_match=r_match,
                r_end=r_end,
                rtol=2e-11,
                atol=2e-13,
            )
        )
        domains.append(row)
    values = [row["dimensionless_B_over_A_M5"] for row in domains]
    result = {
        "calculation": "neutral ell=1 J=L=2 stationary axial response",
        "definition": "h0=A r^2(r-2M)+B r^-2[1+O(M/r)]; response=B/(A M^5)",
        "background_a1_0": solution.a0,
        "domains": domains,
        "mean_dimensionless_response": sum(values) / len(values),
        "domain_spread": max(values) - min(values),
    }
    target = ROOT / "reports" / "axial" / "axial_static_response.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
