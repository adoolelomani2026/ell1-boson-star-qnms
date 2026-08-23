"""Generate the independent Schwarzschild Regge--Wheeler benchmark record."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonradial.schwarzschild_rw import discover_fundamental_mode


def main() -> None:
    options = {"mass": 1.0, "ell": 2, "spin": 2, "terms": 300}
    mode, scan = discover_fundamental_mode(**options)
    literature = complex(0.37367168, -0.08896232)
    result = {
        "calculation": "independent Schwarzschild ell=2 Regge-Wheeler QNM control",
        "boundary_conditions": "Leaver minimal solution: ingoing horizon and outgoing infinity",
        "search_bounds": {"real": [0.34, 0.41], "imaginary": [-0.115, -0.065]},
        "scan_shape": [6, 5],
        "seed": {"real": mode.seed.real, "imaginary": mode.seed.imag},
        "frequency": {"real": mode.frequency.real, "imaginary": mode.frequency.imag},
        "relative_mismatch": mode.residual,
        "solver_success": mode.success,
        "function_evaluations": mode.evaluations,
        "literature_reference": {"real": literature.real, "imaginary": literature.imag},
        "absolute_error": abs(mode.frequency - literature),
        "options": options,
        "scan": scan,
    }
    target = ROOT / "reports" / "controls" / "schwarzschild_qnm_benchmark.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("frequency", "absolute_error", "relative_mismatch")}, indent=2))


if __name__ == "__main__":
    main()
