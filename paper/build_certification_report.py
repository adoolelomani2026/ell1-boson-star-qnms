"""Build the publication-style background and radial certification report."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from background.ell_boson_star import scan_sequence, solve_background, solve_by_continuation
from radial.bvp import solve_radial_bvp
from radial.mode_tracking import eigenfunction_overlap
from radial.spectral import solve_radial_spectrum


OUTPUT = ROOT / "output" / "pdf" / "boson_star_background_radial_certification_v031.pdf"
ASSETS = ROOT / "tmp" / "pdfs" / "certification_report"
SEQUENCE = ROOT / "reports" / "background" / "ell1_sequence.csv"
RADIAL = ROOT / "reports" / "radial" / "radial_benchmarks.csv"
GROUND_UNCERTAINTY = ROOT / "reports" / "radial" / "radial_uncertainty.json"
OVERTONE_UNCERTAINTY = ROOT / "reports" / "radial" / "radial_overtone_uncertainty.json"
RADIUS_AUDIT = ROOT / "reports" / "background" / "background_radius_audit.json"

NAVY = colors.HexColor("#15324B")
BLUE = colors.HexColor("#1F6F8B")
TEAL = colors.HexColor("#2A9D8F")
GOLD = colors.HexColor("#E9C46A")
RED = colors.HexColor("#C84C4C")
INK = colors.HexColor("#202A33")
MUTED = colors.HexColor("#5D6B78")
PALE = colors.HexColor("#EDF4F6")
GRID = colors.HexColor("#CAD5DB")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#384650",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": "#DCE4E8",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig, name: str) -> Path:
    path = ASSETS / f"{name}.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _build_numerical_assets():
    sequence_rows = _read_csv(SEQUENCE)
    radial_rows = _read_csv(RADIAL)
    radius_audit = json.loads(RADIUS_AUDIT.read_text(encoding="utf-8"))
    ground_uncertainty = json.loads(GROUND_UNCERTAINTY.read_text(encoding="utf-8"))
    overtone_uncertainty = json.loads(OVERTONE_UNCERTAINTY.read_text(encoding="utf-8"))

    # Representative solutions and an independently regenerated radial curve.
    representatives = {
        a0: solve_by_continuation(1, a0) for a0 in (0.04, 0.08, 0.10)
    }
    stability_a0 = np.linspace(0.05, 0.11, 13)
    stability_backgrounds = scan_sequence(1, stability_a0)
    stability = []
    previous = None
    for background in stability_backgrounds:
        spectrum = solve_radial_spectrum(
            background,
            points=80,
            r_max=40.0,
            sigma2_min=-2e-3,
            sigma2_max=0.02,
        )
        if previous is None:
            mode = next(item for item in spectrum if item.node_count == 0)
        else:
            # Frequency continuity is unambiguous for this short ground branch.
            mode = min(spectrum, key=lambda item: abs(item.sigma2 - previous.sigma2))
        stability.append((background, mode))
        previous = mode

    domain_backgrounds = [
        solve_background(1, 0.08, r_max=rmax, points=800, tolerance=1e-7)
        for rmax in (60.0, 80.0, 100.0)
    ]

    benchmark = representatives[0.08]
    ground_bvp = solve_radial_bvp(
        benchmark, r_max=40.0, points=350, tolerance=3e-7
    )
    overtone_bvp = solve_radial_bvp(
        benchmark,
        sigma2_guess=8.2272e-3,
        center_c_guess=-4.19e-2,
        r_max=60.0,
        points=350,
        tolerance=3e-7,
    )
    spectrum40 = solve_radial_spectrum(
        benchmark, points=80, r_max=40.0, sigma2_min=-1e-3, sigma2_max=0.02
    )
    ground_sp = next(item for item in spectrum40 if item.node_count == 0)
    overtone_sp = next(item for item in spectrum40 if item.node_count == 1)

    # Figure 1: global sequence.
    a0 = np.array([float(row["a0"]) for row in sequence_rows])
    omega = np.array([float(row["omega"]) for row in sequence_rows])
    mass = np.array([float(row["adm_mass"]) for row in sequence_rows])
    charge = np.array([float(row["noether_charge"]) for row in sequence_rows])
    r99 = np.array([float(row["r99"]) for row in sequence_rows])
    compactness = np.array([float(row["compactness99"]) for row in sequence_rows])
    imax = int(np.argmax(mass))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), sharex=True)
    for ax, values, ylabel in zip(
        axes.flat,
        (mass, charge, r99, compactness),
        (r"$M\mu$", r"$Q\mu^2$", r"$R_{99}\mu$", r"$C_{99}$"),
    ):
        ax.plot(a0, values, "o-", color="#1F6F8B", ms=3.5, lw=1.4)
        ax.axvline(a0[imax], color="#C84C4C", ls="--", lw=1.0)
        for selected in (0.04, 0.08, 0.10):
            ax.axvline(selected, color="#2A9D8F", alpha=0.18, lw=4)
        ax.set_ylabel(ylabel)
    axes[1, 0].set_xlabel(r"central amplitude $a_1^0$")
    axes[1, 1].set_xlabel(r"central amplitude $a_1^0$")
    axes[0, 0].annotate(
        "grid maximum",
        (a0[imax], mass[imax]),
        xytext=(-55, -24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#C84C4C"},
        color="#8D3030",
    )
    fig.suptitle("Equilibrium sequence and selected certification models", y=1.01)
    fig.tight_layout()
    fig1 = _save_figure(fig, "figure_01_equilibrium_sequence")

    # Figure 2: representative profiles.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    palette = {0.04: "#4C78A8", 0.08: "#2A9D8F", 0.10: "#C84C4C"}
    for amplitude, solution in representatives.items():
        label = rf"$a_1^0={amplitude:.2f}$"
        color = palette[amplitude]
        axes[0, 0].plot(solution.r, solution.psi, label=label, color=color)
        axes[0, 1].plot(solution.r, solution.alpha, label=label, color=color)
        axes[1, 0].plot(solution.r, solution.gamma, label=label, color=color)
        axes[1, 1].plot(
            solution.r,
            solution.mass / solution.adm_mass,
            label=label,
            color=color,
        )
    for ax in axes.flat:
        ax.set_xlim(0, 30)
        ax.set_xlabel(r"$r\mu$")
    axes[0, 0].set_ylabel(r"$\psi_1$")
    axes[0, 1].set_ylabel(r"$\alpha$")
    axes[1, 0].set_ylabel(r"$\gamma$")
    axes[1, 1].set_ylabel(r"$M(r)/M_T$")
    axes[0, 0].legend(ncol=1)
    fig.suptitle(r"Representative relativistic $\ell=1$ boson-star backgrounds", y=1.01)
    fig.tight_layout()
    fig2 = _save_figure(fig, "figure_02_representative_backgrounds")

    # Figure 3: radius audit.
    maximum = representatives[0.10]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fraction = maximum.mass / maximum.adm_mass
    ax.plot(maximum.r, fraction, color="#15324B", lw=1.8)
    ax.axhline(0.99, color="#2A9D8F", ls="--", label="0.99 mass fraction")
    ax.axhline(0.999, color="#E09F3E", ls="--", label="0.999 mass fraction")
    ax.axvline(radius_audit["r99_mass"], color="#2A9D8F", lw=1.2)
    ax.axvline(radius_audit["r999_mass"], color="#E09F3E", lw=1.2)
    ax.axvline(radius_audit["published_radius"], color="#C84C4C", ls=":", lw=1.6)
    ax.annotate(r"$R_{99}=10.18$", (radius_audit["r99_mass"], 0.985), xytext=(-70, -32), textcoords="offset points")
    ax.annotate(r"$R_{999}=12.70$", (radius_audit["r999_mass"], 0.999), xytext=(8, -35), textcoords="offset points")
    ax.annotate("published 12.75", (12.75, 0.997), xytext=(22, -5), textcoords="offset points", color="#8D3030")
    ax.set_xlim(5, 18)
    ax.set_ylim(0.93, 1.002)
    ax.set_xlabel(r"$r\mu$")
    ax.set_ylabel(r"$M(r)/M_T$")
    ax.legend(loc="lower right")
    ax.set_title("Radius-definition audit at the maximum-mass grid model")
    fig.tight_layout()
    fig3 = _save_figure(fig, "figure_03_radius_audit")

    # Figure 4: radial stability curve.
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    stab_a = np.array([item[0].a0 for item in stability])
    stab_sigma2 = np.array([item[1].sigma2 for item in stability])
    ax.axhline(0.0, color="#202A33", lw=0.9)
    ax.plot(stab_a, stab_sigma2, "o-", color="#1F6F8B", lw=1.5, ms=4)
    positive = stab_sigma2 >= 0
    ax.fill_between(stab_a, 0, stab_sigma2, where=positive, color="#2A9D8F", alpha=0.18, label="radially stable")
    ax.fill_between(stab_a, 0, stab_sigma2, where=~positive, color="#C84C4C", alpha=0.16, label="radially unstable")
    ax.axvline(0.10, color="#C84C4C", ls="--", lw=1.0, label="mass maximum grid point")
    ax.set_xlabel(r"central amplitude $a_1^0$")
    ax.set_ylabel(r"ground eigenvalue $\sigma_0^2/\mu^2$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.legend(loc="lower left")
    ax.set_title("Radial stability transition from the independent spectral solver")
    fig.tight_layout()
    fig4 = _save_figure(fig, "figure_04_radial_stability")

    # Figure 5: numerical convergence and conditioning.
    bvp_ground = [
        row
        for row in radial_rows
        if row["formulation"] == "nonlinear_global_bvp"
        and row["background_representation"] == "hermite"
        and row["a0"] == "0.08"
        and row["mode_index"] == "0"
    ]
    sp_ground = [
        row
        for row in radial_rows
        if row["formulation"].startswith("chebyshev")
        and row["a0"] == "0.08"
        and row["r_max"] == "40.0"
        and row["mode_index"] == "0"
    ]
    bvp_over = [
        row
        for row in radial_rows
        if row["formulation"] == "nonlinear_global_bvp"
        and row["a0"] == "0.08"
        and row["mode_index"] == "1"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    axes[0, 0].plot(
        [float(row["r_max"]) for row in bvp_ground],
        [float(row["sigma2"]) for row in bvp_ground],
        "o-",
        color="#1F6F8B",
    )
    axes[0, 0].set_xlabel(r"$r_{max}\mu$")
    axes[0, 0].set_ylabel(r"$\sigma_0^2$")
    axes[0, 0].set_title("BVP outer-domain convergence")
    axes[0, 1].plot(
        [int(row["resolution"].split()[0]) for row in sp_ground],
        [float(row["sigma2"]) for row in sp_ground],
        "o-",
        color="#2A9D8F",
    )
    axes[0, 1].axhline(ground_bvp.sigma2, color="#15324B", ls="--", lw=1)
    axes[0, 1].set_xlabel("Chebyshev points N")
    axes[0, 1].set_ylabel(r"$\sigma_0^2$")
    axes[0, 1].set_title("Spectral resolution envelope")
    axes[1, 0].semilogy(
        [int(row["resolution"].split()[0]) for row in sp_ground],
        [float(row["scaled_generalized_eigen_residual"]) for row in sp_ground],
        "o-",
        label="scaled residual",
        color="#2A9D8F",
    )
    axes[1, 0].semilogy(
        [int(row["resolution"].split()[0]) for row in sp_ground],
        [float(row["unscaled_generalized_eigen_residual"]) for row in sp_ground],
        "s--",
        label="unscaled residual",
        color="#C84C4C",
    )
    axes[1, 0].set_xlabel("Chebyshev points N")
    axes[1, 0].set_ylabel("relative pencil residual")
    axes[1, 0].set_title("Equilibration diagnostic")
    axes[1, 0].legend()
    axes[1, 1].plot(
        [float(row["r_max"]) for row in bvp_over],
        [np.sqrt(float(row["sigma2"])) for row in bvp_over],
        "o-",
        color="#E09F3E",
    )
    axes[1, 1].set_xlabel(r"$r_{max}\mu$")
    axes[1, 1].set_ylabel(r"$\sigma_1/\mu$")
    axes[1, 1].set_title("First-overtone domain convergence")
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    fig.suptitle("Radial numerical certification", y=1.01)
    fig.tight_layout()
    fig5 = _save_figure(fig, "figure_05_radial_convergence")

    # Figure 6: normalized physical eigenfunctions at common domain.
    def normalized(values):
        return values / max(np.max(np.abs(values)), 1e-300)

    def aligned(reference_r, reference_values, candidate_r, candidate_values):
        candidate = normalized(candidate_values)
        reference_on_candidate = np.interp(candidate_r, reference_r, normalized(reference_values))
        if np.trapezoid(reference_on_candidate * candidate, candidate_r) < 0.0:
            candidate = -candidate
        return candidate

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), sharex=True)
    axes[0, 0].plot(ground_bvp.r, normalized(ground_bvp.physical_scalar), color="#15324B", label="BVP")
    axes[0, 0].plot(ground_sp.r, aligned(ground_bvp.r, ground_bvp.physical_scalar, ground_sp.r, ground_sp.physical_scalar), "--", color="#2A9D8F", label="spectral")
    axes[0, 1].plot(ground_bvp.r, normalized(ground_bvp.delta_lambda), color="#15324B")
    axes[0, 1].plot(ground_sp.r, aligned(ground_bvp.r, ground_bvp.delta_lambda, ground_sp.r, ground_sp.delta_lambda), "--", color="#2A9D8F")
    overtone40 = solve_radial_bvp(
        benchmark,
        sigma2_guess=8.2272e-3,
        center_c_guess=-4.19e-2,
        r_max=40.0,
        points=350,
        tolerance=3e-7,
    )
    axes[1, 0].plot(overtone40.r, normalized(overtone40.physical_scalar), color="#15324B")
    axes[1, 0].plot(overtone_sp.r, aligned(overtone40.r, overtone40.physical_scalar, overtone_sp.r, overtone_sp.physical_scalar), "--", color="#E09F3E")
    axes[1, 1].plot(overtone40.r, normalized(overtone40.delta_lambda), color="#15324B")
    axes[1, 1].plot(overtone_sp.r, aligned(overtone40.r, overtone40.delta_lambda, overtone_sp.r, overtone_sp.delta_lambda), "--", color="#E09F3E")
    axes[0, 0].set_title("Ground: scalar field (zero nodes)")
    axes[0, 1].set_title("Ground: metric field")
    axes[1, 0].set_title("Overtone: scalar field (one node)")
    axes[1, 1].set_title("Overtone: metric field")
    for ax in axes.flat:
        ax.set_xlim(0, 30)
        ax.set_xlabel(r"$r\mu$")
        ax.set_ylabel("normalized amplitude")
    axes[0, 0].legend()
    fig.suptitle("BVP-spectral eigenfunction agreement at a common domain", y=1.01)
    fig.tight_layout()
    fig6 = _save_figure(fig, "figure_06_radial_eigenfunctions")

    stability_path = ROOT / "reports" / "radial" / "report_radial_stability.csv"
    with stability_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("a0", "omega", "adm_mass", "sigma2", "node_count", "points", "r_max"),
        )
        writer.writeheader()
        for background, mode in stability:
            writer.writerow(
                {
                    "a0": background.a0,
                    "omega": background.omega,
                    "adm_mass": background.adm_mass,
                    "sigma2": mode.sigma2,
                    "node_count": mode.node_count,
                    "points": 80,
                    "r_max": 40.0,
                }
            )

    return {
        "sequence_rows": sequence_rows,
        "radial_rows": radial_rows,
        "radius_audit": radius_audit,
        "ground_uncertainty": ground_uncertainty,
        "overtone_uncertainty": overtone_uncertainty,
        "representatives": representatives,
        "domain_backgrounds": domain_backgrounds,
        "ground_bvp": ground_bvp,
        "overtone_bvp": overtone_bvp,
        "ground_sp": ground_sp,
        "overtone_sp": overtone_sp,
        "figures": (fig1, fig2, fig3, fig4, fig5, fig6),
    }


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=29,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=MUTED,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=BLUE,
            spaceBefore=9,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.4,
            leading=13.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        "lead": ParagraphStyle(
            "Lead",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=11.2,
            leading=15.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.4,
            textColor=MUTED,
            alignment=TA_JUSTIFY,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "table_caption": ParagraphStyle(
            "TableCaption",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10.5,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=10,
            textColor=INK,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.2,
            leading=11,
            textColor=NAVY,
            backColor=PALE,
            borderPadding=7,
            spaceBefore=5,
            spaceAfter=9,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
            backColor=colors.HexColor("#EAF4F3"),
            borderColor=TEAL,
            borderWidth=0.8,
            borderPadding=9,
            spaceBefore=7,
            spaceAfter=10,
        ),
    }


def _table(data, widths, styles, *, font_size=7.2, alignments=None):
    formatted = []
    for ridx, row in enumerate(data):
        formatted.append(
            [
                Paragraph(str(value), styles["small"] if ridx else ParagraphStyle(
                    f"Header{ridx}{cidx}", parent=styles["small"], fontName="Helvetica-Bold", textColor=colors.white
                ))
                for cidx, value in enumerate(row)
            ]
        )
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for row in range(1, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    if alignments:
        for column, alignment in enumerate(alignments):
            commands.append(("ALIGN", (column, 1), (column, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


def _figure(path: Path, caption: str, number: int, styles):
    image = Image(str(path), width=17.2 * cm, height=13.0 * cm)
    image.hAlign = "CENTER"
    return KeepTogether(
        [
            image,
            Paragraph(f"<b>Figure {number}.</b> {caption}", styles["caption"]),
        ]
    )


def _footer(canvas, document):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(1.8 * cm, 1.45 * cm, width - 1.8 * cm, 1.45 * cm)
    canvas.setFont("Helvetica", 7.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.8 * cm, 1.0 * cm, "Boson Star QNM Project | radial-v0.3.1-hardened")
    page = f"{document.page}"
    canvas.drawRightString(width - 1.8 * cm, 1.0 * cm, page)
    canvas.restoreState()


def _build_pdf(data) -> None:
    styles = _styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.75 * cm,
        bottomMargin=1.75 * cm,
        title="Relativistic ell=1 boson stars: background and radial-mode certification",
        author="Boson Star QNM Project",
        subject="Publication checkpoint v0.3.1",
    )
    story = []
    p = lambda text, style="body": story.append(Paragraph(text, styles[style]))
    h1 = lambda text: story.append(Paragraph(text, styles["h1"]))
    h2 = lambda text: story.append(Paragraph(text, styles["h2"]))
    spacer = lambda amount=0.25: story.append(Spacer(1, amount * cm))

    # Title page.
    spacer(2.1)
    p("PUBLICATION CHECKPOINT / v0.3.1", "subtitle")
    p("Relativistic ell=1 boson stars:<br/>background and radial-mode certification", "title")
    p(
        "A reproducible numerical evidence report for the equilibrium sequence, radius-definition audit, radial stability transition, ground mode, and first overtone.",
        "subtitle",
    )
    spacer(0.45)
    badge = Table(
        [["RADIAL GATE B: PASS", "GATE A: IN PROGRESS", "FULL GATE B: IN PROGRESS"]],
        colWidths=[5.3 * cm, 5.3 * cm, 5.3 * cm],
    )
    badge.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), TEAL),
                ("BACKGROUND", (1, 0), (-1, 0), GOLD),
                ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
                ("FONT", (0, 0), (-1, -1), "Helvetica-Bold", 7.8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.6, NAVY),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, NAVY),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(badge)
    spacer(1.0)
    p(
        "Headline result", "h2"
    )
    p(
        "sigma0^2 = 2.4004311443e-4 at a1^0 = 0.08, with a conservative deterministic-systematics bound of 7.70e-9 in absolute sigma0^2. The domain-converged first overtone is sigma1 = 0.0907036982.",
        "callout",
    )
    spacer(2.0)
    p("Prepared 22 August 2026", "subtitle")
    p("Tagged numerical release: radial-v0.3.1-hardened", "subtitle")
    p("Scope: certified background collocation and relativistic radial ell=1 perturbations. Ordinary ell=0 and new nonradial QNMs are not included.", "subtitle")
    story.append(PageBreak())

    # Abstract and evidence chain.
    h1("Abstract")
    p(
        "We present a publication-style certification of static ell=1 boson-star backgrounds and their relativistic radial pulsations. The equilibrium sequence reproduces the published frequency at a1^0=0.08 and the maximum-mass configuration. A dedicated radius audit finds R99 = 10.18 for the maximum-mass grid model, while the published value 12.75 closely tracks R999 = 12.70; this discrepancy remains unresolved and prevents Gate A from closing. Radial modes are computed with two well-conditioned discretizations: a nonlinear global boundary-value problem and an independently transcribed, equilibrated Chebyshev generalized eigenproblem. The methods agree in eigenvalue and eigenfunction shape, reproduce the stability sign change, and identify the one-node first overtone. Conservative deterministic uncertainty budgets are reported separately for both low modes. The radial portion of Gate B passes; full Gate B awaits an ordinary ell=0 boson-star QNM benchmark.",
        "lead",
    )
    p("Evidence chain", "h2")
    chain = Table(
        [["equations", "boundary data", "two discretizations", "convergence", "physics", "gate"]],
        colWidths=[2.55 * cm] * 6,
    )
    chain.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.8),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                ("BOX", (0, 0), (-1, 0), 0.7, BLUE),
                ("INNERGRID", (0, 0), (-1, 0), 0.5, BLUE),
                ("TOPPADDING", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ]
        )
    )
    story.append(chain)
    h2("Claims and non-claims")
    p(
        "The report certifies the radial benchmark as a numerical result. It does not close the independent-background requirement, resolve the R99/R999 literature discrepancy, validate an ordinary boson-star QNM solver, or present any new nonradial mode. Algebraic pencil residuals are treated as discretization diagnostics rather than continuum error estimates. The spectral condition number is reported because low algebraic residual alone does not establish a well-conditioned physical eigenvalue.",
    )
    h2("Report map")
    p(
        "Sections 1-2 define conventions and background equations. Section 3 gives the equilibrium evidence and radius audit. Sections 4-5 derive the radial problem and document both numerical formulations. Section 6 assembles the benchmark, convergence, eigenfunction, and uncertainty evidence. Sections 7-9 state limitations, gate decisions, and the reproducibility manifest. Six figures and eight interpretive tables are supported by versioned CSV and JSON records.",
    )
    story.append(PageBreak())

    # 1. Scope and conventions.
    h1("1. Scope, conventions, and acceptance policy")
    p(
        "The project uses geometrized units G = c = hbar = mu = 1 and metric signature (-,+,+,+). The background line element is written in areal radius as ds^2 = -alpha(r)^2 dt^2 + gamma(r)^2 dr^2 + r^2 dOmega^2, with gamma^2 = [1 - 2M(r)/r]^-1. Perturbations vary as exp(-i sigma t), so Im(sigma) < 0 denotes damping and Im(sigma) > 0 growth. These choices are frozen in the repository and every comparison is converted into this normalization.",
    )
    table1 = [
        ["Quantity", "Project convention", "Consequence"],
        ["Units", "G = c = hbar = mu = 1", "Frequencies and radii are dimensionless"],
        ["Metric", "(-,+,+,+), areal radius", "gamma^2 = (1-2M/r)^-1"],
        ["Matter", "2 ell+1 equal-mass complex fields", "kappa_ell = 2 ell+1"],
        ["Center amplitude", "psi/r^ell -> a_ell^0/kappa_ell", "Reproduces the published Table II normalization"],
        ["Time dependence", "exp(-i sigma t)", "Im(sigma)<0 is damped"],
        ["Radius", "M(R99)=0.99 M_T", "R999 is stored separately"],
        ["Certification", "domain + resolution + method + residual", "Solver success alone cannot pass a gate"],
    ]
    p("Table 1. Frozen units, signs, normalizations, and acceptance rules.", "table_caption")
    story.append(_table(table1, [3.3 * cm, 6.1 * cm, 7.0 * cm], styles))
    h2("Numerical acceptance")
    p(
        "A headline mode must survive outer-domain variation, resolution variation, background-representation checks, direct boundary or operator residual diagnostics, and an independent discretization. Mode identity is checked by resolved nodes and normalized physical-eigenfunction overlap. Deterministic shifts are summed conservatively for the headline uncertainty; quadrature is retained only as a secondary descriptive diagnostic.",
    )
    p(
        "accepted radial mode = converged BVP + equilibrated spectral pencil + shape agreement + physical boundary data + conservative uncertainty",
        "code",
    )
    story.append(PageBreak())

    # 2. Background formulation.
    h1("2. Relativistic ell=1 backgrounds")
    h2("2.1 Field equations")
    p(
        "For kappa = 2 ell+1, the static Einstein-Klein-Gordon reduction evolves the Misner-Sharp mass M, lapse alpha, radial field psi, and its derivative. In compact notation the implemented equations are",
    )
    p(
        "M' = (kappa r^2/2)[psi'^2/gamma^2 + (omega^2/alpha^2 + 1 + ell(ell+1)/r^2)psi^2]<br/>"
        "alpha'/alpha = gamma^2{M/r^2 + (kappa r/2)[psi'^2/gamma^2 + (omega^2/alpha^2 - 1 - ell(ell+1)/r^2)psi^2]}<br/>"
        "psi'' + (2/r + alpha'/alpha - gamma'/gamma)psi' + gamma^2[omega^2/alpha^2 - 1 - ell(ell+1)/r^2]psi = 0.",
        "code",
    )
    p(
        "Regularity fixes psi = (a_ell^0/kappa) r^ell + O(r^(ell+2)) and M = O(r^(2ell+1)). The outer boundary imposes the Schwarzschild lapse and a massive decaying Robin tail through first order in 1/r. The eigenfrequency omega is constrained to 0 < omega < 1 by an unconstrained logistic parameter.",
    )
    h2("2.2 Collocation and continuation")
    p(
        "The nonlinear background boundary-value problem is solved by adaptive collocation on a geometric radial mesh. Continuation begins at the validated a1^0=0.08 anchor and advances along the nodeless branch. Stored profiles contain M, alpha, gamma, psi, psi', global charges and radii, tail diagnostics, and solver status. Gate A nevertheless remains open because a genuinely independent background shooting formulation has not yet reproduced the profiles.",
    )
    story.append(PageBreak())

    # 3. Background results.
    h1("3. Background sequence and radius audit")
    story.append(_figure(data["figures"][0], "The 21-model equilibrium sequence from a1^0=0.02 to 0.12. The vertical dashed line marks the maximum-mass grid point; translucent bands identify representative profiles used throughout this report. The sequence data are stored in reports/background/ell1_sequence.csv.", 1, styles))
    h2("3.1 Representative models")
    reps = data["representatives"]
    table2 = [["Model", "a1^0", "omega", "M_T", "Q", "R99", "C99"]]
    for name, amplitude in (("dilute", 0.04), ("benchmark", 0.08), ("maximum grid", 0.10)):
        sol = reps[amplitude]
        table2.append([name, f"{amplitude:.3f}", f"{sol.omega:.8f}", f"{sol.adm_mass:.7f}", f"{sol.noether_charge:.7f}", f"{sol.r99:.4f}", f"{sol.compactness99:.5f}"])
    p("Table 2. Representative equilibrium models used for plots and perturbation checks.", "table_caption")
    story.append(_table(table2, [2.5 * cm, 1.6 * cm, 2.5 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm], styles))
    story.append(PageBreak())
    story.append(_figure(data["figures"][1], "Scalar, lapse, radial-metric, and enclosed-mass profiles for dilute, benchmark, and maximum-mass-grid configurations. Increasing central amplitude compacts the matter distribution and deepens the relativistic lapse.", 2, styles))
    h2("3.2 Published benchmarks")
    table3 = [
        ["Benchmark", "Published", "Computed", "Assessment"],
        ["a1^0=0.08 frequency", "omega = 0.8519", f"omega = {reps[0.08].omega:.7f}", "agrees at quoted precision"],
        ["maximum model frequency", "omega about 0.836", f"omega = {reps[0.10].omega:.7f}", "consistent"],
        ["maximum model mass", "M about 1.18", f"M = {reps[0.10].adm_mass:.7f}", "consistent"],
        ["radial ground mode", "sigma0^2 = 2.40e-4", f"{data['ground_bvp'].sigma2:.10e}", "independently certified"],
        ["first radial overtone", "sigma1 = 0.0907", f"{np.sqrt(data['overtone_bvp'].sigma2):.10f}", "agrees beyond quoted precision"],
    ]
    p("Table 3. Published-versus-computed background and radial benchmarks from Alcubierre et al. (2021).", "table_caption")
    story.append(_table(table3, [4.0 * cm, 4.0 * cm, 4.7 * cm, 3.7 * cm], styles))
    story.append(PageBreak())

    story.append(_figure(data["figures"][2], "Enclosed mass fraction for the maximum-mass grid model. The literal 99 percent radius is 10.18, whereas the published 12.75 value lies near the independently computed 99.9 percent radius 12.70. The plot supports an unresolved labeling or numerical discrepancy, not a claim of published error.", 3, styles))
    audit = data["radius_audit"]
    table4 = [
        ["Definition", "Radius", "Mass fraction / note"],
        ["M(R)=0.99 M_T", f"{audit['r99_mass']:.6f}", "literal R99"],
        ["M(R)=0.999 M_T", f"{audit['r999_mass']:.6f}", "literal R999"],
        ["Published value", f"{audit['published_radius']:.2f}", f"computed enclosed fraction = {audit['eta_at_published_radius']:.7f}"],
        ["Proper-energy 99 percent", f"{audit['r99_proper_energy']:.6f}", "cross-check"],
        ["Noether-charge 99 percent", f"{audit['r99_noether_charge']:.6f}", "cross-check"],
    ]
    p("Table 4. Radius-definition audit at a1^0=0.10.", "table_caption")
    story.append(_table(table4, [5.0 * cm, 3.6 * cm, 7.8 * cm], styles))
    h2("3.3 Outer-domain convergence")
    domains = data["domain_backgrounds"]
    table5 = [["r_max", "omega", "M_T", "R99", "tail residual"]]
    for sol in domains:
        table5.append([f"{sol.r[-1]:.0f}", f"{sol.omega:.12f}", f"{sol.adm_mass:.12f}", f"{sol.r99:.8f}", f"{sol.tail_residual:.2e}"])
    p("Table 5. Background outer-domain convergence for a1^0=0.08 at fixed collocation tolerance 1e-7.", "table_caption")
    story.append(_table(table5, [2.2 * cm, 3.6 * cm, 3.6 * cm, 3.3 * cm, 3.4 * cm], styles))
    p(
        "The frequency and mass are stable across the tested outer domains to substantially more digits than required by the published benchmark. This is strong same-formulation convergence evidence, but it is not the missing independent background method.",
    )
    story.append(PageBreak())

    # 4. Radial formulation.
    h1("4. Relativistic radial perturbation problem")
    h2("4.1 Variables and eigenvalue")
    p(
        "The radial Appendix system is implemented in delta-varphi_11 and delta-L = delta-lambda/(2 kappa psi_1^2), with sigma^2 as the eigenvalue. Using sigma^2 permits stable and unstable modes to share a real-valued code path. The physical comparison fields are psi delta-varphi_11 and delta-lambda. At the regular center, the normalization and leading series determine five boundary conditions for four first-order fields plus the unknown eigenvalue.",
    )
    p(
        "y = (delta-varphi, delta-varphi', delta-L, delta-L')<br/>"
        "y' = F(r, y; sigma^2), &nbsp;&nbsp; A x = sigma^2 B x",
        "code",
    )
    h2("4.2 Physical outer conditions")
    p(
        "The global BVP directly imposes the two published finite-radius physical conditions F1=F2=0. For the spectral pencil they are theoretically equivalent to Dirichlet conditions on both numerical perturbation variables because the background scalar and omega are nonzero at the finite outer boundary. Endpoint residuals demonstrate that the imposed conditions are met; domain, resolution, representation, and cross-method changes provide the independent numerical evidence.",
    )
    h2("4.3 Why affine outward shooting was rejected")
    p(
        "The original outward-basis experiment subtracted exponentially amplified solutions. It could reproduce an interior profile while producing misleading normalized boundary residuals through catastrophic cancellation. It remains in the repository only as deprecated historical code. No result or uncertainty in this report uses it for certification.",
    )
    story.append(PageBreak())

    # 5. Numerical methods.
    h1("5. Two radial discretizations")
    h2("5.1 Nonlinear global BVP")
    p(
        "SciPy's adaptive collocation solver treats sigma^2 as a free parameter and enforces the regular-center and physical outer equations simultaneously. An outward IVP supplies only the Newton starting profile. Diagnostics distinguish the maximum SciPy interval RMS relative residual from a separately sampled maximum pointwise relative ODE residual. This terminology is retained in the machine-readable table.",
    )
    h2("5.2 Independent generalized eigenproblem")
    p(
        "The interval [epsilon,r_max] is discretized at Chebyshev-Lobatto points. A separately transcribed coefficient module assembles the linear pencil A x = sigma^2 B x without calling the BVP right-hand side or center coefficient. Tests compare the assembled interior operator and all center/outer rows against direct evaluations on arbitrary smooth fields. The pencil is equilibrated by positive row and column scalings before the dense generalized eigensolve.",
    )
    p(
        "A_s = D_r A D_c, &nbsp;&nbsp; B_s = D_r B D_c, &nbsp;&nbsp; A_s v = sigma^2 B_s v",
        "code",
    )
    h2("5.3 Root filtering and mode tracking")
    p(
        "Finite eigenvalues are retained only when their imaginary contamination lies below the numerical tolerance and their real value falls inside the requested search interval. Physical fields are reconstructed before classification. Resolved interior zeros are located by shape-preserving interpolation and bracketing with only a floating-point noise floor; the former fixed 1e-4 amplitude cutoff is not used. Across resolutions, normalized L2 eigenfunction overlap supplements frequency and nodal continuity.",
    )
    h2("5.4 Background smoothness policy")
    p(
        "The certification solver requires a C1 cubic-Hermite representation of u=psi/r^ell built from stored psi and psi'. PCHIP is rejected by the public spectral API because a small matrix residual can coexist with a biased continuum eigenvalue when coefficient derivatives are insufficiently smooth. PCHIP is retained only as a local-BVP representation variation.",
    )
    story.append(PageBreak())

    # 6. Radial results.
    h1("6. Radial spectrum and certification")
    story.append(_figure(data["figures"][3], "Ground radial eigenvalue across the central-amplitude sequence. Positive sigma0^2 denotes oscillatory stability; the sign becomes negative beyond the maximum-mass neighborhood, reproducing the turning-point stability transition.", 4, styles))
    h2("6.1 Low-mode frequencies")
    ground = data["ground_bvp"]
    overtone = data["overtone_bvp"]
    table6 = [
        ["Mode", "Method/domain", "sigma^2", "sigma", "nodes", "Published"],
        ["ground", "BVP, r_max=40", f"{ground.sigma2:.12e}", f"{np.sqrt(ground.sigma2):.10f}", "0", "sigma0^2=2.40e-4"],
        ["first overtone", "BVP, r_max=60", f"{overtone.sigma2:.12e}", f"{np.sqrt(overtone.sigma2):.10f}", "1", "sigma1=0.0907"],
    ]
    p("Table 6. Published-versus-computed relativistic radial frequencies at a1^0=0.08.", "table_caption")
    story.append(_table(table6, [2.5 * cm, 3.5 * cm, 3.7 * cm, 3.0 * cm, 1.6 * cm, 3.2 * cm], styles))
    story.append(PageBreak())

    story.append(_figure(data["figures"][4], "Four complementary convergence diagnostics: BVP outer-domain convergence, the broad N=50-160 spectral envelope, scaled versus unscaled algebraic residuals, and the stronger outer-domain sensitivity of the first overtone. Oscillatory spectral drift motivates an envelope rather than a monotonic extrapolation.", 5, styles))
    h2("6.2 Cross-method comparison")
    ground_sp = data["ground_sp"]
    # Locate the r=60 N=160 overtone row for the common-domain table.
    r60_overtone_row = next(
        row
        for row in data["radial_rows"]
        if row["formulation"].startswith("chebyshev")
        and row["r_max"] == "60.0"
        and row["resolution"].startswith("160 ")
        and row["mode_index"] == "1"
    )
    table7 = [
        ["Mode/domain", "BVP sigma^2", "Spectral sigma^2", "absolute difference", "condition number"],
        ["ground, r_max=40, N=80", f"{ground.sigma2:.12e}", f"{ground_sp.sigma2:.12e}", f"{abs(ground.sigma2-ground_sp.sigma2):.3e}", f"{ground_sp.eigenvalue_condition_number:.3e}"],
        ["overtone, r_max=60, N=160", f"{overtone.sigma2:.12e}", f"{float(r60_overtone_row['sigma2']):.12e}", f"{abs(overtone.sigma2-float(r60_overtone_row['sigma2'])):.3e}", f"{float(r60_overtone_row['eigenvalue_condition_number']):.3e}"],
    ]
    p("Table 7. BVP-spectral comparison at common finite domains. The overtone condition number quantifies the harder large-domain pencil.", "table_caption")
    story.append(_table(table7, [4.0 * cm, 3.4 * cm, 3.4 * cm, 3.0 * cm, 2.8 * cm], styles))
    story.append(PageBreak())

    story.append(_figure(data["figures"][5], "Physical scalar and metric eigenfunctions for the ground mode and first overtone at a common r_max=40 domain. Curves are independently normalized. Their near-coincidence verifies mode shape, while the overtone panels visibly contain one resolved interior node.", 6, styles))
    h2("6.3 Eigenfunction agreement")
    scalar_overlap = eigenfunction_overlap(ground.r, ground.physical_scalar, ground_sp.r, ground_sp.physical_scalar)
    metric_overlap = eigenfunction_overlap(ground.r, ground.delta_lambda, ground_sp.r, ground_sp.delta_lambda)
    p(
        f"At the common domain, the normalized L2 overlaps are {scalar_overlap:.9f} for the physical scalar perturbation and {metric_overlap:.9f} for delta-lambda. This shape comparison is stronger evidence than eigenvalue agreement alone because it probes the reconstructed physical fields across the radial interval.",
    )
    story.append(PageBreak())

    # 7. Uncertainty.
    h1("7. Deterministic numerical uncertainty")
    gu = data["ground_uncertainty"]
    ou = data["overtone_uncertainty"]
    table8 = [["Mode/component", "absolute shift in sigma^2", "Interpretation"]]
    for key, value in gu["components_absolute_sigma2"].items():
        table8.append(["ground: " + key.replace("_", " "), f"{value:.3e}", "systematic variation"])
    table8.append(["GROUND CONSERVATIVE SUM", f"{gu['conservative_sum_absolute_sigma2']:.3e}", "headline bound"])
    for key, value in ou["components_absolute_sigma2"].items():
        table8.append(["overtone: " + key.replace("_", " "), f"{value:.3e}", "systematic variation"])
    table8.append(["OVERTONE CONSERVATIVE SUM", f"{ou['conservative_sum_absolute_sigma2']:.3e}", "headline bound"])
    p("Table 8. Conservative numerical-systematics budgets. Components are not assumed statistically independent.", "table_caption")
    story.append(_table(table8, [8.2 * cm, 3.7 * cm, 4.5 * cm], styles))
    h2("Recommended reporting")
    p(
        "For the ground mode, a defensible rounded result is sigma0^2 = (2.40043 +/- 0.00008)e-4, where the uncertainty is a deterministic numerical-systematics bound. For the first overtone, sigma1 = 0.09070370 is the domain-converged reference, but the substantially larger sigma1^2 bound should accompany any precision claim. Neither uncertainty is a statistical standard deviation.",
        "callout",
    )
    p(
        "The ground budget includes BVP domain, BVP-spectral method, broad spectral-resolution, and Hermite-versus-PCHIP local-BVP representation changes. The overtone budget separately includes its r_max=40 to 60 domain shift, the r_max=60 BVP-spectral comparison, and the maximum displacement of the r_max=40 spectral resolution sequence from the common-domain BVP.",
    )
    story.append(PageBreak())

    # 8. Limitations and gates.
    h1("8. Limitations, gate decision, and next milestone")
    h2("8.1 What is certified")
    p(
        "The relativistic radial ell=1 benchmark passes. Two independent discretizations reproduce the ground eigenvalue, stability sign crossing, first overtone, and physical eigenfunction shapes. Direct matrix-operator tests independently validate the spectral coefficient transcription. The embedded release history and runtime provenance make the result reproducible.",
    )
    h2("8.2 What remains open")
    p(
        "Gate A remains in progress because the background collocation profiles still lack an independent shooting reproduction and the radius discrepancy is unresolved. Full Gate B remains in progress because the ordinary ell=0 boson-star QNM benchmark has not been implemented. A smooth-background mapped multidomain architecture, explicit branch handling for massive scalar sidebands, constraint and gauge diagnostics, and a time-domain or other independent physical validation are required before new nonradial spectra can be frozen for publication.",
    )
    gates = [
        ["Gate", "Decision", "Evidence / blocker"],
        ["A: background", "IN PROGRESS", "Collocation converged; independent shooting and R99 discrepancy remain"],
        ["B: radial portion", "PASS", "BVP + independent equilibrated spectral formulation + shape agreement"],
        ["B: full", "IN PROGRESS", "Ordinary ell=0 QNM benchmark remains"],
        ["Production nonradial", "NOT CERTIFIED", "Multidomain QNM system, constraints, branches, and independent validation remain"],
    ]
    story.append(_table(gates, [3.3 * cm, 3.0 * cm, 10.1 * cm], styles))
    h2("8.3 Next hard milestone")
    p(
        "The next miniature publication report should treat ordinary ell=0 boson-star QNMs with the same evidence chain: asymptotics and branch conventions, a frequency-domain solver, three-resolution or contour-based root filtering, published scalar-led and gravitational-led benchmarks, eigenfunctions, a time-domain or second-method cross-check, and an explicit uncertainty/provenance table. No new ell=1 nonradial frequency should be promoted to a paper claim before that benchmark passes.",
    )
    story.append(PageBreak())

    # 9. Reproducibility.
    h1("9. Reproducibility manifest")
    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    pin_sha = hashlib.sha256((ROOT / "environment" / "requirements-pins.txt").read_bytes()).hexdigest()
    manifest = [
        ["Item", "Value"],
        ["Report source commit", commit],
        ["Report source tree", tree],
        ["Certified numerical implementation", gu["provenance"]["implementation_commit"]],
        ["Release tag", "radial-v0.3.1-hardened"],
        ["Python", platform.python_version()],
        ["NumPy / SciPy", f"{np.__version__} / {gu['provenance']['scipy_version']}"],
        ["Platform", platform.platform()],
        ["Top-level pin SHA-256", pin_sha],
        ["Background sequence", "reports/background/ell1_sequence.csv"],
        ["Radial certification", "reports/radial/radial_benchmarks.csv (27 rows)"],
        ["Ground uncertainty", "reports/radial/radial_uncertainty.json"],
        ["Overtone uncertainty", "reports/radial/radial_overtone_uncertainty.json"],
        ["Report stability curve", "reports/radial/report_radial_stability.csv (13 regenerated models)"],
        ["Automated suite", "20 tests"],
    ]
    story.append(_table(manifest, [5.0 * cm, 11.4 * cm], styles))
    h2("Regeneration commands")
    p(
        "python -m radial.diagnostics --output reports/radial/radial_benchmarks.csv --uncertainty-output reports/radial/radial_uncertainty.json --overtone-uncertainty-output reports/radial/radial_overtone_uncertainty.json<br/>"
        "python paper/build_certification_report.py<br/>"
        "python -m pytest -q",
        "code",
    )
    h2("Data and code statement")
    p(
        "All plotted radial certification values and uncertainty components are stored in machine-readable repository files. Figure source PDFs and high-resolution raster previews are generated in the temporary report asset directory; the final report is emitted under output/pdf. The tagged Git bundle in the presentation release records the complete numerical history.",
    )
    story.append(PageBreak())

    # References and appendix.
    h1("References")
    refs = [
        "[1] M. Alcubierre et al., On the linear stability of ell-boson stars with respect to radial perturbations, Phys. Rev. D 103, 124046 (2021), arXiv:2103.15012.",
        "[2] M. Alcubierre et al., L-boson stars, Class. Quantum Grav. 35, 19LT01 (2018), arXiv:1805.11488.",
        "[3] SciPy documentation for solve_bvp and scipy.linalg.eig, runtime version recorded in the manifest.",
    ]
    for ref in refs:
        p(ref)
    h1("Appendix A. Diagnostic definitions")
    p(
        "SciPy interval RMS relative residual: the maximum entry of solve_bvp.rms_residuals; this is not a maximum pointwise defect. Dense pointwise relative residual: max |y'_collocation - F(r,y)|/(1+|F|) on a 3000-point geometric audit grid. Scaled generalized residual: ||A_s v - sigma^2 B_s v||/[||A_s v|| + |sigma^2| ||B_s v||]. Unscaled generalized residual: the same expression after transforming the right eigenvector back to the original pencil. Left/right condition number: ||w|| ||v||/|w^H B_s v| for the equilibrated pencil. These algebraic quantities diagnose the discrete solve; continuum uncertainty comes from domain, resolution, representation, and method changes.",
    )
    h1("Appendix B. Release checklist")
    checklist = [
        ["Check", "Status"],
        ["Portable tagged source archive", "complete in radial-v0.3.1-hardened release"],
        ["Full Git history bundle", "verified"],
        ["Runtime and pin provenance", "recorded in CSV/JSON and this report"],
        ["BVP and spectral low modes", "reproduced"],
        ["Independent spectral coefficients", "direct operator identity tests pass"],
        ["Conservative ground uncertainty", "7.70e-9 absolute sigma0^2"],
        ["Separate overtone uncertainty", "6.44e-7 absolute sigma1^2"],
        ["Ordinary ell=0 QNM", "not implemented"],
        ["New nonradial spectrum", "not implemented"],
    ]
    story.append(_table(checklist, [8.2 * cm, 8.2 * cm], styles))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    _configure_plotting()
    data = _build_numerical_assets()
    _build_pdf(data)
    print(OUTPUT)


if __name__ == "__main__":
    main()
