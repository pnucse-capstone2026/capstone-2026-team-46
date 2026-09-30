#!/usr/bin/env python3
"""Three-panel E16 relational-versus-marginal decomposition figure.

Registered by ``journal/experiments/e16_relational_response_decomposition/PREREG.md``
sections 12 and 14.2, as amended 2026-07-25 v1.

A summary band of three key-figure cards sits above the panels; every number it
shows is read from the same published tables the panels use.

Panels
------
A  Decomposition per generator realization: the total ``Rule - Real``
   payload-destination response split into the marginal-exposure component
   ``Placebo - Real`` and the relational increment ``theta = Rule - Placebo``.
B  ``theta_g`` across the twenty prospective generator realizations against the
   historical SESOI band ``[-delta, +delta]`` (not the article claim), with
   the aggregate 90% interval.
C  The mandatory ``P = 1`` secondary in both registered forms: absolute recall of
   all three arms, and the Rule-to-placebo gain ratio summarised by median and
   interquartile range.

The figure is built only from published E16 analysis outputs; it recomputes no
outcome.  Colours are Okabe-Ito with marker and hatch redundancy so the figure
survives grayscale reproduction.  Publication is byte-identity checked: the
figure is rendered twice and published only if both passes agree exactly.
Re-rendering an already published version requires ``--republish``, which
archives the superseded outputs under ``journal/results/archive/`` first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_e16_relational_response as e16an  # noqa: E402
import generate_e16_rule_placebo_pairs as e16gen  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _environment_record,
    _stage_bytes,
    _stage_json,
    atomic_publish_bundle,
)

REPO = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO / "journal" / "results" / "figures"
LOG_DIR = REPO / "journal" / "results" / "logs"
TABLE_DIR = REPO / "journal" / "results" / "tables"
ARCHIVE_DIR = REPO / "journal" / "results" / "archive"

OUTPUT_VERSION = e16gen.OUTPUT_VERSION
FIGURE_SCHEMA = "e16.figure_provenance.v2"

# Okabe-Ito, with marker/hatch redundancy for grayscale reproduction.
BLUE = "#0072B2"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GRAY = "#777777"
LIGHT_GRAY = "#D5D5D5"
INK = "#222222"
GRID = "#D9D9D9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# Keys are the machine verdict codes emitted in
# ``e16_relational_response_v2.log``; the values are the reader-visible panel
# text, which must stand on its own without an undefined branch code.
VERDICT_TEXT = {
    "A": "historical SESOI (not the article claim)",
    "B": "relational increment is material",
    "C": "realization-contingent",
    "D": "inconclusive",
}


class E16FigureError(RuntimeError):
    """Raised when the figure cannot be built from published outputs."""


def _stop(message: str) -> E16FigureError:
    return E16FigureError(f"T-STOP-E16-FIGURE: {message}")



def publish_and_cleanup(staged: Sequence[tuple[Path, Path]]) -> None:
    """Publish atomically, then remove the staged hard-link sources.

    ``atomic_publish_bundle`` links staged files into place and deliberately
    leaves them behind so a failed bundle can be diagnosed.  On success the
    caller owns that cleanup; skipping it leaves dot-prefixed temporaries that
    a later stage's clean-source gate will reject.
    """
    atomic_publish_bundle(staged)
    for source, _ in staged:
        try:
            Path(source).unlink()
        except FileNotFoundError:
            pass

def figure_paths() -> dict[str, Path]:
    stem = f"e16_relational_response_decomposition_{OUTPUT_VERSION}"
    return {
        "pdf": FIGURE_DIR / f"{stem}.pdf",
        "png": FIGURE_DIR / f"{stem}.png",
        "provenance": LOG_DIR
        / f"e16_relational_response_figure_provenance_{OUTPUT_VERSION}.json",
    }


def _require(path: Path) -> Path:
    if not path.exists():
        raise _stop(f"missing published analysis output: {path}")
    return path


def load_inputs() -> dict[str, Any]:
    analysis = e16an.output_paths()
    decomposition = pd.read_csv(_require(analysis["decomposition"]))
    p1_cell = pd.read_csv(_require(analysis["p1_cell"]))
    verdict = json.loads(_require(analysis["verdict"]).read_text())
    primary = verdict["primary"]
    decomposition = decomposition[decomposition["role"] == "primary"].copy()
    if len(decomposition) != len(e16gen.CONSTRUCTION_SEEDS):
        raise _stop(
            f"decomposition has {len(decomposition)} primary generator "
            f"realizations, "
            f"registered {len(e16gen.CONSTRUCTION_SEEDS)}")
    return {"decomposition": decomposition, "p1_cell": p1_cell,
            "verdict": verdict, "primary": primary}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _panel_a(ax, decomposition: pd.DataFrame) -> None:
    frame = decomposition.sort_values("delta_rule_minus_real").reset_index(
        drop=True)
    y = np.arange(len(frame))
    ax.barh(y, frame["delta_placebo_minus_real"], height=0.62,
            color=SKY, edgecolor=INK, linewidth=0.5,
            label="marginal exposure (Placebo $-$ Real)")
    ax.barh(y, frame["theta_rule_minus_placebo"], height=0.62,
            left=frame["delta_placebo_minus_real"], color=ORANGE,
            edgecolor=INK, linewidth=0.5, hatch="///",
            label=r"relational increment ($\theta$)")
    ax.plot(frame["delta_rule_minus_real"], y, "D", color=INK,
            markersize=3.8, markeredgewidth=0.6, markeredgecolor="white",
            label="total (Rule $-$ Real)", linestyle="none")
    ax.axvline(0.0, color=GRAY, linewidth=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([str(int(s)) for s in frame["construction_seed"]],
                       fontsize=5.6)
    # reserve three empty rows below the bars so the legend never covers data
    ax.set_ylim(-3.6, len(frame) - 0.2)
    span = min(float(frame["delta_rule_minus_real"].min()),
               float(frame["delta_placebo_minus_real"].min()))
    ax.set_xlim(span * 1.06, abs(span) * 0.10)
    ax.set_xlabel("payload-destination effect $C_P$ (exact macro recall)",
                  fontsize=7.2)
    ax.set_ylabel("generator realization", fontsize=7.2)
    ax.set_title("(a)  Payload-destination collapse per generator realization",
                 fontsize=8.0, loc="left", color=INK)
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, loc="lower left", frameon=False, ncol=1)


def _panel_b(ax, decomposition: pd.DataFrame, primary: dict[str, Any]) -> None:
    theta = decomposition["theta_rule_minus_placebo"].to_numpy(dtype=float)
    delta = float(primary["delta"])
    order = np.argsort(theta)
    y = np.arange(len(theta))

    ax.axvspan(-delta, delta, color=LIGHT_GRAY, alpha=0.55, zorder=0,
               label=r"historical SESOI $[-\delta,+\delta]$ (not the claim)")
    ax.axvline(0.0, color=GRAY, linewidth=0.8, zorder=1)
    inside = np.abs(theta[order]) <= delta
    ax.plot(theta[order][inside], y[inside], "o", color=BLUE, markersize=3.6,
            linestyle="none", label=r"$\theta_g$ inside historical SESOI")
    if (~inside).any():
        ax.plot(theta[order][~inside], y[~inside], "s", color=VERMILLION,
                markersize=3.8, linestyle="none",
                label=r"$\theta_g$ outside historical SESOI")

    mean = float(primary["theta_mean"])
    ax.axvline(mean, color=GREEN, linewidth=1.2, linestyle="--", zorder=3,
               label=r"aggregate $\bar{\theta}$")
    ax.annotate(
        "", xy=(primary["ci90_low"], -1.4), xytext=(primary["ci90_high"], -1.4),
        arrowprops={"arrowstyle": "|-|,widthA=0.35,widthB=0.35",
                    "color": GREEN, "linewidth": 1.1})
    ax.text(mean, -2.6, "90% interval (historical TOST)", ha="center",
            va="top", fontsize=6.0, color=GREEN)

    ax.set_title("(b)  Realization-level $\\theta_g$ (historical SESOI shown)",
                 fontsize=8.0, loc="left", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [str(int(s)) for s in
         decomposition["construction_seed"].to_numpy()[order]], fontsize=5.6)
    # Reserve empty rows below for the TOST annotation and above for the
    # legend, so neither ever sits on a plotted point.
    ax.set_ylim(-3.4, len(theta) + 2.6)
    ax.set_xlabel(r"relational increment "
                  r"$\theta_g = C_P(\mathrm{Rule}) - C_P(\mathrm{Placebo})$",
                  fontsize=7.2)
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    # The margin legend is wide enough to reach the zero and aggregate rules,
    # so it needs an opaque backing to stay legible.
    ax.legend(fontsize=6.0, loc="upper left", frameon=True, facecolor="white",
              framealpha=0.88, edgecolor="none", borderpad=0.35).set_zorder(6)


def _panel_c(ax, p1_cell: pd.DataFrame, verdict: dict[str, Any]) -> None:
    summary = verdict["p1_cell_summary"]
    scopes = ["equal_macro", "Gear", "RPM"]
    labels = ["equal macro", "Gear", "RPM"]
    width = 0.26
    x = np.arange(len(scopes))

    real = [float(p1_cell[p1_cell["attack_scope"] == s]["real_recall"].mean())
            for s in scopes]
    rule = [float(summary[s]["rule_recall_mean"]) for s in scopes]
    placebo = [float(summary[s]["placebo_recall_mean"]) for s in scopes]

    ax.bar(x - width, real, width, color=GRAY, edgecolor=INK, linewidth=0.5,
           label="real only")
    ax.bar(x, placebo, width, color=SKY, edgecolor=INK, linewidth=0.5,
           label="placebo")
    ax.bar(x + width, rule, width, color=ORANGE, edgecolor=INK, linewidth=0.5,
           hatch="///", label="Rule")

    for index, scope in enumerate(scopes):
        entry = summary[scope]
        median = entry.get("ratio_median")
        if median is None:
            text = "ratio unstable"
        else:
            text = (f"{median:.2f}x\n[{entry['ratio_iqr_low']:.2f}, "
                    f"{entry['ratio_iqr_high']:.2f}]")
        top = max(real[index], rule[index], placebo[index])
        ax.text(x[index] + width * 0.5, top + 0.012, text, ha="center",
                va="bottom", fontsize=5.8, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("exact attack-type recall at $P=1$", fontsize=7.2)
    ax.set_title("(c)  Out-of-grammar cell: absolute recall and gain ratio",
                 fontsize=8.0, loc="left", color=INK)
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, loc="upper right", frameon=False)
    ax.margins(y=0.22)


# Card geometry fits KPI_VALUE_FIT_CHARS rendered glyphs at the full value
# size; longer values shrink by the reciprocal of their length so a future
# verdict string cannot silently overflow the card.  The floor keeps the value
# well above the 6.6pt card label, so the two never read as one weight.
KPI_LABEL_FONTSIZE = 6.6
KPI_VALUE_FONTSIZE = 13.5
KPI_VALUE_FIT_CHARS = 15
KPI_VALUE_FONTSIZE_MIN = 8.4

_MATHTEXT_COMMAND = re.compile(r"\\[A-Za-z]+")


def _rendered_length(text: str) -> int:
    """Glyph count of a mathtext label: one glyph per command, no delimiters."""
    return len(_MATHTEXT_COMMAND.sub("x", text).translate(
        str.maketrans("", "", "${}")))


def _kpi_value_fontsize(value: str) -> float:
    length = max(_rendered_length(value), KPI_VALUE_FIT_CHARS)
    return max(KPI_VALUE_FONTSIZE * KPI_VALUE_FIT_CHARS / length,
               KPI_VALUE_FONTSIZE_MIN)


def _kpi_card(ax, x: float, w: float, accent: str, tint: str, title: str,
              value: str, caption: str) -> None:
    pad = 0.012
    box = FancyBboxPatch(
        (x + pad, 0.06), w - 2 * pad, 0.88,
        boxstyle="round,pad=0,rounding_size=0.05",
        transform=ax.transAxes, clip_on=False,
        facecolor=tint, edgecolor=accent, linewidth=1.0, zorder=1)
    ax.add_patch(box)
    ax.text(x + pad + 0.028, 0.72, title, transform=ax.transAxes,
            fontsize=KPI_LABEL_FONTSIZE, fontweight="bold", color=accent,
            va="top")
    ax.text(x + pad + 0.028, 0.5, value, transform=ax.transAxes,
            fontsize=_kpi_value_fontsize(value), fontweight="bold", color=INK,
            va="top")
    ax.text(x + pad + 0.028, 0.2, caption, transform=ax.transAxes,
            fontsize=6.2, color=INK, va="top")


def _kpi_band(ax, decomposition: pd.DataFrame, primary: dict[str, Any],
              p1_cell: pd.DataFrame) -> None:
    # Placebo recovery: what share of the registered Rule-minus-real
    # payload-destination collapse the marginal-matched placebo reproduces.
    primary_rows = decomposition[decomposition["role"] == "primary"]
    mean_rule = float(primary_rows["delta_rule_minus_real"].mean())
    mean_placebo = float(primary_rows["delta_placebo_minus_real"].mean())
    recovery_pct = 100.0 * mean_placebo / mean_rule

    theta_mean = float(primary["theta_mean"])
    ci_lo, ci_hi = float(primary["ci90_low"]), float(primary["ci90_high"])
    g = float(p1_cell.loc[
        p1_cell["attack_scope"] == "equal_macro", "placebo_gain"].mean())
    share_lo = 100.0 * ci_lo / g
    share_hi = 100.0 * ci_hi / g

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    gap = 0.028
    w = (1 - 2 * gap) / 3
    _kpi_card(
        ax, 0, w, ORANGE, "#FCEFDA",
        "PLACEBO RECOVERY", f"{recovery_pct:.1f}%",
        "of the payload-destination collapse $C_P$")
    _kpi_card(
        ax, w + gap, w, BLUE, "#E3F0FA",
        "RELATIONAL INCREMENT", rf"$\theta$ = {theta_mean:+.4f}",
        "Rule $-$ placebo, registered contrast")
    _kpi_card(
        ax, 2 * (w + gap), w, GREEN, "#E1F3EC",
        "SHARE OF CELL GAIN $G$",
        rf"$[{share_lo:.0f}\%, {share_hi:.0f}\%]$",
        rf"90% interval vs $G={g:.4f}$ (point estimate)")


def build_figure(inputs: dict[str, Any]) -> Any:
    fig = plt.figure(figsize=(7.45, 8.5))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[0.34, 2.05, 1.0], hspace=0.30, wspace=0.24,
        left=0.085, right=0.985, top=0.965, bottom=0.068)
    _kpi_band(fig.add_subplot(grid[0, :]), inputs["decomposition"],
              inputs["primary"], inputs["p1_cell"])
    _panel_a(fig.add_subplot(grid[1, 0]), inputs["decomposition"])
    _panel_b(fig.add_subplot(grid[1, 1]), inputs["decomposition"],
             inputs["primary"])
    _panel_c(fig.add_subplot(grid[2, :]), inputs["p1_cell"], inputs["verdict"])
    return fig


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO)),
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
    }


def _git(arguments: Sequence[str]) -> str:
    try:
        return subprocess.run(["git", *arguments], cwd=REPO, check=True,
                              capture_output=True, text=True,
                              timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise _stop(f"git failed: git {' '.join(arguments)}") from exc


def _source_record_for_render(input_paths: dict[str, Path]) -> dict[str, Any]:
    """Honest source provenance for a re-render of frozen published outcomes.

    The E15 pool-grade gate (``_source_record``) demands a completely clean
    worktree because it guards stages that *produce* outcomes.  This figure
    recomputes nothing: it renders published E16 tables, so it is legitimately
    re-rendered mid-revision, when the manuscript tree is dirty by definition.
    Rather than bypass or fake that gate, this records what is actually true:
    HEAD, the exact renderer and input bytes the figure was built from, and the
    verbatim dirty-path list.  ``worktree_clean`` is never asserted, it is
    observed.
    """
    head = _git(("rev-parse", "HEAD")).strip()
    raw = _git(("status", "--porcelain=v1", "-z", "--untracked-files=all"))
    records = [item for item in raw.split("\0") if item]
    clean = not records
    return {
        "head_commit": head,
        "worktree_clean": clean,
        "source_status": {
            "mode": "clean_worktree_render" if clean
                    else "dirty_worktree_render",
            "stage": "T-STOP-E16-FIGURE",
            "record_count": len(records),
            "status_records": records,
        },
        "pool_grade_source_gate": {
            "applied": False,
            "reason": "figure recomputes no outcome; it renders published E16 "
                      "tables, so the clean-tree pool gate does not apply and "
                      "is not asserted",
        },
        "renderer": _file_record(Path(__file__).resolve()),
        "input_artifacts": {key: _file_record(path)
                            for key, path in sorted(input_paths.items())},
    }


def _archive_superseded(present: Sequence[Path], stamp: str) -> dict[str, Any]:
    """Move superseded outputs into a dated archive; never delete them."""
    destination = ARCHIVE_DIR / stamp
    destination.mkdir(parents=True, exist_ok=True)
    archived: dict[str, Any] = {}
    for path in present:
        target = destination / path.name
        if target.exists():
            raise _stop(f"archive slot already occupied: {target}")
        record = _file_record(path)
        path.rename(target)
        record["archived_to"] = str(target.relative_to(REPO))
        archived[path.name] = record
    return archived


def execute(*, republish: bool = False,
            allow_foreign_environment: bool = False) -> dict[str, Any]:
    inputs = load_inputs()
    paths = figure_paths()
    present = [path for path in paths.values() if path.exists()]
    if present and not republish:
        raise _stop(
            "refusing to overwrite existing figure outputs: "
            f"{[str(p.relative_to(REPO)) for p in present]}; pass --republish "
            "to archive them under journal/results/archive/ and re-render at "
            "the same version")

    input_paths = {key: path for key, path in e16an.output_paths().items()
                   if key in ("decomposition", "p1_cell", "verdict")}
    source = _source_record_for_render(input_paths)
    if allow_foreign_environment:
        if not republish:
            raise _stop(
                "--allow-foreign-environment requires --republish; it is a "
                "display-relabel path, not a substitute for the original "
                "analysis environment")
        environment = {
            "gate": "skipped_display_relabel",
            "reason": "figure labels only; frozen E16 tables unchanged",
            "python": sys.version.split()[0],
            "matplotlib": matplotlib.__version__,
        }
    else:
        environment = _environment_record(REPO)
    registration = e16gen.assert_registration_committed(REPO)

    pdf_bytes, png_bytes, identity = _render_twice(inputs)

    stamp = e16gen.utc_now()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    archived = _archive_superseded(
        present, stamp.replace("-", "").replace(":", "")) if present else {}

    primary = inputs["primary"]
    provenance = {
        "schema_version": FIGURE_SCHEMA,
        "created_utc": stamp,
        "registration": registration,
        "environment": environment,
        "source": source,
        "matplotlib": matplotlib.__version__,
        "backend": matplotlib.get_backend(),
        "inputs": {key: str(path.relative_to(REPO))
                   for key, path in input_paths.items()},
        "verdict": primary["verdict"],
        "verdict_text": VERDICT_TEXT[str(primary["verdict"])],
        "delta": primary["delta"],
        "theta_mean": primary["theta_mean"],
        "outputs": {kind: str(paths[kind].relative_to(REPO))
                    for kind in ("pdf", "png")},
        "outputs_sha256": {"pdf": _sha256_bytes(pdf_bytes),
                           "png": _sha256_bytes(png_bytes)},
        "render_byte_identity": identity,
        "superseded": archived,
        "recomputes_outcomes": False,
    }
    staged = [
        (_stage_bytes(paths["pdf"], pdf_bytes), paths["pdf"]),
        (_stage_bytes(paths["png"], png_bytes), paths["png"]),
        (_stage_json(paths["provenance"], provenance), paths["provenance"]),
    ]
    publish_and_cleanup(staged)
    return provenance


def _render_twice(inputs: dict[str, Any]) -> tuple[bytes, bytes, dict[str, Any]]:
    """Render the figure twice and require byte identity before publishing.

    This binds the published bytes to the frozen machine verdict: if the render
    were sensitive to anything outside the published inputs, the two passes
    would diverge and the figure would not be published at all.
    """
    passes = []
    for _ in range(2):
        fig = build_figure(inputs)
        try:
            passes.append((_render(fig, "pdf"), _render(fig, "png")))
        finally:
            plt.close(fig)
    first, second = passes
    mismatched = [kind for kind, index in (("pdf", 0), ("png", 1))
                  if first[index] != second[index]]
    if mismatched:
        raise _stop(
            f"repeated render is not byte-identical for: {mismatched}")
    return first[0], first[1], {
        "passes": 2,
        "identical": True,
        "pdf_sha256": _sha256_bytes(first[0]),
        "png_sha256": _sha256_bytes(first[1]),
    }


def _render(fig: Any, kind: str) -> bytes:
    import io

    buffer = io.BytesIO()
    if kind == "pdf":
        # Drop the wall-clock CreationDate so repeated renders of the same
        # frozen inputs are byte-identical.
        fig.savefig(buffer, format="pdf", bbox_inches="tight",
                    metadata={"CreationDate": None})
    else:
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    return buffer.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="render and publish the figure")
    parser.add_argument(
        "--republish", action="store_true",
        help="archive the superseded pdf/png/provenance under "
             "journal/results/archive/<UTC stamp>/ and re-render at the same "
             "version; nothing is deleted and OUTPUT_VERSION is unchanged")
    parser.add_argument(
        "--allow-foreign-environment", action="store_true",
        help="with --republish, skip the original analysis-machine "
             "environment hash gate for a display-only relabel")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({
            "stage": "e16_figure",
            "mode": "read-only",
            "inputs_available": {
                key: e16an.output_paths()[key].exists()
                for key in ("decomposition", "p1_cell", "verdict")},
            "outputs": {k: str(v.relative_to(REPO))
                        for k, v in figure_paths().items()},
        }, indent=2, sort_keys=True))
        return 0
    print(json.dumps(execute(
        republish=args.republish,
        allow_foreign_environment=args.allow_foreign_environment), indent=2,
                     sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
