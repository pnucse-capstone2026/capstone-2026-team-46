#!/usr/bin/env python3
"""Four-panel E17 Random-Forest relational-replication figure.

Registered by ``journal/experiments/e17_rf_relational_replication/PREREG.md``
sections 12 and 14.1.

Panels
------
A  Decomposition per generator realization: the total ``Rule - Real``
   payload-destination response split into the marginal-exposure component
   ``Placebo - Real`` and the relational increment ``theta = Rule - Placebo``,
   for the Random-Forest family.
B  ``theta_g`` across the twenty prospective generator realizations against
   **both** registered margins -- the inherited absolute band ``[-delta, +delta]``
   and the stricter ``delta_strict`` rescaled to the Random Forest's own
   out-of-grammar placebo gain -- with the aggregate 90% TOST interval and both
   machine verdicts.  The dual-margin display is mandatory (PREREG 12.2): the
   figure may not show one margin without the other.
C  The mandatory cell secondaries: absolute exact-macro recall of all three
   arms at ``P = 1`` (out of grammar) and ``P = 0`` (in grammar), with the
   registered saturation determination printed.
D  The descriptive CNN/Random-Forest family comparison: the twenty paired
   ``(theta^CNN_g, theta^RF_g)`` values.  Descriptive only -- no test, no
   verdict, no invariance claim (PREREG 12.7).

The figure is built only from published E17 analysis outputs and the frozen E16
decomposition; it recomputes no outcome.  Colours are Okabe-Ito with marker and
hatch redundancy so the figure survives grayscale reproduction.  Publication is
byte-identity checked: the figure is rendered twice and published only if both
passes agree exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_e17_rf_relational_response as e17an  # noqa: E402
import run_e17_rf_paired_training as e17train  # noqa: E402
from generate_rule_construction_sensitivity import (  # noqa: E402
    _environment_record,
    _stage_bytes,
    _stage_json,
)
from make_e16_relational_response_figure import (  # noqa: E402
    BLUE,
    GRAY,
    GRID,
    INK,
    ORANGE,
    SKY,
    VERMILLION,
    _file_record,
    _render,
    _sha256_bytes,
    _source_record_for_render,
)

REPO = e17train.REPO
FIGURE_DIR = REPO / "journal" / "results" / "figures"
LOG_DIR = e17train.LOG_DIR
ARCHIVE_DIR = REPO / "journal" / "results" / "archive"

OUTPUT_VERSION = e17train.OUTPUT_VERSION
FIGURE_SCHEMA = "e17.figure_provenance.v1"

publish_and_cleanup = e17train.publish_and_cleanup

VERDICT_TEXT = {
    "A": "marginal exposure reproduces the response",
    "B": "relational increment is material",
    "C": "realization-contingent",
    "D": "inconclusive",
    "undefined": "not computable",
}


class E17FigureError(RuntimeError):
    """Raised when the figure cannot be built from published outputs."""


def _stop(message: str) -> E17FigureError:
    return E17FigureError(f"T-STOP-E17-FIGURE: {message}")


def figure_paths() -> dict[str, Path]:
    stem = f"e17_rf_relational_replication_{OUTPUT_VERSION}"
    return {
        "pdf": FIGURE_DIR / f"{stem}.pdf",
        "png": FIGURE_DIR / f"{stem}.png",
        "provenance": LOG_DIR
        / f"e17_rf_relational_replication_figure_provenance_"
          f"{OUTPUT_VERSION}.json",
    }


def _require(path: Path) -> Path:
    if not path.exists():
        raise _stop(f"missing published analysis output: {path}")
    return path


def load_inputs() -> dict[str, Any]:
    analysis = e17an.output_paths()
    decomposition = pd.read_csv(_require(analysis["decomposition"]))
    decomposition = decomposition[decomposition["role"] == "primary"].copy()
    if len(decomposition) != len(e17train.CONSTRUCTION_SEEDS):
        raise _stop(
            f"decomposition has {len(decomposition)} primary generator "
            f"realizations, registered "
            f"{len(e17train.CONSTRUCTION_SEEDS)}")
    return {
        "decomposition": decomposition,
        "p1_cell": pd.read_csv(_require(analysis["p1_cell"])),
        "p0_cell": pd.read_csv(_require(analysis["p0_cell"])),
        "family": pd.read_csv(_require(analysis["family"])),
        "verdict": json.loads(_require(analysis["verdict"]).read_text()),
    }


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _panel_a(ax, decomposition: pd.DataFrame) -> None:
    frame = decomposition.sort_values("delta_rule_minus_real").reset_index(
        drop=True)
    y = np.arange(len(frame))
    ax.barh(y, frame["delta_placebo_minus_real"], height=0.62,
            color=SKY, edgecolor=INK, linewidth=0.5,
            label="marginal-exposure component (Placebo $-$ Real)")
    ax.barh(y, frame["theta_rule_minus_placebo"], height=0.62,
            left=frame["delta_placebo_minus_real"], color=ORANGE,
            edgecolor=INK, linewidth=0.5, hatch="///",
            label=r"relational increment ($\theta$)")
    ax.plot(frame["delta_rule_minus_real"], y, "D", color=INK,
            markersize=3.8, markeredgewidth=0.6, markeredgecolor="white",
            label="total response (Rule $-$ Real)", linestyle="none")
    ax.axvline(0.0, color=GRAY, linewidth=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([str(int(s)) for s in frame["construction_seed"]],
                       fontsize=5.6)
    ax.set_ylim(-3.6, len(frame) - 0.2)
    low = min(float(frame["delta_rule_minus_real"].min()),
              float(frame["delta_placebo_minus_real"].min()), 0.0)
    high = max(float(frame["delta_rule_minus_real"].max()),
               float(frame["delta_placebo_minus_real"].max()), 0.0)
    pad = 0.08 * max(abs(low), abs(high), 1e-3)
    ax.set_xlim(low - pad, high + pad)
    ax.set_xlabel("payload-destination effect $C_P$ (exact macro recall)",
                  fontsize=7.2)
    ax.set_ylabel("generator realization", fontsize=7.2)
    ax.set_title("(a)  Random-Forest response decomposition\nper generator "
                 "realization", fontsize=7.0, loc="left", color=INK)
    ax.tick_params(labelsize=6.4)
    ax.legend(fontsize=6.0, loc="lower left", frameon=False, ncol=1)


def _panel_b(ax, decomposition: pd.DataFrame, verdict: dict[str, Any]) -> None:
    primary = verdict["primary"]
    theta = decomposition["theta_rule_minus_placebo"].to_numpy(dtype=float)
    delta = float(primary["delta"])
    strict = verdict.get("delta_strict")
    x = np.arange(len(theta))

    ax.axhspan(-delta, delta, color=SKY, alpha=0.18, zorder=0)
    ax.axhline(delta, color=BLUE, linewidth=0.9, linestyle="--",
               label=rf"inherited margin $\pm\delta$ = {delta:.4f}")
    ax.axhline(-delta, color=BLUE, linewidth=0.9, linestyle="--")
    if strict is not None:
        strict = float(strict)
        ax.axhline(strict, color=VERMILLION, linewidth=0.9, linestyle=":",
                   label=rf"strict margin $\pm\delta_{{\rm strict}}$ = "
                         rf"{strict:.4f}")
        ax.axhline(-strict, color=VERMILLION, linewidth=0.9, linestyle=":")
    ax.axhline(0.0, color=GRAY, linewidth=0.8, zorder=0)
    ax.plot(x, theta, "o", color=ORANGE, markersize=4.0, markeredgewidth=0.6,
            markeredgecolor=INK, linestyle="none",
            label=r"$\theta_g$ per generator realization")
    mean = float(primary["theta_mean"])
    ax.errorbar(
        [len(theta) + 1.2], [mean],
        yerr=[[mean - float(primary["ci90_low"])],
              [float(primary["ci90_high"]) - mean]],
        fmt="D", color=INK, markersize=4.6, capsize=3.0, linewidth=1.1,
        label="aggregate mean, 90% TOST interval")
    ax.set_xlim(-1.2, len(theta) + 2.6)
    # Reserve empty band below -delta so the legend never sits over data or
    # over either margin line.
    ax.set_ylim(-delta * 1.62, delta * 1.14)
    ax.set_xticks([])
    ax.set_xlabel("twenty prospective generator realizations "
                  r"($n = 20$, $\mathrm{df} = 19$)", fontsize=7.2)
    ax.set_ylabel(r"relational increment $\theta_g$", fontsize=7.2)
    strict_verdict = str(verdict["verdict_strict_margin"])
    ax.set_title(
        f"(b)  Dual-margin verdict, and the two margins disagree\n"
        f"inherited $\\delta$ \u2192 Branch {primary['verdict']} "
        f"({VERDICT_TEXT[str(primary['verdict'])]})\n"
        f"$\\delta_{{\\rm strict}}$ \u2192 Branch {strict_verdict} "
        f"({VERDICT_TEXT[strict_verdict]})",
        fontsize=7.0, loc="left", color=INK)
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.8, loc="lower center", frameon=False, ncol=2,
              bbox_to_anchor=(0.52, 0.045))


# Panel (c) gets one sub-axis per cell rather than one shared 0-1 axis.  The
# two cells sit an order of magnitude apart, so on a shared axis the P = 1
# bars -- the cell the panel exists to show, and the one the caption says
# carries the entire contrast -- stand at 8% of the height and the Rule versus
# placebo gap collapses to a hairline.  Independent tops fix that: the P = 1
# top clears its tallest bar by just enough to leave label room, so the
# 0.0816 versus 0.0771 gap spans several points, while the P = 0 top sits just
# above 1.0 so saturation still reads as bars filling their axis.  A log scale
# is not available here -- the real-only reference is exactly 0.000 in both
# cells and would vanish.
PANEL_C_WSPACE = 0.30
PANEL_C_TOP_P1 = 0.104
PANEL_C_TOP_P0 = 1.06


def _panel_c(fig, subspec, verdict: dict[str, Any]) -> None:
    arms = [
        ("real-only reference", GRAY, ""),
        ("Rule arm", ORANGE, "///"),
        ("placebo arm", SKY, "\\\\\\"),
    ]
    cells = [
        ("$P = 1$ (out of grammar)",
         verdict["p1_cell_summary"]["equal_macro"], PANEL_C_TOP_P1),
        ("$P = 0$ (in grammar)",
         verdict["p0_cell_summary"]["equal_macro"], PANEL_C_TOP_P0),
    ]
    width = 0.26
    inner = subspec.subgridspec(1, 2, wspace=PANEL_C_WSPACE)
    axes = []
    for column, (group, cell, top) in enumerate(cells):
        ax = fig.add_subplot(inner[0, column])
        axes.append(ax)
        values = [cell["real_recall"], cell["rule_recall_mean"],
                  cell["placebo_recall_mean"]]
        for index, ((label, colour, hatch), value) in enumerate(
                zip(arms, values)):
            position = (index - 1) * width
            # The real-only bar is exactly zero in both cells; it is still
            # drawn, so its edge marks the baseline, and it still carries its
            # 0.000 label.  The forest never types an attack correctly without
            # augmentation, and that is a reported finding, not an omission.
            ax.bar(position, value, width=width, color=colour,
                   edgecolor=INK, linewidth=0.5, hatch=hatch, label=label)
            # Offsets are fractions of each sub-axis's own top so the labels
            # sit identically on both scales.
            inside = value > 0.5 * top
            ax.text(position,
                    value - 0.055 * top if inside else value + 0.022 * top,
                    f"{value:.3f}", ha="center",
                    va="top" if inside else "bottom", fontsize=5.2,
                    color="white" if inside else INK)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([0.0])
        ax.set_xticklabels([group], fontsize=6.8)
        ax.set_ylim(0.0, top)
        ax.tick_params(labelsize=6.4)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    left, _right = axes
    left.set_ylabel("exact-macro recall, canonical strata", fontsize=7.2)
    saturation = verdict["p0_saturation"]
    left.set_title(
        f"(c)  Mandatory cell secondaries, independent scale per cell\n"
        f"$P = 0$ is {saturation['determination']}: Rule $-$ placebo = "
        f"{saturation['rule_minus_placebo']:+.2e}, so it cancels from "
        r"$\theta$",
        fontsize=7.0, loc="left", color=INK, pad=15.0)
    # One shared legend spanning both sub-axes, centred over the pair: each
    # sub-axis is one unit wide with a PANEL_C_WSPACE gap between them, so the
    # midpoint sits that many left-axes widths from the left-axes origin.  It
    # sits in the gap between the axes and the title rather than inside the
    # frame, because spanning two frames it would otherwise land on the right
    # sub-axis's top tick label.  It is a figure legend rather than an axes
    # legend because an axes legend belonging to the left sub-axis is painted
    # underneath the right sub-axis's background and loses its third entry.
    fig.legend(*left.get_legend_handles_labels(), fontsize=6.0,
               loc="lower center", frameon=False, ncol=3,
               bbox_to_anchor=((2.0 + PANEL_C_WSPACE) / 2.0, 1.012),
               bbox_transform=left.transAxes, columnspacing=1.0,
               handlelength=1.4)


def _panel_d(ax, family: pd.DataFrame, verdict: dict[str, Any]) -> None:
    cnn = family["theta_rule_minus_placebo_cnn"].to_numpy(dtype=float)
    rf = family["theta_rule_minus_placebo_rf"].to_numpy(dtype=float)
    summary = verdict["cnn_rf_family_comparison"]
    low = float(min(cnn.min(), rf.min()))
    high = float(max(cnn.max(), rf.max()))
    pad = 0.10 * max(high - low, 1e-3)
    span = [low - pad, high + pad]
    ax.plot(span, span, color=GRAY, linewidth=0.8, linestyle="--",
            label="equality")
    ax.axhline(0.0, color=GRID, linewidth=0.7, zorder=0)
    ax.axvline(0.0, color=GRID, linewidth=0.7, zorder=0)
    ax.plot(cnn, rf, "o", color=BLUE, markersize=4.0, markeredgewidth=0.6,
            markeredgecolor="white", linestyle="none")
    ax.set_xlim(span)
    ax.set_ylim(span)
    ax.set_xlabel(r"CNN $\theta_g$ (E16)", fontsize=7.2)
    ax.set_ylabel(r"Random-Forest $\theta_g$ (E17)", fontsize=7.2)
    ax.set_title(
        f"(d)  Paired family comparison, descriptive only\n"
        f"Pearson $r$ = {summary['pearson_r']:.3f},  "
        f"Spearman $\\rho$ = {summary['spearman_rho']:.3f}",
        fontsize=7.0, loc="left", color=INK)
    ax.tick_params(labelsize=6.4)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], color=GRAY, linewidth=0.8, linestyle="--",
               label="equality"),
        Line2D([], [], color=BLUE, marker="o", linestyle="none",
               markersize=4.0, label="generator realization"),
    ], fontsize=6.0, loc="best", frameon=False)


def build_figure(inputs: dict[str, Any]) -> Any:
    fig = plt.figure(figsize=(7.45, 8.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.05, 1.05], hspace=0.34,
                            wspace=0.26)
    _panel_a(fig.add_subplot(grid[0, 0]), inputs["decomposition"])
    _panel_b(fig.add_subplot(grid[0, 1]), inputs["decomposition"],
             inputs["verdict"])
    _panel_c(fig, grid[1, 0], inputs["verdict"])
    _panel_d(fig.add_subplot(grid[1, 1]), inputs["family"], inputs["verdict"])
    return fig


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


def _render_twice(inputs: dict[str, Any]) -> tuple[bytes, bytes, dict[str, Any]]:
    """Render twice and require byte identity before publishing."""
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
        raise _stop(f"repeated render is not byte-identical for: {mismatched}")
    return first[0], first[1], {
        "passes": 2, "identical": True,
        "pdf_sha256": _sha256_bytes(first[0]),
        "png_sha256": _sha256_bytes(first[1]),
    }


def _archive_superseded(present: Sequence[Path], stamp: str) -> dict[str, Any]:
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


def execute(*, republish: bool = False) -> dict[str, Any]:
    inputs = load_inputs()
    paths = figure_paths()
    present = [path for path in paths.values() if path.exists()]
    if present and not republish:
        raise _stop(
            "refusing to overwrite existing figure outputs: "
            f"{[str(p.relative_to(REPO)) for p in present]}; pass --republish "
            "to archive them under journal/results/archive/ and re-render at "
            "the same version")

    analysis = e17an.output_paths()
    input_paths = {key: analysis[key] for key in
                   ("decomposition", "p1_cell", "p0_cell", "family", "verdict")}
    source = _source_record_for_render(input_paths)
    environment = _environment_record(REPO)
    registration = e17train.assert_registration_committed()

    pdf_bytes, png_bytes, identity = _render_twice(inputs)
    stamp = e17train.utc_now()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    archived = _archive_superseded(
        present, stamp.replace("-", "").replace(":", "")) if present else {}

    verdict = inputs["verdict"]
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
        "verdict": verdict["verdict"],
        "verdict_text": VERDICT_TEXT[str(verdict["verdict"])],
        "verdict_strict_margin": verdict["verdict_strict_margin"],
        "margin_disagreement": verdict["margin_disagreement"],
        "delta": verdict["delta"],
        "delta_strict": verdict["delta_strict"],
        "theta_mean": verdict["primary"]["theta_mean"],
        "outputs": {kind: str(paths[kind].relative_to(REPO))
                    for kind in ("pdf", "png")},
        "outputs_sha256": {"pdf": _sha256_bytes(pdf_bytes),
                           "png": _sha256_bytes(png_bytes)},
        "render_byte_identity": identity,
        "superseded": archived,
        "recomputes_outcomes": False,
    }
    publish_and_cleanup([
        (_stage_bytes(paths["pdf"], pdf_bytes), paths["pdf"]),
        (_stage_bytes(paths["png"], png_bytes), paths["png"]),
        (_stage_json(paths["provenance"], provenance), paths["provenance"]),
    ])
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="render and publish the figure")
    parser.add_argument(
        "--republish", action="store_true",
        help="archive the superseded pdf/png/provenance under "
             "journal/results/archive/<UTC stamp>/ and re-render at the same "
             "version; nothing is deleted and OUTPUT_VERSION is unchanged")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        analysis = e17an.output_paths()
        print(json.dumps({
            "stage": "e17_figure",
            "mode": "read-only",
            "inputs_available": {
                key: analysis[key].exists() for key in
                ("decomposition", "p1_cell", "p0_cell", "family", "verdict")},
            "outputs": {k: str(v.relative_to(REPO))
                        for k, v in figure_paths().items()},
        }, indent=2, sort_keys=True))
        return 0
    print(json.dumps(execute(republish=args.republish), indent=2,
                     sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
