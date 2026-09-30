#!/usr/bin/env python3
"""Render the frozen E14 L4 factorial-localization figure.

The figure has two deliberately separated evidence panels:

* panel (a) shows the preregistered primary, canonical-ID, equal-macro exact
  Rule-minus-matched-real factorial effects and their nominal 95% Student-t
  intervals;
* panel (b) shows the corresponding Gear and RPM point estimates only.  These
  attack-specific estimates are supporting analyses and are not presented as
  separately tested findings.

The plotted estimands are equal-training-budget augmentation-pipeline
contrasts, marginal over the other construction factors.  They are not pure
causal effects of physical CAN mechanisms.  Every Rule interpretation remains
conditional on construction seed 314159.

Inputs (read-only, frozen by SHA-256):
  journal/results/tables/e14_l4_factorial_effects_v1.csv
  journal/results/logs/e14_l4_factorial_artifact_manifest_v1.json

Outputs (versioned, staged, and published without overwrite):
  journal/results/figures/e14_l4_factorial_localization_v1.pdf
  journal/results/figures/e14_l4_factorial_localization_v1.png
  journal/results/logs/e14_l4_factorial_figure_provenance_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Keep the rendered PDF metadata deterministic across reruns.
os.environ["SOURCE_DATE_EPOCH"] = "0"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT = Path(__file__).resolve()
JOURNAL_ROOT = SCRIPT.parents[1]
REPO_ROOT = JOURNAL_ROOT.parent
RESULTS_ROOT = JOURNAL_ROOT / "results"

EFFECTS_PATH = RESULTS_ROOT / "tables" / "e14_l4_factorial_effects_v1.csv"
UPSTREAM_MANIFEST_PATH = (
    RESULTS_ROOT / "logs" / "e14_l4_factorial_artifact_manifest_v1.json"
)
PDF_OUTPUT = (
    RESULTS_ROOT / "figures" / "e14_l4_factorial_localization_v1.pdf"
)
PNG_OUTPUT = (
    RESULTS_ROOT / "figures" / "e14_l4_factorial_localization_v1.png"
)
PROVENANCE_OUTPUT = (
    RESULTS_ROOT / "logs" / "e14_l4_factorial_figure_provenance_v1.json"
)

EXPECTED_EFFECTS_BYTES = 2_818_084
EXPECTED_EFFECTS_SHA256 = (
    "63ea3c4bb09167033c1b2c8d7760c2329fe701de6608b274d26a6cb357e4c21f"
)
EXPECTED_MANIFEST_BYTES = 23_584
EXPECTED_MANIFEST_SHA256 = (
    "41243eca977c5acf4b115b9f30d7a9c905d08867b23c3d5751d7e8a02246897b"
)
EXPECTED_TABLE_ROWS = 15_552
EXPECTED_SEEDS = 5
EXPECTED_VERDICT = "F-MAIN+attack-heterogeneous"
EXPECTED_HOLM_FAMILY = "primary_exact_rule_minus_real"
EFFECTS = ("P", "S", "D", "PS", "PD", "SD", "PSD")
HETEROGENEOUS_EFFECTS = ("S", "PS", "SD", "PSD")

EFFECT_LABELS = {
    "P": "P — payload-role destination",
    "S": "S — frame-mask layout/span",
    "D": "D — payload dynamics",
    "PS": "P×S",
    "PD": "P×D",
    "SD": "S×D",
    "PSD": "P×S×D",
}

REQUIRED_COLUMNS = {
    "contrast",
    "endpoint",
    "id_scope",
    "attack_scope",
    "effect",
    "row_type",
    "unit_id",
    "value",
    "status",
    "n_available",
    "mean",
    "sd",
    "ci_low",
    "ci_high",
    "raw_p",
    "holm_adjusted_p",
    "p_status",
    "degenerate",
    "strict_positive",
    "strict_negative",
    "resolved_and_stable",
    "holm_family",
    "legacy_budget_classification",
    "localization_verdict",
    "attack_heterogeneous_effects",
}

# Okabe-Ito colors, with marker/fill redundancy for grayscale reproduction.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#666666"
INK = "#222222"
GRID = "#D9D9D9"
LIGHT_HETERO = "#FFF3CD"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})


@dataclass(frozen=True)
class EffectSummary:
    """One validated summary row used by the figure."""

    effect: str
    mean: float
    sd: float
    ci_low: float
    ci_high: float
    raw_p: float | None
    holm_adjusted_p: float | None
    strict_positive: int
    strict_negative: int
    resolved_and_stable: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes repository: {path}")
    return resolved.relative_to(root).as_posix()


def artifact_record(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required regular artifact is missing: {path}")
    return {
        "path": repo_relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _require_frozen_artifact(
    record: Mapping[str, object],
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    if (
        record["bytes"] != expected_bytes
        or record["sha256"] != expected_sha256
    ):
        raise ValueError(
            "frozen E14 input identity mismatch for "
            f"{record['path']}: expected bytes={expected_bytes}, "
            f"sha256={expected_sha256}; observed bytes={record['bytes']}, "
            f"sha256={record['sha256']}"
        )


def validate_upstream_manifest() -> tuple[dict[str, object], dict[str, object]]:
    manifest_record = artifact_record(UPSTREAM_MANIFEST_PATH)
    _require_frozen_artifact(
        manifest_record,
        expected_bytes=EXPECTED_MANIFEST_BYTES,
        expected_sha256=EXPECTED_MANIFEST_SHA256,
    )
    payload = json.loads(UPSTREAM_MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "e14_l4_factorial_artifact_manifest_v1"
    ):
        raise ValueError("unexpected E14 artifact-manifest schema")
    if payload.get("technical_status") != "T-PASS":
        raise ValueError("E14 artifact manifest is not technically complete")
    if payload.get("localization_verdict") != EXPECTED_VERDICT:
        raise ValueError("E14 artifact-manifest verdict changed")

    expected_effect_record = {
        "path": repo_relative(EFFECTS_PATH),
        "bytes": EXPECTED_EFFECTS_BYTES,
        "sha256": EXPECTED_EFFECTS_SHA256,
    }
    if payload.get("effects") != expected_effect_record:
        raise ValueError(
            "E14 artifact manifest does not bind the expected effects table"
        )

    expected_family = {
        "attack_scope": "equal_macro",
        "contrast": "rule_ms-real_ms",
        "effects": list(EFFECTS),
        "endpoint": "exact_recall",
        "id_scope": "canonical",
    }
    observed_family = (
        payload.get("holm_families", {})
        .get(EXPECTED_HOLM_FAMILY)
    )
    if observed_family != expected_family:
        raise ValueError("primary E14 Holm-family declaration changed")
    return payload, manifest_record


def _finite_float(row: pd.Series, field: str) -> float:
    raw = str(row[field])
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{row['attack_scope']}/{row['effect']}: "
            f"{field} is not numeric: {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"{row['attack_scope']}/{row['effect']}: "
            f"{field} is not finite"
        )
    return value


def _optional_float(row: pd.Series, field: str) -> float | None:
    if str(row[field]) == "":
        return None
    return _finite_float(row, field)


def _strict_bool(row: pd.Series, field: str) -> bool:
    raw = str(row[field])
    if raw not in {"True", "False"}:
        raise ValueError(
            f"{row['attack_scope']}/{row['effect']}: "
            f"{field} is not a strict boolean: {raw!r}"
        )
    return raw == "True"


def _strict_int(row: pd.Series, field: str) -> int:
    raw = str(row[field])
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{row['attack_scope']}/{row['effect']}: "
            f"{field} is not an integer: {raw!r}"
        ) from exc
    return value


def _validate_family_rows(
    frame: pd.DataFrame,
    *,
    attack_scope: str,
) -> pd.DataFrame:
    family = frame[
        frame["contrast"].eq("rule_ms-real_ms")
        & frame["endpoint"].eq("exact_recall")
        & frame["id_scope"].eq("canonical")
        & frame["attack_scope"].eq(attack_scope)
        & frame["effect"].isin(EFFECTS)
    ].copy()
    expected_counts = {"seed": 35, "block": 21, "summary": 7}
    observed_counts = dict(Counter(family["row_type"]))
    if len(family) != 63 or observed_counts != expected_counts:
        raise ValueError(
            f"{attack_scope}: incomplete E14 effect family; "
            f"expected 63 rows {expected_counts}, observed "
            f"{len(family)} rows {observed_counts}"
        )
    summaries = family[
        family["row_type"].eq("summary")
        & family["unit_id"].eq("summary")
    ].copy()
    if (
        len(summaries) != len(EFFECTS)
        or set(summaries["effect"]) != set(EFFECTS)
        or summaries["effect"].duplicated().any()
    ):
        raise ValueError(
            f"{attack_scope}: expected one summary for each of {EFFECTS}"
        )
    indexed = summaries.set_index("effect", drop=False)
    if not indexed.index.is_unique:
        raise ValueError(f"{attack_scope}: duplicate summary effect index")
    return indexed


def _parse_summary(
    row: pd.Series,
    *,
    inferential: bool,
) -> EffectSummary:
    if row["status"] != "complete":
        raise ValueError(
            f"{row['attack_scope']}/{row['effect']}: summary is incomplete"
        )
    if _strict_int(row, "n_available") != EXPECTED_SEEDS:
        raise ValueError(
            f"{row['attack_scope']}/{row['effect']}: "
            "expected five paired pipeline seeds"
        )
    if _strict_bool(row, "degenerate"):
        raise ValueError(
            f"{row['attack_scope']}/{row['effect']}: "
            "degenerate summary cannot enter this figure"
        )

    mean = _finite_float(row, "mean")
    value = _finite_float(row, "value")
    sd = _finite_float(row, "sd")
    ci_low = _finite_float(row, "ci_low")
    ci_high = _finite_float(row, "ci_high")
    if not math.isclose(mean, value, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f"{row['attack_scope']}/{row['effect']}: value/mean mismatch"
        )
    if sd < 0.0 or not ci_low <= mean <= ci_high:
        raise ValueError(
            f"{row['attack_scope']}/{row['effect']}: invalid SD or CI"
        )

    raw_p = _optional_float(row, "raw_p")
    holm_p = _optional_float(row, "holm_adjusted_p")
    resolved = _strict_bool(row, "resolved_and_stable")
    strict_positive = _strict_int(row, "strict_positive")
    strict_negative = _strict_int(row, "strict_negative")
    if min(strict_positive, strict_negative) < 0:
        raise ValueError("strict-sign counts cannot be negative")
    if strict_positive + strict_negative > EXPECTED_SEEDS:
        raise ValueError("strict-sign counts exceed the five seed effects")

    if inferential:
        if (
            row["p_status"] != "inferential_p"
            or row["holm_family"] != EXPECTED_HOLM_FAMILY
            or row["localization_verdict"] != EXPECTED_VERDICT
            or row["attack_heterogeneous_effects"]
            != ",".join(HETEROGENEOUS_EFFECTS)
            or raw_p is None
            or holm_p is None
        ):
            raise ValueError(
                f"equal_macro/{row['effect']}: primary inference metadata "
                "does not match the preregistered E14 family"
            )
    else:
        if (
            row["p_status"] != "not_tested_supporting"
            or row["holm_family"] != ""
            or row["localization_verdict"] != ""
            or raw_p is not None
            or holm_p is not None
            or resolved
        ):
            raise ValueError(
                f"{row['attack_scope']}/{row['effect']}: "
                "attack-specific row was promoted beyond supporting status"
            )

    return EffectSummary(
        effect=str(row["effect"]),
        mean=mean,
        sd=sd,
        ci_low=ci_low,
        ci_high=ci_high,
        raw_p=raw_p,
        holm_adjusted_p=holm_p,
        strict_positive=strict_positive,
        strict_negative=strict_negative,
        resolved_and_stable=resolved,
    )


def load_validated_effects(
) -> tuple[
    list[EffectSummary],
    dict[str, list[EffectSummary]],
    dict[str, object],
]:
    upstream_manifest, manifest_record = validate_upstream_manifest()
    effects_record = artifact_record(EFFECTS_PATH)
    _require_frozen_artifact(
        effects_record,
        expected_bytes=EXPECTED_EFFECTS_BYTES,
        expected_sha256=EXPECTED_EFFECTS_SHA256,
    )

    frame = pd.read_csv(
        EFFECTS_PATH,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"E14 effects table is missing columns: {missing}")
    if len(frame) != EXPECTED_TABLE_ROWS:
        raise ValueError(
            f"E14 effects table row count changed: {len(frame)}"
        )

    primary_frame = _validate_family_rows(frame, attack_scope="equal_macro")
    primary = [
        _parse_summary(primary_frame.loc[effect], inferential=True)
        for effect in EFFECTS
    ]
    attack_specific: dict[str, list[EffectSummary]] = {}
    for attack in ("Gear", "RPM"):
        attack_frame = _validate_family_rows(frame, attack_scope=attack)
        attack_specific[attack] = [
            _parse_summary(attack_frame.loc[effect], inferential=False)
            for effect in EFFECTS
        ]

    resolved = [
        summary.effect for summary in primary
        if summary.resolved_and_stable
    ]
    if resolved != ["P"]:
        raise ValueError(
            f"expected P as the sole resolved primary effect; observed {resolved}"
        )
    for summary in primary:
        assert summary.holm_adjusted_p is not None
        if summary.effect == "P":
            if (
                summary.mean >= 0.0
                or summary.holm_adjusted_p >= 0.05
                or summary.strict_negative != EXPECTED_SEEDS
                or summary.strict_positive != 0
            ):
                raise ValueError("resolved P constraints changed")
            source_row = primary_frame.loc["P"]
            if source_row["legacy_budget_classification"] != "budget-stable":
                raise ValueError("P is no longer classified as budget-stable")
        elif (
            summary.resolved_and_stable
            or summary.holm_adjusted_p < 0.05
        ):
            raise ValueError(
                f"{summary.effect}: unexpected resolved primary effect"
            )

    by_attack = {
        attack: {summary.effect: summary for summary in summaries}
        for attack, summaries in attack_specific.items()
    }
    opposite_signs = {
        effect for effect in EFFECTS
        if (
            by_attack["Gear"][effect].mean
            * by_attack["RPM"][effect].mean
        ) < 0.0
    }
    if opposite_signs != set(HETEROGENEOUS_EFFECTS):
        raise ValueError(
            "Gear/RPM opposite-sign set changed: "
            f"expected {HETEROGENEOUS_EFFECTS}, "
            f"observed {sorted(opposite_signs)}"
        )
    if (
        by_attack["Gear"]["P"].mean >= 0.0
        or by_attack["RPM"]["P"].mean >= 0.0
    ):
        raise ValueError("P is not negative for both attack types")

    all_primary_limits = [
        value
        for summary in primary
        for value in (summary.ci_low, summary.ci_high)
    ]
    all_supporting_means = [
        summary.mean
        for summaries in attack_specific.values()
        for summary in summaries
    ]
    if min([*all_primary_limits, *all_supporting_means]) < -1.25:
        raise ValueError("validated effect falls below the fixed figure range")
    if max([*all_primary_limits, *all_supporting_means]) > 0.35:
        raise ValueError("validated effect exceeds the fixed figure range")

    provenance_inputs = {
        "effects_table": effects_record,
        "upstream_artifact_manifest": manifest_record,
        "upstream_manifest_schema": upstream_manifest["schema_version"],
    }
    return primary, attack_specific, provenance_inputs


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-1.25, 0.35)
    ax.set_xticks((-1.2, -0.9, -0.6, -0.3, 0.0, 0.3))
    ax.axvline(0.0, color=INK, linewidth=0.9, linestyle="--", zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_figure(
    primary: Sequence[EffectSummary],
    attack_specific: Mapping[str, Sequence[EffectSummary]],
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    positions = list(reversed(range(len(EFFECTS))))
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(7.35, 4.55),
        sharey=False,
    )

    for position, summary in zip(positions, primary):
        resolved = summary.resolved_and_stable
        color = BLUE if resolved else GRAY
        left.errorbar(
            summary.mean,
            position,
            xerr=[
                [summary.mean - summary.ci_low],
                [summary.ci_high - summary.mean],
            ],
            fmt="D" if resolved else "o",
            color=color,
            ecolor=color,
            markerfacecolor=color if resolved else "white",
            markeredgecolor=color,
            markeredgewidth=1.15,
            markersize=5.2,
            elinewidth=1.2,
            capsize=2.5,
            capthick=1.0,
            zorder=3,
        )
    left.set_yticks(positions)
    left.set_yticklabels(
        [EFFECT_LABELS[effect] for effect in EFFECTS],
        fontsize=8.6,
    )
    left.set_ylim(-0.65, len(EFFECTS) - 0.35)
    left.set_title(
        "(a) Primary exact effects",
        loc="left",
        fontsize=10.4,
        fontweight="bold",
        pad=8,
    )
    left.set_xlabel(
        "Rule − matched-real exact-recall effect",
        fontsize=8.7,
    )
    _style_axis(left)
    p_summary = primary[0]
    assert p_summary.holm_adjusted_p is not None
    left.annotate(
        f"Holm $p$={p_summary.holm_adjusted_p:.4f}; 5/5 negative",
        xy=(p_summary.ci_high, positions[0]),
        xytext=(-0.61, positions[0] + 0.02),
        fontsize=8.6,
        color=BLUE,
        ha="left",
        va="center",
    )
    attack_map = {
        attack: {summary.effect: summary for summary in summaries}
        for attack, summaries in attack_specific.items()
    }
    for position, effect in zip(positions, EFFECTS):
        if effect in HETEROGENEOUS_EFFECTS:
            right.axhspan(
                position - 0.38,
                position + 0.38,
                color=LIGHT_HETERO,
                alpha=0.55,
                zorder=0,
            )
        gear = attack_map["Gear"][effect].mean
        rpm = attack_map["RPM"][effect].mean
        right.plot(
            [gear, rpm],
            [position + 0.09, position - 0.09],
            color="#B0B0B0",
            linewidth=0.8,
            zorder=2,
        )
        right.scatter(
            gear,
            position + 0.09,
            marker="o",
            s=25,
            facecolor=BLUE,
            edgecolor=BLUE,
            linewidth=0.8,
            zorder=3,
        )
        right.scatter(
            rpm,
            position - 0.09,
            marker="s",
            s=23,
            facecolor=ORANGE,
            edgecolor=ORANGE,
            linewidth=0.8,
            zorder=3,
        )
    right.set_yticks(positions)
    right.set_yticklabels([
        f"{effect}†" if effect in HETEROGENEOUS_EFFECTS else effect
        for effect in EFFECTS
    ])
    right.set_ylim(-0.65, len(EFFECTS) - 0.35)
    right.set_title(
        "(b) Attack-specific point estimates",
        loc="left",
        fontsize=10.4,
        fontweight="bold",
        pad=8,
    )
    right.set_xlabel(
        "Rule − matched-real exact-recall effect",
        fontsize=8.7,
    )
    _style_axis(right)
    right.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", color=BLUE,
                markerfacecolor=BLUE, markersize=5.2, label="Gear",
            ),
            Line2D(
                [], [], marker="s", linestyle="none", color=ORANGE,
                markerfacecolor=ORANGE, markersize=5.0, label="RPM",
            ),
            Patch(
                facecolor=LIGHT_HETERO,
                edgecolor="none",
                alpha=0.75,
                label="† Opposite point-estimate signs",
            ),
        ],
        loc="upper right",
        fontsize=8.6,
        frameon=False,
        handletextpad=0.45,
        borderaxespad=0.4,
    )

    fig.subplots_adjust(
        left=0.19,
        right=0.985,
        bottom=0.14,
        top=0.91,
        wspace=0.30,
    )

    pdf_metadata = {
        "Title": "E14 L4 factorial localization",
        "Author": "WISA CAN IDS journal extension",
        "Subject": (
            "Matched-budget Rule-minus-real exact-recall factorial effects"
        ),
        "Creator": "make_e14_l4_factorial_figure.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata=pdf_metadata,
    )
    fig.savefig(
        png_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata={"Software": "make_e14_l4_factorial_figure.py"},
    )
    plt.close(fig)


def validate_staged_outputs(paths: Iterable[Path]) -> None:
    paths = list(paths)
    if len(paths) != 2:
        raise ValueError("expected exactly one PDF and one PNG")
    for path in paths:
        if not path.is_file() or path.stat().st_size < 10_000:
            raise ValueError(f"staged figure is missing or too small: {path}")
    pdf = next(path for path in paths if path.suffix == ".pdf")
    png = next(path for path in paths if path.suffix == ".png")
    if not pdf.read_bytes()[:5] == b"%PDF-":
        raise ValueError("staged PDF signature is invalid")
    with png.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("staged PNG signature is invalid")
    width, height = struct.unpack(">II", header[16:24])
    if width < 1_800 or height < 1_000:
        raise ValueError(
            f"staged PNG resolution is unexpectedly small: {width}x{height}"
        )


def git_metadata() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "worktree_clean": not status_lines,
        "status_lines": status_lines,
    }


def publish_no_clobber(pairs: Sequence[tuple[Path, Path]]) -> None:
    collisions = [
        str(destination)
        for _source, destination in pairs
        if destination.exists() or destination.is_symlink()
    ]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite E14 figure artifacts: "
            f"{collisions}"
        )
    published: list[tuple[Path, Path]] = []
    try:
        for source, destination in pairs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            published.append((source, destination))
    except BaseException:
        for source, destination in reversed(published):
            try:
                source_stat = source.stat(follow_symlinks=False)
                destination_stat = destination.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (
                source_stat.st_dev == destination_stat.st_dev
                and source_stat.st_ino == destination_stat.st_ino
            ):
                destination.unlink()
        raise


def build(*, check_only: bool = False) -> dict[str, object]:
    primary, attack_specific, provenance_inputs = load_validated_effects()
    if check_only:
        return {
            "status": "inputs_valid",
            "primary_effects": list(EFFECTS),
            "paired_pipeline_seeds": EXPECTED_SEEDS,
            "localization_verdict": EXPECTED_VERDICT,
        }

    destinations = (PDF_OUTPUT, PNG_OUTPUT, PROVENANCE_OUTPUT)
    frozen_wisa = (REPO_ROOT / "wisa").resolve()
    for destination in destinations:
        resolved = destination.resolve()
        if resolved == frozen_wisa or resolved.is_relative_to(frozen_wisa):
            raise ValueError("refusing to publish under frozen wisa/")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite E14 figure artifact: {destination}"
            )

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    source_state = git_metadata()
    temporary = Path(tempfile.mkdtemp(
        prefix=".e14-l4-factorial-figure-",
        dir=RESULTS_ROOT,
    ))
    published = False
    try:
        stage_figures = temporary / "figures"
        stage_logs = temporary / "logs"
        stage_figures.mkdir()
        stage_logs.mkdir()
        staged_pdf = stage_figures / PDF_OUTPUT.name
        staged_png = stage_figures / PNG_OUTPUT.name
        staged_provenance = stage_logs / PROVENANCE_OUTPUT.name

        render_figure(
            primary,
            attack_specific,
            pdf_path=staged_pdf,
            png_path=staged_png,
        )
        validate_staged_outputs((staged_pdf, staged_png))

        output_records = [
            {
                "path": repo_relative(destination),
                "bytes": staged.stat().st_size,
                "sha256": sha256_file(staged),
            }
            for staged, destination in (
                (staged_pdf, PDF_OUTPUT),
                (staged_png, PNG_OUTPUT),
            )
        ]
        provenance = {
            "schema_version": (
                "e14_l4_factorial_figure_provenance_v1"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": "E14 matched L4 counterfactual factorial",
            "localization_verdict": EXPECTED_VERDICT,
            "scientific_scope": {
                "contrast": "rule_ms-real_ms",
                "training_budget": "matched_6156_update_maximum",
                "endpoint": "exact_recall",
                "id_scope": "canonical",
                "primary_attack_scope": "equal_macro",
                "primary_effects": list(EFFECTS),
                "primary_interval": (
                    "nominal_two_sided_95pct_student_t_df4"
                ),
                "primary_multiplicity": (
                    "holm_adjusted_across_seven_factorial_effects"
                ),
                "resolved_primary_effects": ["P"],
                "supporting_attack_scopes": ["Gear", "RPM"],
                "supporting_inference": (
                    "point_estimates_only_not_separately_tested"
                ),
                "attack_heterogeneous_effects": list(
                    HETEROGENEOUS_EFFECTS
                ),
                "paired_pipeline_seeds": EXPECTED_SEEDS,
                "rule_construction_seed": 314159,
                "estimand_boundary": (
                    "equal-budget augmentation-pipeline contrast marginal "
                    "over other factors; not an isolated causal or physical "
                    "mechanism"
                ),
            },
            "selected_rows": {
                "primary_summary_rows": len(primary),
                "supporting_summary_rows": sum(
                    len(rows) for rows in attack_specific.values()
                ),
                "validated_rows_per_attack_scope": {
                    "seed": 35,
                    "block": 21,
                    "summary": 7,
                },
            },
            "inputs": provenance_inputs,
            "script": artifact_record(SCRIPT),
            "git": source_state,
            "outputs": output_records,
            "publication": {
                "method": (
                    "same-filesystem staging followed by no-clobber hard links"
                ),
                "overwrites_allowed": False,
            },
        }
        staged_provenance.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        json.loads(staged_provenance.read_text(encoding="utf-8"))
        publish_no_clobber((
            (staged_pdf, PDF_OUTPUT),
            (staged_png, PNG_OUTPUT),
            (staged_provenance, PROVENANCE_OUTPUT),
        ))
        published = True
    finally:
        if published:
            shutil.rmtree(temporary)
        elif temporary.exists():
            # Preserve a failed staging directory for forensic inspection.
            pass

    return {
        "status": "published",
        "outputs": [
            repo_relative(PDF_OUTPUT),
            repo_relative(PNG_OUTPUT),
        ],
        "provenance": repo_relative(PROVENANCE_OUTPUT),
        "localization_verdict": EXPECTED_VERDICT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen E14 L4 factorial-localization figure."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate frozen inputs and selected evidence without publishing",
    )
    args = parser.parse_args()
    try:
        result = build(check_only=args.check_only)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
