#!/usr/bin/env python3
"""Publish the E16 placebo response-share summary as a first-class artifact.

The manuscript quotes a single headline share -- the fraction of the Rule
generator's attack-type recall response that its marginal-matched placebo twin
already reproduces.  Until now that number was only recoverable by hand from the
twenty primary rows of ``e16_decomposition_by_construction_v2.csv``.  This
script recomputes it from that published table and writes a one-row summary so
the quoted value has an artifact of its own.

The estimator is a RATIO OF MEANS: the mean placebo-minus-real response over the
twenty generator realizations divided by the mean Rule-minus-real response over
the same twenty realizations.  It is deliberately NOT a mean of per-realization
ratios; the per-realization share distribution is wide and its centre differs
from the ratio of means, so both are reported side by side to keep the
distinction auditable.

This script computes no new outcome.  It reads a published E16 score table only
and writes new, no-clobber outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "journal" / "results" / "tables"
LOG_DIR = REPO / "journal" / "results" / "logs"

DECOMPOSITION = TABLE_DIR / "e16_decomposition_by_construction_v2.csv"
PRIMARY_SUMMARY = TABLE_DIR / "e16_primary_theta_summary_v2.csv"
OUTPUT = TABLE_DIR / "e16_response_share_v1.csv"
PROVENANCE = LOG_DIR / "e16_response_share_v1.json"

EXPECTED_REALIZATIONS = 20
ESTIMATOR = (
    "ratio_of_means: mean(delta_placebo_minus_real) / mean(delta_rule_minus_real) "
    "over the n=20 primary generator realizations; NOT a mean of per-realization "
    "ratios"
)
THETA_TOLERANCE = 1e-12


class ResponseShareError(RuntimeError):
    """Raised when a frozen-input, consistency, or no-clobber gate fails."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ResponseShareError(f"refusing to overwrite published output: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compute() -> pd.DataFrame:
    decomposition = pd.read_csv(DECOMPOSITION)
    primary = decomposition[decomposition["role"] == "primary"]
    if len(primary) != EXPECTED_REALIZATIONS:
        raise ResponseShareError(
            f"expected {EXPECTED_REALIZATIONS} primary realizations, "
            f"found {len(primary)}"
        )
    if primary["construction_seed"].nunique() != EXPECTED_REALIZATIONS:
        raise ResponseShareError("primary generator realizations are not unique")

    rule = primary["delta_rule_minus_real"].to_numpy(dtype=float)
    placebo = primary["delta_placebo_minus_real"].to_numpy(dtype=float)
    theta = primary["theta_rule_minus_placebo"].to_numpy(dtype=float)

    mean_rule = float(rule.mean())
    mean_placebo = float(placebo.mean())
    theta_mean = float(theta.mean())
    if not np.isclose(theta_mean, mean_rule - mean_placebo, rtol=0.0, atol=1e-12):
        raise ResponseShareError("theta column disagrees with the arm means")

    registered = pd.read_csv(PRIMARY_SUMMARY).iloc[0]
    registered_theta = float(registered["theta_mean"])
    if abs(theta_mean - registered_theta) > THETA_TOLERANCE:
        raise ResponseShareError(
            "recomputed theta does not match the registered primary summary: "
            f"{theta_mean!r} vs {registered_theta!r}"
        )

    response_share = mean_placebo / mean_rule
    per_realization_share = placebo / rule
    q1, median, q3 = (
        float(value) for value in np.percentile(per_realization_share, [25, 50, 75])
    )

    row = {
        "n_realizations": EXPECTED_REALIZATIONS,
        "degrees_of_freedom": EXPECTED_REALIZATIONS - 1,
        "inferential_unit": "generator_realization",
        "contrast": "payload_destination_C_P_exact_macro_attack_type_recall",
        "mean_rule_minus_real": mean_rule,
        "sd_rule_minus_real": float(rule.std(ddof=1)),
        "mean_placebo_minus_real": mean_placebo,
        "sd_placebo_minus_real": float(placebo.std(ddof=1)),
        "theta_mean": theta_mean,
        "theta_sd": float(theta.std(ddof=1)),
        "response_share": response_share,
        "response_share_pct": 100.0 * response_share,
        "estimator": ESTIMATOR,
        "mean_of_per_realization_ratios": float(per_realization_share.mean()),
        "per_realization_share_median": median,
        "per_realization_share_q1": q1,
        "per_realization_share_q3": q3,
        "registered_theta_mean": registered_theta,
        "source_table": str(DECOMPOSITION.relative_to(REPO)),
    }
    return pd.DataFrame.from_records([row])


def main() -> None:
    required = (DECOMPOSITION, PRIMARY_SUMMARY, Path(__file__))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ResponseShareError(f"missing required input(s): {missing}")
    if OUTPUT.exists() or PROVENANCE.exists():
        raise ResponseShareError(
            f"refusing to overwrite outputs: {OUTPUT}, {PROVENANCE}"
        )

    result = compute()
    atomic_write_bytes(
        OUTPUT, result.to_csv(index=False, lineterminator="\n").encode("utf-8")
    )

    row = result.iloc[0]
    provenance = {
        "schema_version": "e16.response_share.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "derived_summary_of_published_e16_table",
        "registered_analysis_unchanged": True,
        "inputs": {
            str(path.relative_to(REPO)): sha256(path)
            for path in (DECOMPOSITION, PRIMARY_SUMMARY)
        },
        "source": {str(Path(__file__).relative_to(REPO)): sha256(Path(__file__))},
        "estimator": ESTIMATOR,
        "result": {
            "n_realizations": int(row["n_realizations"]),
            "mean_rule_minus_real": float(row["mean_rule_minus_real"]),
            "mean_placebo_minus_real": float(row["mean_placebo_minus_real"]),
            "theta_mean": float(row["theta_mean"]),
            "response_share": float(row["response_share"]),
            "response_share_4dp": round(float(row["response_share"]), 4),
            "mean_of_per_realization_ratios": float(
                row["mean_of_per_realization_ratios"]
            ),
            "per_realization_share_iqr": [
                float(row["per_realization_share_q1"]),
                float(row["per_realization_share_q3"]),
            ],
        },
        "limitations": [
            "The share is a ratio of means over twenty generator realizations; "
            "it is not a mean of per-realization ratios and carries no interval.",
            "The per-realization share distribution is wide, so the ratio of "
            "means describes the pooled response, not a typical realization.",
            "The quantity is descriptive; the registered equivalence verdict "
            "rests on theta and its TOST interval, not on this share.",
        ],
        "output": {str(OUTPUT.relative_to(REPO)): sha256(OUTPUT)},
    }
    atomic_write_bytes(
        PROVENANCE,
        (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"wrote {OUTPUT.relative_to(REPO)}")
    print(f"wrote {PROVENANCE.relative_to(REPO)}")
    print(f"response_share = {float(row['response_share']):.4f}")


if __name__ == "__main__":
    main()
