#!/usr/bin/env python3
"""Merge sharded evaluate_generator_extension outputs into a new suffixed pair.
Canonical tables are never overwritten. Recomputes the summary from merged
by-seed rows and rejects duplicate cells across shards.
"""
import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import lib_common as lc
from evaluate_generator_extension import summarize


SAFE_SUFFIX = re.compile(r"_[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
KEY = ["family", "setting", "seed", "rung", "subset"]


def publish_no_clobber(staged_to_final: list[tuple[Path, Path]]) -> None:
    """Publish each staged file atomically and roll back on a collision."""
    published: list[Path] = []
    try:
        for staged, final in staged_to_final:
            final.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged, final)
            published.append(final)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suffixes", nargs="+",
                        help="input by-seed suffixes, each beginning with '_'")
    parser.add_argument("--output-suffix", required=True,
                        help="new merged suffix beginning with '_'")
    args = parser.parse_args()
    suffixes = args.suffixes
    if any(not SAFE_SUFFIX.fullmatch(s) for s in suffixes):
        parser.error(
            "every input suffix must match _[A-Za-z0-9][A-Za-z0-9_.-]* "
            "(path separators are forbidden)"
        )
    if len(suffixes) != len(set(suffixes)):
        parser.error("input suffixes must be unique")
    if not SAFE_SUFFIX.fullmatch(args.output_suffix):
        parser.error(
            "--output-suffix must match _[A-Za-z0-9][A-Za-z0-9_.-]* "
            "(path separators are forbidden)"
        )
    out_by = lc.TABLES / f"generator_extension_by_seed{args.output_suffix}.csv"
    out_summary = lc.TABLES / f"generator_extension_summary{args.output_suffix}.csv"
    if out_by.exists() or out_summary.exists():
        parser.error(f"refusing to overwrite merged output: {out_by} / {out_summary}")
    inputs = [lc.TABLES / f"generator_extension_by_seed{s}.csv" for s in suffixes]
    missing_inputs = [path for path in inputs if not path.is_file()]
    if missing_inputs:
        parser.error("missing input shard(s): " + ", ".join(str(path) for path in missing_inputs))
    parts = []
    for path in inputs:
        part = pd.read_csv(path)
        missing_columns = [column for column in KEY if column not in part.columns]
        if missing_columns:
            parser.error(f"input shard {path} misses key columns {missing_columns}")
        if part.empty:
            parser.error(f"input shard is empty: {path}")
        parts.append(part)
    merged = pd.concat(parts, ignore_index=True)
    dup = merged.duplicated(KEY, keep=False)
    if dup.any():
        rows = merged.loc[dup, KEY].drop_duplicates().to_dict("records")
        parser.error(f"duplicate cells across shards: {rows[:5]}")
    rows = merged.to_dict("records")
    summary_rows = summarize(rows)
    if not summary_rows:
        parser.error("merged summary is empty")

    lc.TABLES.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".generator-merge-publish-", dir=lc.TABLES))
    try:
        staged_by = staging / out_by.name
        staged_summary = staging / out_summary.name
        merged.to_csv(staged_by, index=False)
        lc.write_csv(staged_summary, summary_rows)
        if len(pd.read_csv(staged_by)) != len(merged):
            raise RuntimeError("staged merged table row-count mismatch")
        if len(pd.read_csv(staged_summary)) != len(summary_rows):
            raise RuntimeError("staged merged summary row-count mismatch")
        publish_no_clobber([
            (staged_by, out_by),
            (staged_summary, out_summary),
        ])
    except FileExistsError as exc:
        parser.error(f"refusing concurrent overwrite while publishing: {exc}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"merged {len(parts)} shards -> {len(merged)} rows at {args.output_suffix}")


if __name__ == "__main__":
    main()
