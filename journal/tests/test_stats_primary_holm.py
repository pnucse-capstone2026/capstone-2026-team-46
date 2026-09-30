from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import stats_primary_holm as sph  # noqa: E402


def test_holm_adjust_matches_step_down_example():
    got = sph.holm_adjust(np.asarray([0.01, 0.04, 0.03]))
    assert got == pytest.approx([0.03, 0.06, 0.06])


def test_build_table_keeps_only_focal_rows_and_reports_family():
    source = pd.DataFrame(
        [
            {
                "comparison": "a",
                "analysis_role": "focal_primary",
                "delta_mean": 1.0,
                "delta_std": 0.2,
                "n_seeds": 5,
            },
            {
                "comparison": "b",
                "analysis_role": "descriptive",
                "delta_mean": 0.0,
                "delta_std": 1.0,
                "n_seeds": 5,
            },
        ]
    )
    result = sph.build_table(source)
    assert result["comparison"].tolist() == ["a"]
    assert bool(result.loc[0, "holm_reject_alpha_0p05"])
    assert result.loc[0, "multiplicity_family"].startswith("six_declared")
