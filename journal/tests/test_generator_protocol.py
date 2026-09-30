from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_generator_separability import build_purged_real_partitions  # noqa: E402
from generator_protocol import (enforce_protocol_constraints,  # noqa: E402
                                protocol_repair_statistics,
                                summarize_condition_metadata,
                                summarize_protocol_validity)


def test_repair_enforces_ranges_and_dlc_payload_cross_constraint():
    raw = np.zeros((2, 2, 11), dtype=np.float64)
    raw[0, 0] = [100.4, 2.2, 1, 2, 3, 4, 5, 6, 7, 8, -0.1]
    raw[0, 1] = [3000, -2, -1, 1.4, 256, 9, 8, 7, 6, 5, 0.25]
    raw[1, 0] = [-4, 8.4, 255.4, 254.6, 3, 4, 5, 6, 7, 8, 0.5]
    raw[1, 1] = [50, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1.5]

    repaired = enforce_protocol_constraints(raw)
    assert repaired.dtype == np.float32
    np.testing.assert_array_equal(repaired[0, 0, 2:10], [1, 2, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(repaired[0, 1, 2:10], np.zeros(8))
    np.testing.assert_array_equal(repaired[1, 1, 2:10], np.zeros(8))
    assert repaired[0, 1, 0] == 2047
    assert repaired[0, 1, 1] == 0
    assert repaired[0, 0, 10] == 0

    validity = summarize_protocol_validity(
        repaired, reference_delta_t_range=(0.0, 1.0), chunk_windows=1)
    assert validity["protocol_valid"] is True
    assert validity["nonzero_padding_slots"] == 0
    assert validity["padding_zero_frac"] == 1.0
    assert validity["delta_t_above_reference_max"] == 1
    assert validity["protocol_valid_frame_frac"] == 1.0

    changes = protocol_repair_statistics(raw, repaired)
    assert changes["nonzero_padding_slots_zeroed"] > 0
    assert changes["negative_delta_t_clipped"] == 1


def test_invalid_padding_is_a_protocol_violation_and_nonfinite_is_rejected():
    x = np.zeros((1, 1, 11), dtype=np.float32)
    x[0, 0, 1] = 1
    x[0, 0, 3] = 7  # data1 is outside DLC=1
    validity = summarize_protocol_validity(x)
    assert validity["protocol_valid"] is False
    assert validity["nonzero_padding_slots"] == 1
    assert validity["frames_with_nonzero_padding"] == 1

    x[0, 0, 10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        enforce_protocol_constraints(x)


def test_condition_metadata_checks_requested_condition_alignment():
    y = np.asarray([1, 2, 3, 4], dtype=np.int8)
    names = np.asarray(["DoS", "Fuzzy", "Gear", "RPM"], dtype=object)
    good = summarize_condition_metadata(
        np.ones(4, dtype=np.int8), y, names,
        condition_label=y.copy(), condition_name=names.copy())
    assert good["condition_metadata_valid"] is True
    assert good["condition_metadata_valid_frac"] == 1.0

    wrong_names = names.copy()
    wrong_names[2] = "RPM"
    bad = summarize_condition_metadata(np.ones(4), y, wrong_names, condition_label=y)
    assert bad["condition_metadata_valid"] is False
    assert bad["synthetic_type_name_match_frac"] == 0.75


class _FakeNPZ(dict):
    @property
    def files(self):
        return list(self)


def test_group_split_discards_boundary_crossers_and_has_disjoint_blocks():
    labels, sources, segments, starts, ends = [], [], [], [], []
    for cls in range(1, 5):
        for block in range(12):
            labels.append(cls)
            sources.append(f"class{cls}.csv")
            segments.append(0)
            starts.append(block * 128)
            ends.append(block * 128 + 127)
        # This window spans two blocks and must be discarded.
        labels.append(cls)
        sources.append(f"class{cls}.csv")
        segments.append(0)
        starts.append(64)
        ends.append(191)
    data = _FakeNPZ({
        "y_attack_type": np.asarray(labels),
        "source_file": np.asarray(sources, dtype=object),
        "segment_id": np.asarray(segments),
        "start_index": np.asarray(starts),
        "end_index": np.asarray(ends),
    })
    parts, meta = build_purged_real_partitions(data, block_frames=128, seed=42)
    assert meta["overlap_purged"] is True
    for cls in range(1, 5):
        train_blocks = set((data["start_index"][parts[cls]["train"]] // 128).tolist())
        test_blocks = set((data["start_index"][parts[cls]["test"]] // 128).tolist())
        assert train_blocks.isdisjoint(test_blocks)
        row = next(r for r in meta["class_rows"] if r["attack_type_id"] == cls)
        assert row["discarded_cross_block"] == 1
