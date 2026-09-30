#!/usr/bin/env python3
"""CAN-frame protocol constraints shared by learned-generator samplers/audits.

The detector representation has eleven channels per frame::

    can_id, dlc, data0..data7, delta_t

This module deliberately separates two notions:

* ``protocol_valid`` checks finite values, integer/range constraints, a
  non-negative inter-arrival time, and the DLC/payload cross-field constraint
  (bytes at positions ``>= DLC`` are zero).
* source-support timing checks report values outside a supplied reference
  range.  They are diagnostics, not CAN protocol violations: CAN itself does
  not impose a dataset-independent maximum inter-arrival time.

The functions operate in chunks so a full 260k x 128 pool can be audited
without allocating another pool-sized boolean tensor.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ID_MIN = 0
ID_MAX = 0x7FF
DLC_MIN = 0
DLC_MAX = 8
BYTE_MIN = 0
BYTE_MAX = 255
N_PAYLOAD_BYTES = 8
EXPECTED_CLASS_NAMES = {1: "DoS", 2: "Fuzzy", 3: "Gear", 4: "RPM"}


def _require_window_shape(x: np.ndarray) -> None:
    if not isinstance(x, np.ndarray):
        raise TypeError("windows must be a numpy array")
    if x.ndim != 3 or x.shape[2] != 11:
        raise ValueError(f"expected windows shaped (N,T,11), got {x.shape}")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("windows must contain at least one frame")


def enforce_protocol_constraints(
    x_raw: np.ndarray,
    *,
    delta_t_max: float | None = None,
) -> np.ndarray:
    """Return a repaired float32 copy satisfying the representation protocol.

    ID, DLC, and payload channels are rounded and clipped to their legal
    integer ranges.  ``delta_t`` is clipped to ``[0, delta_t_max]`` when an
    upper bound is explicitly supplied, otherwise only its physical lower
    bound is enforced.  Payload slots at positions ``>= DLC`` are zeroed.

    Non-finite model output is rejected rather than silently imputed.
    """
    x_raw = np.asarray(x_raw)
    _require_window_shape(x_raw)
    if not np.isfinite(x_raw).all():
        where = np.argwhere(~np.isfinite(x_raw))[0].tolist()
        raise ValueError(f"non-finite generator output at index {where}")
    if delta_t_max is not None and (not np.isfinite(delta_t_max) or delta_t_max < 0):
        raise ValueError("delta_t_max must be finite and non-negative")

    x = x_raw.astype(np.float32, copy=True)
    x[:, :, 0] = np.clip(np.rint(x[:, :, 0]), ID_MIN, ID_MAX)
    x[:, :, 1] = np.clip(np.rint(x[:, :, 1]), DLC_MIN, DLC_MAX)
    x[:, :, 2:10] = np.clip(np.rint(x[:, :, 2:10]), BYTE_MIN, BYTE_MAX)
    x[:, :, 10] = np.clip(x[:, :, 10], 0.0, delta_t_max)

    # Enforce the cross-field constraint without constructing an (N,T,8)
    # mask the size of the whole pool.
    dlc = x[:, :, 1].astype(np.int8, copy=False)
    for byte_index in range(N_PAYLOAD_BYTES):
        channel = 2 + byte_index
        x[:, :, channel][dlc <= byte_index] = 0.0
    return x


def protocol_repair_statistics(
    raw: np.ndarray,
    repaired: np.ndarray,
    *,
    chunk_windows: int = 4096,
) -> dict[str, float | int]:
    """Summarize exactly what :func:`enforce_protocol_constraints` changed."""
    raw = np.asarray(raw)
    repaired = np.asarray(repaired)
    _require_window_shape(raw)
    _require_window_shape(repaired)
    if raw.shape != repaired.shape:
        raise ValueError(f"raw/repaired shape mismatch: {raw.shape} vs {repaired.shape}")

    if chunk_windows <= 0:
        raise ValueError("chunk_windows must be positive")
    raw_nonfinite = integer_changed = padding_zeroed = padding_slots = 0
    negative_dt = all_changed = 0
    for start in range(0, len(raw), chunk_windows):
        r = raw[start:start + chunk_windows]
        p = repaired[start:start + chunk_windows]
        changed = ~np.isclose(r, p, rtol=0.0, atol=1e-6, equal_nan=True)
        raw_nonfinite += int((~np.isfinite(r)).sum())
        integer_changed += int(changed[:, :, :10].sum())
        negative_dt += int((r[:, :, 10] < 0).sum())
        all_changed += int(changed.sum())
        dlc_repaired = p[:, :, 1].astype(np.int8, copy=False)
        for byte_index in range(N_PAYLOAD_BYTES):
            pad = dlc_repaired <= byte_index
            padding_slots += int(pad.sum())
            padding_zeroed += int((pad & (r[:, :, 2 + byte_index] != 0)).sum())
    integer_values = raw.shape[0] * raw.shape[1] * 10
    all_values = raw.size
    return {
        "raw_nonfinite_values": raw_nonfinite,
        "integer_channel_values_changed": integer_changed,
        "integer_channel_alter_frac": float(integer_changed / integer_values),
        "negative_delta_t_clipped": negative_dt,
        "padding_slots": padding_slots,
        "nonzero_padding_slots_zeroed": padding_zeroed,
        "all_values_changed": all_changed,
        "all_values_alter_frac": float(all_changed / all_values),
    }


def summarize_protocol_validity(
    x: np.ndarray,
    *,
    reference_delta_t_range: tuple[float, float] | None = None,
    chunk_windows: int = 4096,
) -> dict[str, float | int | bool]:
    """Audit physical and cross-field validity of a window array.

    ``reference_delta_t_range`` adds source-support diagnostics.  Values
    outside it do not change ``protocol_valid``.
    """
    x = np.asarray(x)
    _require_window_shape(x)
    if chunk_windows <= 0:
        raise ValueError("chunk_windows must be positive")
    if reference_delta_t_range is not None:
        ref_lo, ref_hi = map(float, reference_delta_t_range)
        if not np.isfinite([ref_lo, ref_hi]).all() or ref_lo > ref_hi:
            raise ValueError("invalid reference_delta_t_range")
    else:
        ref_lo = ref_hi = None

    counters = {
        "nonfinite_can_id": 0,
        "nonfinite_dlc": 0,
        "nonfinite_payload": 0,
        "nonfinite_delta_t": 0,
        "noninteger_can_id": 0,
        "noninteger_dlc": 0,
        "noninteger_payload": 0,
        "out_of_range_can_id": 0,
        "out_of_range_dlc": 0,
        "out_of_range_payload": 0,
        "negative_delta_t": 0,
        "padding_slots": 0,
        "nonzero_padding_slots": 0,
        "frames_with_nonzero_padding": 0,
        "invalid_frames": 0,
        "delta_t_below_reference_min": 0,
        "delta_t_above_reference_max": 0,
    }
    minima = np.full(11, np.inf, dtype=np.float64)
    maxima = np.full(11, -np.inf, dtype=np.float64)

    for start in range(0, len(x), chunk_windows):
        z = x[start:start + chunk_windows]
        finite = np.isfinite(z)
        for ch in range(11):
            vals = z[:, :, ch]
            good = finite[:, :, ch]
            if good.any():
                minima[ch] = min(minima[ch], float(vals[good].min()))
                maxima[ch] = max(maxima[ch], float(vals[good].max()))

        ids, dlc, payload, dt = z[:, :, 0], z[:, :, 1], z[:, :, 2:10], z[:, :, 10]
        fid, fdlc = finite[:, :, 0], finite[:, :, 1]
        fpayload, fdt = finite[:, :, 2:10], finite[:, :, 10]
        id_integer = fid & (np.abs(ids - np.rint(ids)) <= 1e-6)
        dlc_integer = fdlc & (np.abs(dlc - np.rint(dlc)) <= 1e-6)
        payload_integer = fpayload & (np.abs(payload - np.rint(payload)) <= 1e-6)
        id_range = fid & (ids >= ID_MIN) & (ids <= ID_MAX)
        dlc_range = fdlc & (dlc >= DLC_MIN) & (dlc <= DLC_MAX)
        payload_range = fpayload & (payload >= BYTE_MIN) & (payload <= BYTE_MAX)

        counters["nonfinite_can_id"] += int((~fid).sum())
        counters["nonfinite_dlc"] += int((~fdlc).sum())
        counters["nonfinite_payload"] += int((~fpayload).sum())
        counters["nonfinite_delta_t"] += int((~fdt).sum())
        counters["noninteger_can_id"] += int((fid & ~id_integer).sum())
        counters["noninteger_dlc"] += int((fdlc & ~dlc_integer).sum())
        counters["noninteger_payload"] += int((fpayload & ~payload_integer).sum())
        counters["out_of_range_can_id"] += int((fid & ~id_range).sum())
        counters["out_of_range_dlc"] += int((fdlc & ~dlc_range).sum())
        counters["out_of_range_payload"] += int((fpayload & ~payload_range).sum())
        counters["negative_delta_t"] += int((fdt & (dt < 0)).sum())

        valid_dlc = dlc_integer & dlc_range
        dlc_int = np.clip(np.rint(np.where(fdlc, dlc, 0)), DLC_MIN, DLC_MAX).astype(np.int8)
        padding_bad_frame = np.zeros(dlc.shape, dtype=bool)
        for byte_index in range(N_PAYLOAD_BYTES):
            pad = valid_dlc & (dlc_int <= byte_index)
            bad = pad & ((payload[:, :, byte_index] != 0) | ~fpayload[:, :, byte_index])
            counters["padding_slots"] += int(pad.sum())
            counters["nonzero_padding_slots"] += int(bad.sum())
            padding_bad_frame |= bad
        counters["frames_with_nonzero_padding"] += int(padding_bad_frame.sum())

        frame_bad = (
            ~id_integer | ~id_range | ~dlc_integer | ~dlc_range
            | ~(payload_integer & payload_range).all(axis=2)
            | ~fdt | (dt < 0) | padding_bad_frame
        )
        counters["invalid_frames"] += int(frame_bad.sum())
        if reference_delta_t_range is not None:
            counters["delta_t_below_reference_min"] += int((fdt & (dt < ref_lo)).sum())
            counters["delta_t_above_reference_max"] += int((fdt & (dt > ref_hi)).sum())

    total_frames = int(x.shape[0] * x.shape[1])
    total_payload = total_frames * N_PAYLOAD_BYTES
    invalid_values = (
        counters["nonfinite_can_id"] + counters["nonfinite_dlc"]
        + counters["nonfinite_payload"] + counters["nonfinite_delta_t"]
        + counters["noninteger_can_id"] + counters["noninteger_dlc"]
        + counters["noninteger_payload"] + counters["out_of_range_can_id"]
        + counters["out_of_range_dlc"] + counters["out_of_range_payload"]
        + counters["negative_delta_t"] + counters["nonzero_padding_slots"]
    )
    result: dict[str, float | int | bool] = {
        "windows": int(len(x)),
        "frames": total_frames,
        "can_id_min": float(minima[0]),
        "can_id_max": float(maxima[0]),
        "dlc_min": float(minima[1]),
        "dlc_max": float(maxima[1]),
        "payload_min": float(np.min(minima[2:10])),
        "payload_max": float(np.max(maxima[2:10])),
        "delta_t_min": float(minima[10]),
        "delta_t_max": float(maxima[10]),
        **counters,
        "padding_zero_frac": (
            1.0 - counters["nonzero_padding_slots"] / counters["padding_slots"]
            if counters["padding_slots"] else 1.0
        ),
        "protocol_valid_frame_frac": 1.0 - counters["invalid_frames"] / total_frames,
        "protocol_valid": bool(invalid_values == 0),
        "total_payload_slots": total_payload,
    }
    if reference_delta_t_range is not None:
        result["reference_delta_t_min"] = float(ref_lo)
        result["reference_delta_t_max"] = float(ref_hi)
        finite_dt = total_frames - counters["nonfinite_delta_t"]
        outside_dt = (counters["delta_t_below_reference_min"]
                      + counters["delta_t_above_reference_max"])
        result["delta_t_in_reference_range_frac"] = (
            1.0 - outside_dt / finite_dt if finite_dt else 0.0)
    return result


def summarize_condition_metadata(
    y_binary: np.ndarray,
    y_attack_type: np.ndarray,
    synthetic_type: np.ndarray,
    *,
    condition_label: np.ndarray | None = None,
    condition_name: np.ndarray | None = None,
    expected_class_names: Mapping[int, str] = EXPECTED_CLASS_NAMES,
) -> dict[str, float | int | bool]:
    """Check that saved labels faithfully record the requested condition.

    This is a metadata audit, not evidence that the generated content is
    semantically recognizable as the condition.  The latter is reported as a
    source-classifier proxy by ``audit_generator_separability.py``.
    """
    y_binary = np.asarray(y_binary).reshape(-1)
    y_attack_type = np.asarray(y_attack_type).reshape(-1)
    synthetic_type = np.asarray(synthetic_type).astype(str).reshape(-1)
    n = len(y_attack_type)
    if len(y_binary) != n or len(synthetic_type) != n:
        raise ValueError("condition metadata arrays have different lengths")
    expected_names = np.asarray([expected_class_names.get(int(y), "") for y in y_attack_type])
    valid_labels = np.isin(y_attack_type, list(expected_class_names))
    name_match = valid_labels & (synthetic_type == expected_names)
    if condition_label is None:
        label_match = np.ones(n, dtype=bool)
        has_condition_label = False
    else:
        condition_label = np.asarray(condition_label).reshape(-1)
        if len(condition_label) != n:
            raise ValueError("condition_label length mismatch")
        label_match = condition_label == y_attack_type
        has_condition_label = True
    if condition_name is None:
        condition_name_match = np.ones(n, dtype=bool)
        has_condition_name = False
    else:
        condition_name = np.asarray(condition_name).astype(str).reshape(-1)
        if len(condition_name) != n:
            raise ValueError("condition_name length mismatch")
        condition_name_match = condition_name == synthetic_type
        has_condition_name = True

    all_match = (y_binary == 1) & valid_labels & name_match & label_match & condition_name_match
    return {
        "windows": n,
        "binary_attack_label_match_frac": float((y_binary == 1).mean()),
        "valid_condition_label_frac": float(valid_labels.mean()),
        "synthetic_type_name_match_frac": float(name_match.mean()),
        "has_explicit_condition_label": has_condition_label,
        "condition_label_match_frac": float(label_match.mean()),
        "has_explicit_condition_name": has_condition_name,
        "condition_name_match_frac": float(condition_name_match.mean()),
        "condition_metadata_valid_frac": float(all_match.mean()),
        "condition_metadata_valid": bool(all_match.all()),
    }
