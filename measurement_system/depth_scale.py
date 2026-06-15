from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_scale_csv(path: Path) -> dict[int, dict[str, float | bool]]:
    scales: dict[int, dict[str, float | bool]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row["frame_id"])
            scale = float(row["scale_m_per_depth"])
            scale_valid_raw = str(row.get("scale_valid", "True")).lower()
            scale_valid = scale_valid_raw in {"true", "1", "yes"} and np.isfinite(scale) and scale > 0
            scales[frame_id] = {
                "scale_m_per_depth": scale,
                "scale_valid": scale_valid,
                "scale_disagreement": float(row.get("scale_disagreement", "nan")),
            }
    return scales


def smooth_scales(
    scales: dict[int, dict[str, float | bool]],
    window: int,
) -> dict[int, dict[str, float | bool]]:
    if window <= 1:
        return scales
    if window % 2 == 0:
        window += 1
    frame_ids = sorted(scales)
    values = np.array([float(scales[frame_id]["scale_m_per_depth"]) for frame_id in frame_ids], dtype=np.float32)
    valid = np.array([bool(scales[frame_id]["scale_valid"]) for frame_id in frame_ids], dtype=bool)
    smoothed = values.copy()
    radius = window // 2
    for idx in range(len(frame_ids)):
        lo = max(0, idx - radius)
        hi = min(len(frame_ids), idx + radius + 1)
        use = valid[lo:hi] & np.isfinite(values[lo:hi]) & (values[lo:hi] > 0)
        if np.any(use):
            smoothed[idx] = float(np.median(values[lo:hi][use]))
    output: dict[int, dict[str, float | bool]] = {}
    for idx, frame_id in enumerate(frame_ids):
        output[frame_id] = dict(scales[frame_id])
        output[frame_id]["scale_m_per_depth"] = float(smoothed[idx])
    return output

