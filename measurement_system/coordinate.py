from __future__ import annotations

import numpy as np

from .config import CameraModel


def backproject_pixel(u: float, v: float, z_m: float, camera: CameraModel) -> tuple[float, float, float]:
    x = (float(u) - camera.cx) * float(z_m) / camera.fx
    y = (float(v) - camera.cy) * float(z_m) / camera.fy
    return x, y, float(z_m)


def sample_depth_median(
    depth: np.ndarray,
    u: float,
    v: float,
    *,
    radius: int,
    mask: np.ndarray | None = None,
    mask_label: int | None = None,
) -> float:
    h, w = depth.shape[:2]
    x = int(round(float(u)))
    y = int(round(float(v)))
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return float("nan")

    patch = depth[y0:y1, x0:x1].astype(np.float32)
    valid = np.isfinite(patch) & (patch > 0)
    if mask is not None and mask_label is not None:
        valid &= mask[y0:y1, x0:x1] == mask_label
    values = patch[valid]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def displacement_from_reference(points_xyz: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    displacement = np.full_like(points_xyz, np.nan, dtype=np.float32)
    reference_xyz = np.full((points_xyz.shape[1], 3), np.nan, dtype=np.float32)
    reference_valid = np.zeros(points_xyz.shape[1], dtype=bool)
    for track_id in range(points_xyz.shape[1]):
        valid_indices = np.flatnonzero(valid[:, track_id] & np.all(np.isfinite(points_xyz[:, track_id]), axis=1))
        if valid_indices.size == 0:
            continue
        ref = points_xyz[valid_indices[0], track_id]
        reference_xyz[track_id] = ref
        reference_valid[track_id] = True
        displacement[:, track_id] = points_xyz[:, track_id] - ref
    return displacement, reference_valid

