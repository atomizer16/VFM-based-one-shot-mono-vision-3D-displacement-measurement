from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CameraModel, load_camera, save_json
from .export import read_depth, read_mask, write_csv
from .frame_manifest import build_manifest


@dataclass(frozen=True)
class MarkerScale:
    label_id: int
    marker_id: int
    z_m: float
    relative_depth: float
    scale: float


def aruco_dictionary(name: str):
    import cv2

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module is unavailable. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown aruco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_markers(image: np.ndarray, dictionary_name: str):
    import cv2

    dictionary = aruco_dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _rejected = detector.detectMarkers(image)
    if ids is None:
        return [], np.empty((0,), dtype=np.int32)
    return corners, ids.reshape(-1).astype(np.int32)


def marker_object_points(marker_length_m: float) -> np.ndarray:
    half = float(marker_length_m) / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )


def label_for_marker(mask: np.ndarray, corners: np.ndarray, aruco_label_ids: set[int]) -> int | None:
    import cv2

    polygon_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(polygon_mask, corners.reshape(4, 2).astype(np.int32), 1)
    labels, counts = np.unique(mask[polygon_mask.astype(bool)], return_counts=True)
    best_label = None
    best_count = 0
    for label, count in zip(labels.tolist(), counts.tolist()):
        if int(label) in aruco_label_ids and int(count) > best_count:
            best_label = int(label)
            best_count = int(count)
    if best_label is not None:
        return best_label

    cx, cy = np.mean(corners.reshape(4, 2), axis=0)
    x = int(np.clip(round(cx), 0, mask.shape[1] - 1))
    y = int(np.clip(round(cy), 0, mask.shape[0] - 1))
    center_label = int(mask[y, x])
    return center_label if center_label in aruco_label_ids else None


def median_depth_for_label(depth: np.ndarray, mask: np.ndarray, label_id: int) -> float:
    values = depth[(mask == label_id) & np.isfinite(depth) & (depth > 0)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values.astype(np.float32)))


def estimate_frame_scale(
    frame_path: Path,
    mask_path: Path,
    depth_path: Path,
    camera: CameraModel,
    marker_length_m: float,
    dictionary_name: str,
    aruco_label_ids: tuple[int, int],
) -> list[MarkerScale]:
    import cv2

    image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read frame: {frame_path}")
    mask = read_mask(mask_path)
    depth = read_depth(depth_path)
    corners_list, marker_ids = detect_markers(image, dictionary_name)
    object_points = marker_object_points(marker_length_m)

    results: list[MarkerScale] = []
    for corners, marker_id in zip(corners_list, marker_ids.tolist()):
        label_id = label_for_marker(mask, corners, set(aruco_label_ids))
        if label_id is None:
            continue
        ok, _rvec, tvec = cv2.solvePnP(object_points, corners.reshape(4, 2).astype(np.float32), camera.K, camera.dist)
        if not ok:
            continue
        z_m = float(tvec.reshape(3)[2])
        relative_depth = median_depth_for_label(depth, mask, label_id)
        scale = z_m / relative_depth if np.isfinite(relative_depth) and relative_depth > 0 else float("nan")
        results.append(MarkerScale(label_id, int(marker_id), z_m, relative_depth, scale))
    return results


def rows_for_scale(
    frame_ids: list[int],
    frames_dir: Path,
    masks_dir: Path,
    depth_dir: Path,
    camera: CameraModel,
    marker_length_m: float,
    dictionary_name: str,
    aruco_label_ids: tuple[int, int],
    max_scale_disagreement: float,
) -> list[dict]:
    manifest = build_manifest(min(frame_ids), max(frame_ids), frames_dir, masks_dir, depth_dir)
    wanted = set(frame_ids)
    rows: list[dict] = []
    for record in manifest:
        if record.frame_id not in wanted:
            continue
        marker_scales: list[MarkerScale] = []
        if record.frame_path and record.mask_path and record.depth_path:
            marker_scales = estimate_frame_scale(
                record.frame_path,
                record.mask_path,
                record.depth_path,
                camera,
                marker_length_m,
                dictionary_name,
                aruco_label_ids,
            )
        scale_values = np.array([item.scale for item in marker_scales if np.isfinite(item.scale) and item.scale > 0])
        scale = float(np.median(scale_values)) if scale_values.size else float("nan")
        disagreement = float("nan")
        if scale_values.size >= 2:
            disagreement = float((np.max(scale_values) - np.min(scale_values)) / max(1e-9, np.median(scale_values)))
        row = {
            "frame_id": record.frame_id,
            "scale_m_per_depth": scale,
            "scale_valid": bool(np.isfinite(scale) and (not np.isfinite(disagreement) or disagreement <= max_scale_disagreement)),
            "scale_disagreement": disagreement,
            "num_markers": len(marker_scales),
        }
        for label_id in aruco_label_ids:
            match = next((item for item in marker_scales if item.label_id == label_id), None)
            prefix = f"label_{label_id}"
            row[f"{prefix}_marker_id"] = match.marker_id if match else ""
            row[f"{prefix}_z_m"] = match.z_m if match else float("nan")
            row[f"{prefix}_relative_depth"] = match.relative_depth if match else float("nan")
            row[f"{prefix}_scale"] = match.scale if match else float("nan")
        rows.append(row)
    return rows


def read_frame_ids_from_tracks(track_npz: Path | None, start_frame: int | None, end_frame: int | None) -> list[int]:
    if track_npz is not None and track_npz.exists():
        with np.load(track_npz, allow_pickle=True) as data:
            return [int(x) for x in data["frame_ids"].tolist()]
    if start_frame is None or end_frame is None:
        raise ValueError("Provide --track-npz or both --start-frame and --end-frame.")
    return list(range(start_frame, end_frame + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate per-frame metric scale from two ArUco markers and DA3 depth.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True, help="SAM combined_instance mask directory.")
    parser.add_argument("--depth-dir", type=Path, required=True, help="DA3 raw_depth directory.")
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--track-npz", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--marker-length-m", type=float, default=0.40, help="Actual ArUco code side length, not outer board size.")
    parser.add_argument("--aruco-dictionary", default="DICT_4X4_50")
    parser.add_argument("--aruco-label-ids", default="2,3")
    parser.add_argument("--max-scale-disagreement", type=float, default=0.10)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = load_camera(args.camera_json)
    label_ids = tuple(int(x) for x in args.aruco_label_ids.split(","))
    if len(label_ids) != 2:
        raise ValueError("--aruco-label-ids must contain exactly two labels, e.g. 2,3.")
    frame_ids = read_frame_ids_from_tracks(args.track_npz, args.start_frame, args.end_frame)
    rows = rows_for_scale(
        frame_ids,
        args.frames_dir,
        args.masks_dir,
        args.depth_dir,
        camera,
        args.marker_length_m,
        args.aruco_dictionary,
        label_ids,
        args.max_scale_disagreement,
    )
    fieldnames = list(rows[0].keys()) if rows else ["frame_id", "scale_m_per_depth", "scale_valid"]
    write_csv(args.output_csv, fieldnames, rows)
    valid_count = sum(bool(row["scale_valid"]) for row in rows)
    save_json(
        {
            "frames": len(rows),
            "valid_scale_frames": valid_count,
            "marker_length_m": args.marker_length_m,
            "aruco_label_ids": list(label_ids),
            "output_csv": str(args.output_csv),
        },
        args.output_csv.with_suffix(".summary.json"),
    )
    print(f"Saved ArUco metric scale: {args.output_csv}")
    print(f"Valid scale frames: {valid_count}/{len(rows)}")


if __name__ == "__main__":
    main()
