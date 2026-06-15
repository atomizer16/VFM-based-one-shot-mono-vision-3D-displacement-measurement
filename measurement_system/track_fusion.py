from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import load_camera, save_json
from .coordinate import backproject_pixel, displacement_from_reference, sample_depth_median
from .depth_scale import load_scale_csv, smooth_scales
from .export import read_depth, read_mask, write_csv
from .frame_manifest import build_manifest


@dataclass(frozen=True)
class Tracks2D:
    frame_ids: np.ndarray
    tracks_xy: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    track_ids: np.ndarray


def load_tracks_npz(path: Path) -> Tracks2D:
    with np.load(path, allow_pickle=True) as data:
        tracks_xy = data["tracks_xy"].astype(np.float32)
        valid = data["valid"].astype(bool)
        confidence = data["confidence"].astype(np.float32)
        frame_ids = data["frame_ids"].astype(np.int32)
        track_ids = np.arange(tracks_xy.shape[1], dtype=np.int32)
    return Tracks2D(frame_ids, tracks_xy, valid, confidence, track_ids)


def load_tracks_from_dirs(track_root: Path) -> Tracks2D:
    point_dirs = sorted(path for path in track_root.glob("point_*") if path.is_dir())
    if not point_dirs:
        raise RuntimeError(f"No point_xxxx folders found in: {track_root}")
    parsed: list[list[dict]] = []
    frame_ids: list[int] | None = None
    for point_dir in point_dirs:
        track_file = point_dir / "track.txt"
        rows: list[dict] = []
        with track_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                frame_id, x, y, valid, confidence, *_source = line.split()
                rows.append(
                    {
                        "frame_id": int(frame_id),
                        "x": float(x),
                        "y": float(y),
                        "valid": bool(int(valid)),
                        "confidence": float(confidence),
                    }
                )
        current_frame_ids = [row["frame_id"] for row in rows]
        if frame_ids is None:
            frame_ids = current_frame_ids
        elif frame_ids != current_frame_ids:
            raise RuntimeError(f"Frame IDs differ in {track_file}")
        parsed.append(rows)

    assert frame_ids is not None
    num_frames = len(frame_ids)
    num_tracks = len(parsed)
    tracks_xy = np.full((num_frames, num_tracks, 2), np.nan, dtype=np.float32)
    valid = np.zeros((num_frames, num_tracks), dtype=bool)
    confidence = np.zeros((num_frames, num_tracks), dtype=np.float32)
    track_ids = np.zeros(num_tracks, dtype=np.int32)
    for track_idx, rows in enumerate(parsed):
        digits = "".join(ch for ch in point_dirs[track_idx].name if ch.isdigit())
        track_ids[track_idx] = int(digits) if digits else track_idx
        for frame_idx, row in enumerate(rows):
            tracks_xy[frame_idx, track_idx] = (row["x"], row["y"])
            valid[frame_idx, track_idx] = row["valid"]
            confidence[frame_idx, track_idx] = row["confidence"]
    return Tracks2D(np.asarray(frame_ids, dtype=np.int32), tracks_xy, valid, confidence, track_ids)


def load_tracks(track_npz: Path | None, track_root: Path | None) -> Tracks2D:
    if track_npz is not None and track_npz.exists():
        return load_tracks_npz(track_npz)
    if track_root is not None:
        return load_tracks_from_dirs(track_root)
    raise ValueError("Provide --track-npz or --track-root.")


def fuse_tracks_to_3d(
    tracks: Tracks2D,
    frames_dir: Path | None,
    masks_dir: Path,
    depth_dir: Path,
    camera_json: Path,
    scale_csv: Path,
    output_dir: Path,
    *,
    object_label_id: int,
    depth_radius: int,
    scale_smooth_window: int,
) -> dict:
    camera = load_camera(camera_json)
    scales = smooth_scales(load_scale_csv(scale_csv), scale_smooth_window)
    manifest = {
        record.frame_id: record
        for record in build_manifest(
            int(tracks.frame_ids[0]),
            int(tracks.frame_ids[-1]),
            frames_dir,
            masks_dir,
            depth_dir,
        )
    }
    num_frames, num_tracks = tracks.valid.shape
    points_xyz = np.full((num_frames, num_tracks, 3), np.nan, dtype=np.float32)
    relative_depths = np.full((num_frames, num_tracks), np.nan, dtype=np.float32)
    metric_depths = np.full((num_frames, num_tracks), np.nan, dtype=np.float32)
    fused_valid = np.zeros((num_frames, num_tracks), dtype=bool)

    for frame_idx, frame_id_raw in enumerate(tracks.frame_ids.tolist()):
        frame_id = int(frame_id_raw)
        record = manifest.get(frame_id)
        scale_record = scales.get(frame_id)
        if record is None or record.depth_path is None or record.mask_path is None or scale_record is None:
            continue
        scale = float(scale_record["scale_m_per_depth"])
        if not bool(scale_record["scale_valid"]) or not np.isfinite(scale) or scale <= 0:
            continue
        depth = read_depth(record.depth_path)
        mask = read_mask(record.mask_path)
        for track_idx in range(num_tracks):
            if not tracks.valid[frame_idx, track_idx]:
                continue
            u, v = tracks.tracks_xy[frame_idx, track_idx]
            rel_depth = sample_depth_median(depth, float(u), float(v), radius=depth_radius, mask=mask, mask_label=object_label_id)
            if not np.isfinite(rel_depth) or rel_depth <= 0:
                continue
            z_m = rel_depth * scale
            points_xyz[frame_idx, track_idx] = backproject_pixel(float(u), float(v), z_m, camera)
            relative_depths[frame_idx, track_idx] = rel_depth
            metric_depths[frame_idx, track_idx] = z_m
            fused_valid[frame_idx, track_idx] = True

    displacement, reference_valid = displacement_from_reference(points_xyz, fused_valid)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "tracks_3d.npz",
        frame_ids=tracks.frame_ids,
        track_ids=tracks.track_ids,
        tracks_xy=tracks.tracks_xy,
        points_xyz=points_xyz,
        displacement=displacement,
        relative_depths=relative_depths,
        metric_depths=metric_depths,
        valid=fused_valid,
        roma_valid=tracks.valid,
        confidence=tracks.confidence,
        reference_valid=reference_valid,
    )
    write_per_point_csvs(output_dir / "per_point", tracks, points_xyz, displacement, relative_depths, metric_depths, fused_valid)
    summary = {
        "frames": int(num_frames),
        "tracks": int(num_tracks),
        "valid_3d_observations": int(fused_valid.sum()),
        "tracks_with_reference": int(reference_valid.sum()),
        "object_label_id": object_label_id,
        "depth_radius": depth_radius,
        "tracks_3d_npz": str(output_dir / "tracks_3d.npz"),
    }
    save_json(summary, output_dir / "fusion_summary.json")
    return summary


def write_per_point_csvs(
    output_dir: Path,
    tracks: Tracks2D,
    points_xyz: np.ndarray,
    displacement: np.ndarray,
    relative_depths: np.ndarray,
    metric_depths: np.ndarray,
    valid: np.ndarray,
) -> None:
    fieldnames = [
        "frame_id",
        "u",
        "v",
        "relative_depth",
        "metric_depth_m",
        "X_m",
        "Y_m",
        "Z_m",
        "dX_m",
        "dY_m",
        "dZ_m",
        "valid",
        "confidence",
    ]
    for track_idx, track_id in enumerate(tracks.track_ids.tolist()):
        rows = []
        for frame_idx, frame_id in enumerate(tracks.frame_ids.tolist()):
            rows.append(
                {
                    "frame_id": int(frame_id),
                    "u": float(tracks.tracks_xy[frame_idx, track_idx, 0]),
                    "v": float(tracks.tracks_xy[frame_idx, track_idx, 1]),
                    "relative_depth": float(relative_depths[frame_idx, track_idx]),
                    "metric_depth_m": float(metric_depths[frame_idx, track_idx]),
                    "X_m": float(points_xyz[frame_idx, track_idx, 0]),
                    "Y_m": float(points_xyz[frame_idx, track_idx, 1]),
                    "Z_m": float(points_xyz[frame_idx, track_idx, 2]),
                    "dX_m": float(displacement[frame_idx, track_idx, 0]),
                    "dY_m": float(displacement[frame_idx, track_idx, 1]),
                    "dZ_m": float(displacement[frame_idx, track_idx, 2]),
                    "valid": int(valid[frame_idx, track_idx]),
                    "confidence": float(tracks.confidence[frame_idx, track_idx]),
                }
            )
        write_csv(output_dir / f"point_{int(track_id):04d}_3d.csv", fieldnames, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse RoMa 2D tracks, DA3 depth, ArUco scale, and camera intrinsics into 3D tracks.")
    parser.add_argument("--track-npz", type=Path, default=None)
    parser.add_argument("--track-root", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--scale-csv", type=Path, required=True)
    parser.add_argument("--object-label-id", type=int, default=1)
    parser.add_argument("--depth-radius", type=int, default=3)
    parser.add_argument("--scale-smooth-window", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks = load_tracks(args.track_npz, args.track_root)
    summary = fuse_tracks_to_3d(
        tracks,
        args.frames_dir,
        args.masks_dir,
        args.depth_dir,
        args.camera_json,
        args.scale_csv,
        args.output_dir,
        object_label_id=args.object_label_id,
        depth_radius=args.depth_radius,
        scale_smooth_window=args.scale_smooth_window,
    )
    print(f"Saved 3D tracks: {summary['tracks_3d_npz']}")
    print(f"Valid 3D observations: {summary['valid_3d_observations']}")


if __name__ == "__main__":
    main()

