from __future__ import annotations

import argparse
from pathlib import Path

from .aruco_metric import read_frame_ids_from_tracks, rows_for_scale
from .config import load_camera
from .export import write_csv
from .quality import select_best_tracks
from .track_fusion import fuse_tracks_to_3d, load_tracks
from .visualize import compute_structure_displacement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-model monocular 3D displacement pipeline.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True, help="SAM combined_instance mask directory.")
    parser.add_argument("--depth-dir", type=Path, required=True, help="DA3 raw_depth directory.")
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--track-npz", type=Path, default=None)
    parser.add_argument("--track-root", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--marker-length-m", type=float, default=0.40)
    parser.add_argument("--aruco-dictionary", default="DICT_4X4_50")
    parser.add_argument("--aruco-label-ids", default="2,3")
    parser.add_argument("--object-label-id", type=int, default=1)
    parser.add_argument("--depth-radius", type=int, default=3)
    parser.add_argument("--scale-smooth-window", type=int, default=5)
    parser.add_argument("--max-scale-disagreement", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--quality-smooth-window", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    camera = load_camera(args.camera_json)
    label_ids = tuple(int(x) for x in args.aruco_label_ids.split(","))
    if len(label_ids) != 2:
        raise ValueError("--aruco-label-ids must contain exactly two labels, e.g. 2,3.")

    frame_ids = read_frame_ids_from_tracks(args.track_npz, args.start_frame, args.end_frame)
    scale_csv = args.output_dir / "aruco_scale.csv"
    scale_rows = rows_for_scale(
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
    if not scale_rows:
        raise RuntimeError("No frames found for ArUco scale estimation.")
    write_csv(scale_csv, list(scale_rows[0].keys()), scale_rows)
    print(f"[1/4] Saved ArUco scale: {scale_csv}")

    tracks = load_tracks(args.track_npz, args.track_root)
    fusion_dir = args.output_dir / "fusion"
    fusion_summary = fuse_tracks_to_3d(
        tracks,
        args.frames_dir,
        args.masks_dir,
        args.depth_dir,
        args.camera_json,
        scale_csv,
        fusion_dir,
        object_label_id=args.object_label_id,
        depth_radius=args.depth_radius,
        scale_smooth_window=args.scale_smooth_window,
    )
    tracks_3d_npz = Path(fusion_summary["tracks_3d_npz"])
    print(f"[2/4] Saved fused 3D tracks: {tracks_3d_npz}")

    selection_dir = args.output_dir / "selection"
    selection_summary = select_best_tracks(tracks_3d_npz, selection_dir, args.top_k, args.quality_smooth_window)
    selected_path = Path(selection_summary["selected_tracks_npz"])
    print(f"[3/4] Saved selected tracks: {selected_path}")

    visualization_dir = args.output_dir / "visualization"
    visualization_summary = compute_structure_displacement(tracks_3d_npz, selected_path, visualization_dir)
    print(f"[4/4] Saved structure displacement: {visualization_summary['structure_csv']}")


if __name__ == "__main__":
    main()
