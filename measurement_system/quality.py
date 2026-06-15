from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import save_json
from .export import write_csv


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values.astype(np.float32)
    if window % 2 == 0:
        window += 1
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window <= 1:
        return values.astype(np.float32)
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")


def continuity_score(series: np.ndarray, valid: np.ndarray, smooth_window: int) -> tuple[float, float]:
    use = valid & np.isfinite(series)
    values = series[use].astype(np.float32)
    if values.size < 5:
        return 0.0, float("inf")
    smooth = moving_average(values, smooth_window)
    residual = values - smooth
    jitter = float(np.median(np.abs(residual)))
    steps = np.diff(smooth)
    if steps.size == 0:
        return 0.0, jitter
    step_scale = float(np.median(np.abs(steps)) + 1e-9)
    jumps = np.abs(steps) > max(1e-6, 8.0 * step_scale)
    score = max(0.0, 1.0 - float(np.mean(jumps)) - min(1.0, jitter / (10.0 * step_scale + 1e-9)))
    return score, jitter


def score_tracks_3d(npz_path: Path, smooth_window: int) -> tuple[list[dict], np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        frame_ids = data["frame_ids"].astype(np.int32)
        track_ids = data["track_ids"].astype(np.int32)
        displacement = data["displacement"].astype(np.float32)
        valid = data["valid"].astype(bool)
        roma_valid = data["roma_valid"].astype(bool)
        confidence = data["confidence"].astype(np.float32)

    depth_valid_ratio = valid.mean(axis=0)
    roma_valid_ratio = roma_valid.mean(axis=0)
    mean_confidence = np.divide(
        (confidence * roma_valid).sum(axis=0),
        np.maximum(1, roma_valid.sum(axis=0)),
    )
    median_disp = np.nanmedian(np.where(valid[:, :, None], displacement, np.nan), axis=1)
    rows: list[dict] = []
    for track_idx, track_id in enumerate(track_ids.tolist()):
        axis_scores = []
        jitters = []
        for axis in range(3):
            score, jitter = continuity_score(displacement[:, track_idx, axis], valid[:, track_idx], smooth_window)
            axis_scores.append(score)
            jitters.append(jitter)
        use = valid[:, track_idx] & np.all(np.isfinite(displacement[:, track_idx]), axis=1) & np.all(np.isfinite(median_disp), axis=1)
        if np.any(use):
            residual = displacement[use, track_idx] - median_disp[use]
            group_error = float(np.median(np.linalg.norm(residual, axis=1)))
        else:
            group_error = float("inf")
        group_score = 1.0 / (1.0 + group_error)
        jitter_median = float(np.nanmedian(jitters))
        low_jitter_score = 1.0 / (1.0 + jitter_median)
        continuity = float(np.mean(axis_scores))
        rank_score = (
            0.25 * float(roma_valid_ratio[track_idx])
            + 0.15 * float(min(1.0, mean_confidence[track_idx]))
            + 0.20 * float(depth_valid_ratio[track_idx])
            + 0.20 * continuity
            + 0.10 * low_jitter_score
            + 0.10 * group_score
        )
        rows.append(
            {
                "track_id": int(track_id),
                "track_index": int(track_idx),
                "rank_score": float(rank_score),
                "roma_valid_ratio": float(roma_valid_ratio[track_idx]),
                "depth_valid_ratio": float(depth_valid_ratio[track_idx]),
                "mean_confidence": float(mean_confidence[track_idx]),
                "continuity_score": continuity,
                "low_jitter_score": float(low_jitter_score),
                "group_consistency_score": float(group_score),
                "jitter_m": jitter_median,
                "group_error_m": group_error,
            }
        )
    return rows, frame_ids


def select_best_tracks(npz_path: Path, output_dir: Path, top_k: int, smooth_window: int) -> dict:
    rows, _frame_ids = score_tracks_3d(npz_path, smooth_window)
    rows_sorted = sorted(rows, key=lambda row: row["rank_score"], reverse=True)
    selected = rows_sorted[:top_k]
    fieldnames = list(rows_sorted[0].keys()) if rows_sorted else ["track_id", "rank_score"]
    write_csv(output_dir / "track_quality.csv", fieldnames, rows_sorted)
    write_csv(output_dir / "selected_tracks.csv", fieldnames, selected)
    selected_indices = np.array([row["track_index"] for row in selected], dtype=np.int32)
    selected_track_ids = np.array([row["track_id"] for row in selected], dtype=np.int32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "selected_tracks.npz", selected_indices=selected_indices, selected_track_ids=selected_track_ids)
    summary = {
        "tracks_scored": len(rows),
        "selected_count": len(selected),
        "top_k": top_k,
        "selected_tracks_npz": str(output_dir / "selected_tracks.npz"),
    }
    save_json(summary, output_dir / "selection_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the most stable 3D keypoints from fused tracks.")
    parser.add_argument("--tracks-3d-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--smooth-window", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = select_best_tracks(args.tracks_3d_npz, args.output_dir, args.top_k, args.smooth_window)
    print(f"Selected tracks: {summary['selected_count']}/{summary['tracks_scored']}")
    print(f"Saved selected track IDs: {summary['selected_tracks_npz']}")


if __name__ == "__main__":
    main()

