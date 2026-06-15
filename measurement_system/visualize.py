from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import save_json
from .export import write_csv


def load_selected_indices(path: Path | None, num_tracks: int) -> np.ndarray:
    if path is None:
        return np.arange(num_tracks, dtype=np.int32)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return data["selected_indices"].astype(np.int32)
    indices: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        track_index_col = header.index("track_index")
        for line in f:
            if line.strip():
                indices.append(int(line.strip().split(",")[track_index_col]))
    return np.asarray(indices, dtype=np.int32)


def compute_structure_displacement(tracks_3d_npz: Path, selected_path: Path | None, output_dir: Path) -> dict:
    with np.load(tracks_3d_npz, allow_pickle=True) as data:
        frame_ids = data["frame_ids"].astype(np.int32)
        displacement = data["displacement"].astype(np.float32)
        valid = data["valid"].astype(bool)
    selected_indices = load_selected_indices(selected_path, displacement.shape[1])
    selected_disp = displacement[:, selected_indices]
    selected_valid = valid[:, selected_indices]
    masked = np.where(selected_valid[:, :, None], selected_disp, np.nan)
    structure_disp = np.nanmedian(masked, axis=1)
    valid_counts = np.sum(selected_valid, axis=1)

    rows = []
    for idx, frame_id in enumerate(frame_ids.tolist()):
        rows.append(
            {
                "frame_id": int(frame_id),
                "dX_m": float(structure_disp[idx, 0]),
                "dY_m": float(structure_disp[idx, 1]),
                "dZ_m": float(structure_disp[idx, 2]),
                "num_valid_points": int(valid_counts[idx]),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "structure_displacement.csv", ["frame_id", "dX_m", "dY_m", "dZ_m", "num_valid_points"], rows)
    np.savez_compressed(
        output_dir / "structure_displacement.npz",
        frame_ids=frame_ids,
        displacement=structure_disp,
        valid_counts=valid_counts,
        selected_indices=selected_indices,
    )
    plot_displacement(frame_ids, structure_disp, output_dir / "structure_displacement_xyz.png")
    plot_3d_curve(structure_disp, output_dir / "structure_displacement_3d.png")
    summary = {
        "frames": int(len(frame_ids)),
        "selected_points": int(len(selected_indices)),
        "structure_csv": str(output_dir / "structure_displacement.csv"),
    }
    save_json(summary, output_dir / "visualization_summary.json")
    return summary


def plot_displacement(frame_ids: np.ndarray, displacement: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = ["dX", "dY", "dZ"]
    for axis, label in enumerate(labels):
        ax.plot(frame_ids, displacement[:, axis], label=label)
    ax.set_xlabel("frame_id")
    ax.set_ylabel("displacement (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_3d_curve(displacement: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(displacement[:, 0], displacement[:, 1], displacement[:, 2], linewidth=1.6)
    ax.set_xlabel("dX (m)")
    ax.set_ylabel("dY (m)")
    ax.set_zlabel("dZ (m)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize selected 3D keypoints as structure-level displacement.")
    parser.add_argument("--tracks-3d-npz", type=Path, required=True)
    parser.add_argument("--selected-tracks", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compute_structure_displacement(args.tracks_3d_npz, args.selected_tracks, args.output_dir)
    print(f"Saved structure displacement: {summary['structure_csv']}")


if __name__ == "__main__":
    main()
