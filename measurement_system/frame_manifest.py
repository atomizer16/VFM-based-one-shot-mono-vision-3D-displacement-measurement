from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    frame_path: Path | None
    mask_path: Path | None
    depth_path: Path | None


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEPTH_SUFFIXES = (".npy", ".png", ".tif", ".tiff")


def frame_stem(frame_id: int) -> str:
    return f"frame_{frame_id:06d}"


def find_frame_file(directory: Path | None, frame_id: int, suffixes: tuple[str, ...]) -> Path | None:
    if directory is None:
        return None
    stem = frame_stem(frame_id)
    candidates = [
        *(directory / f"{stem}{suffix}" for suffix in suffixes),
        directory / f"{stem}_instance_mask.png",
        directory / f"{stem}_mask.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(directory.glob(f"{stem}*"))
    return matches[0] if matches else None


def build_manifest(
    start_frame: int,
    end_frame: int,
    frames_dir: Path | None,
    masks_dir: Path | None,
    depth_dir: Path | None,
) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for frame_id in range(start_frame, end_frame + 1):
        records.append(
            FrameRecord(
                frame_id=frame_id,
                frame_path=find_frame_file(frames_dir, frame_id, IMAGE_SUFFIXES),
                mask_path=find_frame_file(masks_dir, frame_id, IMAGE_SUFFIXES),
                depth_path=find_frame_file(depth_dir, frame_id, DEPTH_SUFFIXES),
            )
        )
    return records

