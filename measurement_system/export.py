from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def read_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    import cv2

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Cannot read depth file: {path}")
    return depth.astype(np.float32)


def read_mask(path: Path) -> np.ndarray:
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Cannot read mask file: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
