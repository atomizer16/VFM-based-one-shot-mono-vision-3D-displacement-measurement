from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """OpenCV pinhole camera model."""

    K: np.ndarray
    dist: np.ndarray
    image_width: int | None = None
    image_height: int | None = None
    reprojection_error: float | None = None

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])


def _camera_to_jsonable(camera: CameraModel) -> dict[str, Any]:
    data = asdict(camera)
    data["K"] = camera.K.tolist()
    data["dist"] = camera.dist.reshape(-1).tolist()
    return data


def save_camera(camera: CameraModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_camera_to_jsonable(camera), f, indent=2)


def load_camera(path: Path) -> CameraModel:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return CameraModel(
        K=np.asarray(data["K"], dtype=np.float64),
        dist=np.asarray(data.get("dist", []), dtype=np.float64).reshape(-1, 1),
        image_width=data.get("image_width"),
        image_height=data.get("image_height"),
        reprojection_error=data.get("reprojection_error"),
    )


def save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

