from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import CameraModel, save_camera, save_json
from .frame_manifest import IMAGE_SUFFIXES


def collect_calibration_images(images_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in IMAGE_SUFFIXES:
        paths.extend(images_dir.glob(f"*{suffix}"))
    return sorted(paths)


def calibrate_chessboard(
    images_dir: Path,
    pattern_cols: int,
    pattern_rows: int,
    square_size_m: float,
) -> CameraModel:
    import cv2

    object_template = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    object_template *= float(square_size_m)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image_path in collect_calibration_images(images_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image_size = (image.shape[1], image.shape[0])
        ok, corners = cv2.findChessboardCorners(image, (pattern_cols, pattern_rows))
        if not ok:
            continue
        refined = cv2.cornerSubPix(
            image,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        object_points.append(object_template.copy())
        image_points.append(refined)

    if image_size is None:
        raise RuntimeError(f"No readable calibration images found in: {images_dir}")
    if len(object_points) < 5:
        raise RuntimeError(f"Need at least 5 valid chessboard images, found {len(object_points)}.")

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    return CameraModel(
        K=K,
        dist=dist,
        image_width=image_size[0],
        image_height=image_size[1],
        reprojection_error=float(rms),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a fixed camera with chessboard images.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--pattern-cols", type=int, required=True, help="Inner chessboard corners along columns.")
    parser.add_argument("--pattern-rows", type=int, required=True, help="Inner chessboard corners along rows.")
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True, help="camera_calibration.json output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = calibrate_chessboard(args.images_dir, args.pattern_cols, args.pattern_rows, args.square_size_m)
    save_camera(camera, args.output)
    save_json(
        {
            "valid_output": str(args.output),
            "reprojection_error": camera.reprojection_error,
            "image_width": camera.image_width,
            "image_height": camera.image_height,
        },
        args.output.with_suffix(".summary.json"),
    )
    print(f"Saved camera calibration: {args.output}")
    print(f"RMS reprojection error: {camera.reprojection_error:.6g}")


if __name__ == "__main__":
    main()
