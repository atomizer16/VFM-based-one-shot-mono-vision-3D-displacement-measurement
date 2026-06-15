# VFM-based one-shot Monocular 3D Displacement Measurement System
The source code for "Monocular 3D displacement measurement framework based on vision foundation model using low-cost camera"

This repository code combines the capabilities of three different vision models to achieve satisfactory monocular 3D displacement measurement
accuracy without the need for fine-tuning:

1. `sam3_promptmask.py` creates instance masks.
2. `long_time_track.py` creates RoMa 2D tracks.
3. `infer_depth_sequence_png.py` creates DA3 raw relative depth maps.
4. `aruco_metric_scale.py` estimates metric scale from ArUco markers.
5. `fuse_tracks_3d.py` fuses RoMa, DA3, ArUco scale, and camera intrinsics.
6. `select_best_3d_keypoints.py` selects stable 3D keypoints.
7. `visualize_keypoints_3d.py` exports structure-level displacement curves.
8. `run_vsr.py` super resolution the low-res video to high-res.

## Coordinate System

Outputs use the OpenCV camera coordinate system:

- `X`: image right direction
- `Y`: image down direction
- `Z`: camera optical axis, positive away from camera

Displacements are computed relative to each point's first valid 3D observation.

## Camera Calibration

Run chessboard calibration once:

```bash
python calibrate_camera.py \
  --images-dir /path/to/chessboard_images \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size-m 0.025 \
  --output camera_calibration.json
```

`pattern-cols` and `pattern-rows` are the inner-corner counts.

## ArUco Metric Scale

Use the actual ArUco code side length. For your markers this is `0.40 m`, not
the outer `0.50 m` board size.

```bash
python aruco_metric_scale.py \
  --frames-dir /path/to/frames \
  --masks-dir /path/to/sam_output/masks/combined_instance \
  --depth-dir /path/to/da3_output/raw_depth \
  --camera-json camera_calibration.json \
  --track-npz /path/to/roma_output/tracks.npz \
  --marker-length-m 0.40 \
  --aruco-dictionary DICT_4X4_50 \
  --aruco-label-ids 2,3 \
  --output-csv output/aruco_scale.csv
```

If `tracks.npz` is unavailable, use `--start-frame 501 --end-frame 2001`.

## Fuse 3D Tracks

```bash
python fuse_tracks_3d.py \
  --track-npz /path/to/roma_output/tracks.npz \
  --masks-dir /path/to/sam_output/masks/combined_instance \
  --depth-dir /path/to/da3_output/raw_depth \
  --camera-json camera_calibration.json \
  --scale-csv output/aruco_scale.csv \
  --object-label-id 1 \
  --output-dir output/fusion
```

The output includes:

- `tracks_3d.npz`
- `fusion_summary.json`
- `per_point/point_xxxx_3d.csv`

## Select Stable Keypoints

```bash
python select_best_3d_keypoints.py \
  --tracks-3d-npz output/fusion/tracks_3d.npz \
  --top-k 20 \
  --output-dir output/selection
```

## Visualize Structure Displacement

```bash
python visualize_keypoints_3d.py \
  --tracks-3d-npz output/fusion/tracks_3d.npz \
  --selected-tracks output/selection/selected_tracks.npz \
  --output-dir output/visualization
```

The structure displacement is the frame-wise median of selected keypoint
displacements.

## One-Command Post-Model Pipeline

```bash
python run_pipeline.py \
  --frames-dir /path/to/frames \
  --masks-dir /path/to/sam_output/masks/combined_instance \
  --depth-dir /path/to/da3_output/raw_depth \
  --camera-json camera_calibration.json \
  --track-npz /path/to/roma_output/tracks.npz \
  --output-dir output/measurement \
  --marker-length-m 0.40 \
  --aruco-dictionary DICT_4X4_50 \
  --aruco-label-ids 2,3 \
  --object-label-id 1 \
  --top-k 20
```
