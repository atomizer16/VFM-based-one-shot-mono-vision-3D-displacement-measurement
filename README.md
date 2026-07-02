# VFM-based one-shot Monocular 3D Displacement Measurement System
The source code for "Monocular 3D displacement measurement framework based on vision foundation model using low-cost camera"

This repository code combines the capabilities of three different vision foundation models to achieve satisfactory monocular 3D displacement measurement
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

Use the actual ArUco code side length.

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
  --top-k 100 \
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

# Frame Sequence Relative Depth PNG Export

This script runs `depth-anything/DA3-GIANT-1.1` on an ordered image sequence
and exports one color depth PNG per source frame:

```bash
python tools/infer_depth_sequence_png.py \
    --input-dir /lustre/home/acct-cwj/cwj-user1/sam3-main/assets/videos/1024-1024 \
    --output-dir ./output_pod_depth \
    --model-dir depth-anything/DA3-GIANT-1.1 \
    --mode temporal_window \
    --window-size 8 \
    --overlap 4 \
    --process-res 504
```

Outputs:

```text
output_pod_depth/
  color_depth/    # one RGB-colored PNG for each input frame
  raw_depth/      # one float32 .npy relative-depth map for each input frame
```

The PNG color scale is computed once from the whole sequence. By default it
colors inverse depth (`--visualization inverse_depth`), as the official depth
visualization helper does, while retaining unmodified depth arrays in
`raw_depth/`. This avoids frame-to-frame visualization flicker caused by
normalizing each depth map independently. Use `--visualization depth` if the
PNG color should increase directly with predicted relative depth.

## Inference Modes

`--mode independent` passes one image at a time to the model. It is the
lowest-memory baseline, but each result has no temporal context and its
relative-depth scale can drift between frames.

`--mode temporal_window` is the default and is recommended for extracted
video frames. Each short consecutive window is jointly inferred by DA3, so
the model uses cross-frame attention within that window. Adjacent windows
share frames; the script estimates a robust relative-depth scale from those
duplicate predictions and blends them to suppress window-boundary changes.

This is a practical temporal-consistency method, not a formal guarantee of
temporally perfect depth. In particular, the moving plant stem can legitimately
change depth and silhouette, so applying direct pixel-wise temporal smoothing
between different frames would blur or lag real motion. The script therefore
uses temporal model context and overlap consistency without smoothing away
object movement.

## Mask-Aware ROI Output

If you have per-frame instance masks, pass the mask directory. The script keeps
labels `1,2,3` by default, matching:

```text
beanstalk = 1
aruco1    = 2
aruco2    = 3
```

```bash
python tools/infer_depth_sequence_png.py \
    --input-dir /lustre/home/acct-cwj/cwj-user1/sam3-main/assets/videos/1024-1024 \
    --mask-dir /lustre/home/acct-cwj/cwj-user1/sam3-main/assets/videos/output/masks/combined_instance \
    --output-dir ./output_pod_depth_masked \
    --model-dir depth-anything/DA3-GIANT-1.1 \
    --mode temporal_window \
    --window-size 8 \
    --overlap 4 \
    --process-res 504
```

Mask matching defaults to filename stems. If the image and mask filenames are
different but their natural sorted order is guaranteed to correspond, add:

```bash
--mask-match sorted
```

By default the matcher also strips `_instance_mask` from mask filename stems, so
`frame_000501.png` matches `frame_000501_instance_mask.png`. If your mask suffix
is different, set:

```bash
--mask-stem-suffix _your_suffix
```

With `--mask-dir`, the default behavior keeps the full RGB frames as DA3 input.
This lets background structure continue to help geometry/depth inference. The
mask is applied after inference: pixels outside labels `1,2,3` are set to zero
in `raw_depth/` and black in `color_depth/`. The final color PNGs keep the
original input frame size unless `--processed-size-output` is set.

The color scale is global across all kept mask pixels and all frames, so the
three target instances share one relative-depth visualization range. This is
important when comparing beanstalk, `aruco1`, and `aruco2` against each other.

If you explicitly want to remove background context before inference, add
`--mask-prompt-input`. That writes masked images under
`output_pod_depth_masked/masked_inputs/` and feeds those masked images to DA3,
but this is not recommended when the background carries useful depth cues.

DA3 does not expose a native `mask_prompt` argument, so this does not alter the
model's internal attention implementation directly.

## ROI Fusion Mode

The script has three segment-aware ROI modes:

```text
--roi-mode output_only
```

Runs full-context DA3 inference on the original frames. Background remains
available to the model. Masks are applied only to saved depth arrays and color
PNGs. This is the default and is the most reliable mode for preserving relative
depth between `beanstalk`, `aruco1`, and `aruco2`.

```text
--roi-mode fused_masked_affine
```

Runs two passes:

```text
1. full_context_raw_depth/   # original full RGB frames
2. masked_input_raw_depth/   # non-mask pixels blacked out before DA3 inference
```

Then for each frame and each instance label, it robustly aligns the masked-input
depth to the full-context depth with an affine transform:

```text
D_full ~= a * D_masked + b
```

The final ROI depth is:

```text
D_fused = alpha * D_masked_aligned + (1 - alpha) * D_full
```

Run:

```bash
python tools/infer_depth_sequence_png.py \
    --input-dir /lustre/home/acct-cwj/cwj-user1/sam3-main/assets/videos/1024-1024 \
    --mask-dir /lustre/home/acct-cwj/cwj-user1/sam3-main/assets/videos/output/masks/combined_instance \
    --output-dir ./output_pod_depth_fused \
    --model-dir depth-anything/DA3-GIANT-1.1 \
    --mode temporal_window \
    --roi-mode fused_masked_affine \
    --fusion-alpha 0.6 \
    --window-size 8 \
    --overlap 4 \
    --process-res 504
```

`--fusion-alpha` controls how much the aligned masked-input depth contributes.
Start with `0.6`. Lower it toward `0.3` if inter-instance relative depth looks
less stable than the full-context output; raise it toward `0.8` only if the
masked-input result clearly improves instance-internal structure.

```text
--roi-mode beanstalk_crop_fused
```

This mode is designed for the case where beanstalk internal depth detail matters,
while `aruco1` and `aruco2` are only smooth reference anchors. It runs:

```text
1. full_context_raw_depth/    # full RGB frames; preserves inter-instance depth relation
2. beanstalk_crop_inputs/     # high-resolution crops around beanstalk only
3. beanstalk_crop_raw_depth/  # crop depth aligned back to full-context beanstalk depth
4. raw_depth/                 # final fused ROI depth
5. color_depth/               # final color PNGs
6. depth_metrics.csv          # per-frame robust mean depths and differences
```

Only the target region is refined with high-resolution crop inference. ArUco regions
remain from the full-context pass, and their robust mean depths are written to
`depth_metrics.csv`.

Run:

```bash
python tools/infer_depth_sequence_png.py \
    --input-dir /lustre/home/acct-cwj/cwj-user1/sam3-main/assets/videos/2048-2048 \
    --mask-dir /lustre/home/acct-cwj/cwj-user1/sam3-main/assets/videos/output/masks/combined_instance \
    --output-dir ./output_pod_depth_beanstalk_crop \
    --model-dir depth-anything/DA3-GIANT-1.1 \
    --mode temporal_window \
    --roi-mode beanstalk_crop_fused \
    --window-size 8 \
    --overlap 4 \
    --process-res 1008 \
    --crop-process-res 1344 \
    --crop-margin 0.6 \
    --fusion-alpha 0.6
```

Use lower `--fusion-alpha` values if the beanstalk-to-ArUco relative depth drifts
from the full-context result. Use higher values only if the crop result visibly
improves beanstalk internal structure without harming the anchor relationship.

## Memory Guidance

For a 48 GB GPU, start with:

```bash
--window-size 8 --overlap 4 --process-res 1008
```

If memory headroom remains and stronger within-window context is useful, try
`--window-size 12 --overlap 6`. Do not pass all frames in one inference
call; the any-view model performs global attention over the supplied frames.
