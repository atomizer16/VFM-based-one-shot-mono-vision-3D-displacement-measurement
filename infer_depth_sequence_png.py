#!/usr/bin/env python3
"""Estimate relative depth for an ordered image sequence with Depth Anything 3.

The default temporal mode runs short overlapping clips through DA3 jointly. It
aligns the arbitrary relative-depth scale between neighboring clips using the
overlap and blends duplicate predictions before creating color PNGs with one
shared color range for the whole sequence.

When masks are provided, the default behavior keeps full RGB frames for DA3
inference and uses masks only for output filtering and overlap-scale statistics.
This preserves background context while exporting only the requested objects.
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from depth_anything_3.api import DepthAnything3  # noqa: E402


DEFAULT_MODEL = "depth-anything/DA3-GIANT-1.1"
DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
DEFAULT_MASK_INSTANCE_IDS = (1, 2, 3)


def natural_key(path: Path) -> list[object]:
    return [int(value) if value.isdigit() else value.lower() for value in re.split(r"(\d+)", path.name)]


def collect_images(input_dir: Path) -> list[Path]:
    paths = sorted(
        (path for path in input_dir.iterdir() if path.suffix.lower() in DEFAULT_EXTENSIONS),
        key=natural_key,
    )
    if not paths:
        raise ValueError(f"No supported images found in: {input_dir}")

    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Input images contain duplicate filename stems; output PNG names would collide.")
    return paths


def parse_instance_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not ids:
        raise ValueError("--mask-instance-ids must contain at least one integer label.")
    return ids


def collect_masks(
    images: list[Path],
    mask_dir: Path,
    match_mode: str,
    mask_stem_suffix: str = "_instance_mask",
) -> list[Path]:
    masks = sorted(
        (path for path in mask_dir.iterdir() if path.suffix.lower() in DEFAULT_EXTENSIONS),
        key=natural_key,
    )
    if not masks:
        raise ValueError(f"No supported mask PNGs/images found in: {mask_dir}")

    if match_mode == "sorted":
        if len(masks) != len(images):
            raise ValueError(
                f"Sorted mask matching requires the same count: {len(images)} images, {len(masks)} masks."
            )
        return masks

    mask_by_stem = {path.stem: path for path in masks}
    if mask_stem_suffix:
        for path in masks:
            if path.stem.endswith(mask_stem_suffix):
                base_stem = path.stem[: -len(mask_stem_suffix)]
                mask_by_stem.setdefault(base_stem, path)

    for image_path in images:
        if image_path.stem in mask_by_stem:
            continue
        prefix_matches = [path for path in masks if path.stem.startswith(f"{image_path.stem}_")]
        if len(prefix_matches) == 1:
            mask_by_stem[image_path.stem] = prefix_matches[0]

    missing = [path.name for path in images if path.stem not in mask_by_stem]
    if missing and match_mode == "auto" and len(masks) == len(images):
        print("Mask stems do not fully match image stems; falling back to natural sorted pairing.")
        return masks
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Missing masks for {len(missing)} images when matching by filename stem. "
            f"Examples: {preview}. Use --mask-match sorted only if ordering is guaranteed."
        )
    return [mask_by_stem[path.stem] for path in images]


def load_binary_mask(
    mask_path: Path,
    instance_ids: tuple[int, ...],
    size: tuple[int, int] | None = None,
    dilate: int = 0,
) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if size is not None and (mask.shape[1], mask.shape[0]) != size:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    binary = np.isin(mask, instance_ids)
    if dilate > 0:
        kernel_size = 2 * dilate + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
    return binary


def prepare_masked_prompt_inputs(
    images: list[Path],
    mask_paths: list[Path] | None,
    output_dir: Path,
    instance_ids: tuple[int, ...],
    dilate: int,
    enabled: bool,
) -> list[Path]:
    if mask_paths is None or not enabled:
        return images

    masked_dir = output_dir / "masked_inputs"
    masked_dir.mkdir(parents=True, exist_ok=True)
    masked_paths = []

    for image_path, mask_path in tqdm(
        list(zip(images, mask_paths)), desc="Writing masked prompt inputs"
    ):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read source image: {image_path}")
        height, width = image.shape[:2]
        binary = load_binary_mask(mask_path, instance_ids, size=(width, height), dilate=dilate)
        masked = image.copy()
        masked[~binary] = 0
        output_path = masked_dir / f"{image_path.stem}.png"
        if not cv2.imwrite(str(output_path), masked):
            raise RuntimeError(f"Failed to write masked prompt image: {output_path}")
        masked_paths.append(output_path)
    return masked_paths


def get_model_inputs(
    images: list[Path],
    mask_paths: list[Path] | None,
    output_dir: Path,
    instance_ids: tuple[int, ...],
    dilate: int,
    mask_prompt_input: bool,
) -> list[Path]:
    if mask_paths is None or not mask_prompt_input:
        return images
    return prepare_masked_prompt_inputs(images, mask_paths, output_dir, instance_ids, dilate, True)


def get_window_starts(num_frames: int, window_size: int, overlap: int) -> list[int]:
    if num_frames <= window_size:
        return [0]

    starts = [0]
    step = window_size - overlap
    while starts[-1] + window_size < num_frames:
        next_start = min(starts[-1] + step, num_frames - window_size)
        if next_start == starts[-1]:
            break
        starts.append(next_start)
    return starts


def window_weights(length: int) -> np.ndarray:
    left = np.arange(1, length + 1, dtype=np.float32)
    right = left[::-1]
    return np.minimum(left, right)


def robust_scale(
    reference: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray | None = None,
    stride: int = 4,
) -> float:
    ref = reference[::stride, ::stride]
    cur = current[::stride, ::stride]
    valid = np.isfinite(ref) & np.isfinite(cur) & (ref > 1e-6) & (cur > 1e-6)
    if mask is not None:
        valid &= mask[::stride, ::stride]
    if valid.sum() < 64:
        return 1.0
    ratios = ref[valid] / cur[valid]
    low, high = np.percentile(ratios, [5.0, 95.0])
    ratios = ratios[(ratios >= low) & (ratios <= high)]
    return float(np.median(ratios)) if ratios.size else 1.0


def robust_affine_align(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    min_pixels: int = 64,
) -> tuple[float, float]:
    valid = (
        mask
        & np.isfinite(source)
        & np.isfinite(target)
        & (source > 1e-6)
        & (target > 1e-6)
    )
    if valid.sum() < min_pixels:
        return 1.0, 0.0

    x = source[valid].astype(np.float64)
    y = target[valid].astype(np.float64)
    x_low, x_high = np.percentile(x, [2.0, 98.0])
    y_low, y_high = np.percentile(y, [2.0, 98.0])
    keep = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    if keep.sum() >= min_pixels:
        x = x[keep]
        y = y[keep]

    if np.std(x) < 1e-8:
        return 1.0, float(np.median(y - x))

    design = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (a * x + b)
    med = np.median(residual)
    mad = np.median(np.abs(residual - med)) + 1e-8
    keep = np.abs(residual - med) <= 3.0 * 1.4826 * mad
    if keep.sum() >= min_pixels:
        design = design[keep]
        y = y[keep]
        a, b = np.linalg.lstsq(design, y, rcond=None)[0]

    if not np.isfinite(a) or not np.isfinite(b):
        return 1.0, 0.0
    return float(a), float(b)


def empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_depth(
    depth: np.ndarray,
    index: int,
    images: list[Path],
    raw_dir: Path,
    mask_paths: list[Path] | None = None,
    instance_ids: tuple[int, ...] = DEFAULT_MASK_INSTANCE_IDS,
    mask_dilate: int = 0,
) -> None:
    if mask_paths is not None:
        height, width = depth.shape
        mask = load_binary_mask(
            mask_paths[index], instance_ids, size=(width, height), dilate=mask_dilate
        )
        depth = depth.copy()
        depth[~mask] = 0.0
    np.save(raw_dir / f"{images[index].stem}.npy", depth.astype(np.float32))


def infer_independent(
    model: DepthAnything3,
    images: list[Path],
    model_inputs: list[Path],
    raw_dir: Path,
    process_res: int,
    process_res_method: str,
    mask_paths: list[Path] | None,
    instance_ids: tuple[int, ...],
    mask_dilate: int,
) -> None:
    for index, image_path in enumerate(tqdm(model_inputs, desc="Independent inference")):
        prediction = model.inference(
            image=[str(image_path)],
            process_res=process_res,
            process_res_method=process_res_method,
        )
        save_depth(
            prediction.depth[0], index, images, raw_dir, mask_paths, instance_ids, mask_dilate
        )
        del prediction
        empty_cuda_cache()


def infer_temporal_windows(
    model: DepthAnything3,
    images: list[Path],
    model_inputs: list[Path],
    raw_dir: Path,
    process_res: int,
    process_res_method: str,
    window_size: int,
    overlap: int,
    mask_paths: list[Path] | None,
    instance_ids: tuple[int, ...],
    mask_dilate: int,
) -> None:
    starts = get_window_starts(len(images), window_size, overlap)
    accum: dict[int, tuple[np.ndarray, float]] = {}

    for window_index, start in enumerate(tqdm(starts, desc="Temporal-window inference")):
        end = min(start + window_size, len(images))
        prediction = model.inference(
            image=[str(path) for path in model_inputs[start:end]],
            process_res=process_res,
            process_res_method=process_res_method,
            ref_view_strategy="middle",
        )
        depths = prediction.depth.astype(np.float32, copy=False)

        overlap_scales = []
        for local_index, frame_index in enumerate(range(start, end)):
            if frame_index in accum:
                summed, weight = accum[frame_index]
                scale_mask = None
                if mask_paths is not None:
                    height, width = depths[local_index].shape
                    scale_mask = load_binary_mask(
                        mask_paths[frame_index],
                        instance_ids,
                        size=(width, height),
                        dilate=mask_dilate,
                    )
                overlap_scales.append(
                    robust_scale(summed / weight, depths[local_index], mask=scale_mask)
                )
        scale = float(np.median(overlap_scales)) if overlap_scales else 1.0
        depths = depths * scale

        weights = window_weights(end - start)
        for local_index, frame_index in enumerate(range(start, end)):
            weighted_depth = depths[local_index] * weights[local_index]
            if frame_index in accum:
                summed, total_weight = accum[frame_index]
                accum[frame_index] = (summed + weighted_depth, total_weight + weights[local_index])
            else:
                accum[frame_index] = (weighted_depth.copy(), float(weights[local_index]))

        next_start = starts[window_index + 1] if window_index + 1 < len(starts) else len(images)
        for frame_index in sorted(index for index in accum if index < next_start):
            summed, total_weight = accum.pop(frame_index)
            save_depth(
                summed / total_weight,
                frame_index,
                images,
                raw_dir,
                mask_paths,
                instance_ids,
                mask_dilate,
            )

        del prediction, depths
        empty_cuda_cache()

    for frame_index in sorted(accum):
        summed, total_weight = accum[frame_index]
        save_depth(
            summed / total_weight,
            frame_index,
            images,
            raw_dir,
            mask_paths,
            instance_ids,
            mask_dilate,
        )


def run_depth_inference(
    model: DepthAnything3,
    images: list[Path],
    model_inputs: list[Path],
    raw_dir: Path,
    process_res: int,
    process_res_method: str,
    inference_mode: str,
    window_size: int,
    overlap: int,
    mask_paths: list[Path] | None,
    instance_ids: tuple[int, ...],
    mask_dilate: int,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if inference_mode == "independent":
        infer_independent(
            model,
            images,
            model_inputs,
            raw_dir,
            process_res,
            process_res_method,
            mask_paths,
            instance_ids,
            mask_dilate,
        )
    else:
        print(f"Temporal windows: size={window_size}, overlap={overlap}")
        infer_temporal_windows(
            model,
            images,
            model_inputs,
            raw_dir,
            process_res,
            process_res_method,
            window_size,
            overlap,
            mask_paths,
            instance_ids,
            mask_dilate,
        )


def fuse_full_and_masked_depths(
    images: list[Path],
    mask_paths: list[Path],
    full_raw_dir: Path,
    masked_raw_dir: Path,
    fused_raw_dir: Path,
    instance_ids: tuple[int, ...],
    mask_dilate: int,
    alpha: float,
) -> None:
    fused_raw_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(tqdm(images, desc="Fusing full and masked depths")):
        full_depth = np.load(full_raw_dir / f"{image_path.stem}.npy").astype(np.float32)
        masked_depth = np.load(masked_raw_dir / f"{image_path.stem}.npy").astype(np.float32)
        if full_depth.shape != masked_depth.shape:
            raise RuntimeError(
                f"Depth shape mismatch for {image_path.name}: "
                f"{full_depth.shape} vs {masked_depth.shape}"
            )

        height, width = full_depth.shape
        fused = np.zeros_like(full_depth, dtype=np.float32)
        assigned = np.zeros((height, width), dtype=bool)

        for instance_id in instance_ids:
            inst_mask = load_binary_mask(
                mask_paths[index], (instance_id,), size=(width, height), dilate=mask_dilate
            )
            if not np.any(inst_mask):
                continue

            a, b = robust_affine_align(masked_depth, full_depth, inst_mask)
            aligned_masked = (a * masked_depth + b).astype(np.float32)
            full_valid = np.isfinite(full_depth) & (full_depth > 1e-6)
            masked_valid = np.isfinite(aligned_masked) & (aligned_masked > 1e-6)

            both_valid = inst_mask & full_valid & masked_valid
            full_only = inst_mask & full_valid & ~masked_valid
            masked_only = inst_mask & masked_valid & ~full_valid

            fused[both_valid] = (
                alpha * aligned_masked[both_valid] + (1.0 - alpha) * full_depth[both_valid]
            )
            fused[full_only] = full_depth[full_only]
            fused[masked_only] = aligned_masked[masked_only]
            assigned |= inst_mask

        fused[~assigned] = 0.0
        fused[~np.isfinite(fused)] = 0.0
        fused[fused <= 0] = 0.0
        np.save(fused_raw_dir / f"{image_path.stem}.npy", fused.astype(np.float32))


def visualization_values(depth: np.ndarray, visualization: str) -> np.ndarray:
    values = np.zeros_like(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6)
    if visualization == "depth":
        values[valid] = depth[valid]
    else:
        values[valid] = 1.0 / depth[valid]
    return values


def calculate_global_range(
    images: list[Path], raw_dir: Path, visualization: str
) -> tuple[float, float]:
    samples = []
    for image_path in images:
        depth = np.load(raw_dir / f"{image_path.stem}.npy", mmap_mode="r")
        values = visualization_values(np.asarray(depth[::8, ::8]), visualization).reshape(-1)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            samples.append(values)
    if not samples:
        raise RuntimeError("No valid positive depth values were predicted.")
    merged = np.concatenate(samples)
    low, high = np.percentile(merged, [2.0, 98.0])
    if high <= low:
        high = low + 1e-6
    return float(low), float(high)


def colorize_depth(
    images: list[Path],
    raw_dir: Path,
    color_dir: Path,
    depth_range: tuple[float, float],
    visualization: str,
    original_size: bool,
    mask_paths: list[Path] | None = None,
    instance_ids: tuple[int, ...] = DEFAULT_MASK_INSTANCE_IDS,
    mask_dilate: int = 0,
) -> None:
    low, high = depth_range
    for index, image_path in enumerate(tqdm(images, desc="Writing color PNGs")):
        depth = np.load(raw_dir / f"{image_path.stem}.npy")
        values = visualization_values(depth, visualization)
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        gray = np.round(normalized * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        processed_mask = None
        if mask_paths is not None:
            height, width = depth.shape
            processed_mask = load_binary_mask(
                mask_paths[index],
                instance_ids,
                size=(width, height),
                dilate=mask_dilate,
            )
            color[~processed_mask] = 0
        if original_size:
            source = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if source is None:
                raise RuntimeError(f"Cannot read source image to restore output size: {image_path}")
            height, width = source.shape[:2]
            color = cv2.resize(color, (width, height), interpolation=cv2.INTER_LINEAR)
            if mask_paths is not None:
                original_mask = load_binary_mask(
                    mask_paths[index],
                    instance_ids,
                    size=(width, height),
                    dilate=mask_dilate,
                )
                color[~original_mask] = 0
        if not cv2.imwrite(str(color_dir / f"{image_path.stem}.png"), color):
            raise RuntimeError(f"Failed to write output for: {image_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate color relative-depth PNGs for a frame sequence.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing ordered video frames.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root directory.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL, help="Hugging Face model id or local model path.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--mode",
        choices=("temporal_window", "independent"),
        default="temporal_window",
        help="Joint short-window inference with overlap fusion, or one image per inference.",
    )
    parser.add_argument(
        "--roi-mode",
        choices=("output_only", "fused_masked_affine"),
        default="output_only",
        help=(
            "Segment-aware depth mode. output_only runs full-context inference and masks only "
            "outputs. fused_masked_affine additionally runs masked-input inference and fuses "
            "it after per-instance affine alignment to the full-context result."
        ),
    )
    parser.add_argument("--window-size", type=int, default=8, help="Frames jointly inferred in temporal mode.")
    parser.add_argument("--overlap", type=int, default=4, help="Repeated frames between temporal windows.")
    parser.add_argument("--process-res", type=int, default=504, help="Model processing resolution.")
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=("upper_bound_resize", "upper_bound_crop", "lower_bound_resize", "lower_bound_crop"),
    )
    parser.add_argument(
        "--processed-size-output",
        action="store_true",
        help="Write PNGs at model resolution instead of resizing to the input frame resolution.",
    )
    parser.add_argument(
        "--visualization",
        choices=("inverse_depth", "depth"),
        default="inverse_depth",
        help="Colorize inverse depth (official visualization convention) or raw relative depth.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Directory with instance-mask PNGs matched to input frames.",
    )
    parser.add_argument(
        "--mask-instance-ids",
        default="1,2,3",
        help="Comma-separated instance labels to keep, e.g. '1,2,3' for beanstalk and two ArUco markers.",
    )
    parser.add_argument(
        "--mask-match",
        choices=("stem", "sorted", "auto"),
        default="stem",
        help="Match masks by filename stem, natural sorted order, or stem with sorted fallback.",
    )
    parser.add_argument(
        "--mask-stem-suffix",
        default="_instance_mask",
        help=(
            "Suffix to strip from mask filename stems before matching image stems. "
            "Default matches frame_000501.png to frame_000501_instance_mask.png."
        ),
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=0,
        help="Dilate binary masks by this many pixels before masking.",
    )
    parser.add_argument(
        "--mask-prompt-input",
        action="store_true",
        help=(
            "Black out non-mask image regions before DA3 inference. Disabled by default so "
            "background context remains available to the model."
        ),
    )
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=0.6,
        help=(
            "For --roi-mode fused_masked_affine, weight for affine-aligned masked-input depth. "
            "The full-context depth weight is 1-alpha."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_size < 1:
        raise ValueError("--window-size must be positive.")
    if args.mode == "temporal_window" and not 0 <= args.overlap < args.window_size:
        raise ValueError("--overlap must be in [0, window-size) for temporal mode.")
    if not 0.0 <= args.fusion_alpha <= 1.0:
        raise ValueError("--fusion-alpha must be in [0, 1].")

    images = collect_images(args.input_dir)
    instance_ids = parse_instance_ids(args.mask_instance_ids)
    mask_paths = (
        collect_masks(images, args.mask_dir, args.mask_match, args.mask_stem_suffix)
        if args.mask_dir
        else None
    )
    if args.roi_mode == "fused_masked_affine" and mask_paths is None:
        raise ValueError("--roi-mode fused_masked_affine requires --mask-dir.")
    raw_dir = args.output_dir / "raw_depth"
    color_dir = args.output_dir / "color_depth"
    raw_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} images.")
    if mask_paths is not None:
        print(
            f"Using {len(mask_paths)} masks. Keeping instance labels: "
            f"{','.join(str(item) for item in instance_ids)}"
        )
    print(f"Loading model: {args.model_dir}")
    model = DepthAnything3.from_pretrained(args.model_dir).to(args.device).eval()

    if args.roi_mode == "fused_masked_affine":
        print("ROI mode: fused_masked_affine")
        full_raw_dir = args.output_dir / "full_context_raw_depth"
        masked_raw_dir = args.output_dir / "masked_input_raw_depth"

        print("Pass 1/2: full-context inference")
        run_depth_inference(
            model,
            images,
            images,
            full_raw_dir,
            args.process_res,
            args.process_res_method,
            args.mode,
            args.window_size,
            args.overlap,
            mask_paths,
            instance_ids,
            args.mask_dilate,
        )

        print("Pass 2/2: masked-input inference")
        masked_inputs = prepare_masked_prompt_inputs(
            images, mask_paths, args.output_dir, instance_ids, args.mask_dilate, True
        )
        run_depth_inference(
            model,
            images,
            masked_inputs,
            masked_raw_dir,
            args.process_res,
            args.process_res_method,
            args.mode,
            args.window_size,
            args.overlap,
            mask_paths,
            instance_ids,
            args.mask_dilate,
        )
        fuse_full_and_masked_depths(
            images,
            mask_paths,
            full_raw_dir,
            masked_raw_dir,
            raw_dir,
            instance_ids,
            args.mask_dilate,
            args.fusion_alpha,
        )
    else:
        print("ROI mode: output_only")
        model_inputs = get_model_inputs(
            images,
            mask_paths,
            args.output_dir,
            instance_ids,
            args.mask_dilate,
            mask_prompt_input=args.mask_prompt_input,
        )
        run_depth_inference(
            model,
            images,
            model_inputs,
            raw_dir,
            args.process_res,
            args.process_res_method,
            args.mode,
            args.window_size,
            args.overlap,
            mask_paths,
            instance_ids,
            args.mask_dilate,
        )

    low, high = calculate_global_range(images, raw_dir, args.visualization)
    print(
        f"Global {args.visualization} visualization range (2nd-98th percentile): "
        f"{low:.6g} .. {high:.6g}"
    )
    colorize_depth(
        images,
        raw_dir,
        color_dir,
        (low, high),
        args.visualization,
        original_size=not args.processed_size_output,
        mask_paths=mask_paths,
        instance_ids=instance_ids,
        mask_dilate=args.mask_dilate,
    )
    print(f"Color PNGs saved under: {color_dir}")
    print(f"Raw relative depths saved under: {raw_dir}")


if __name__ == "__main__":
    main()
