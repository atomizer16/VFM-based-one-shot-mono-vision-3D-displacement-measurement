from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from batch_run import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_MASK_DIR,
    frame_name,
    image_path,
    load_masked_image,
    mask_path,
)
from romav2 import RoMaV2
from romav2.geometry import to_normalized, to_pixel
from run import build_model_with_resilient_weights


def choose_uniform_random_points(mask: np.ndarray, num_points: int, seed: int, max_candidates: int) -> np.ndarray:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < num_points:
        raise ValueError(f"Mask has only {len(xs)} valid pixels, cannot choose {num_points} points")

    rng = np.random.default_rng(seed)
    num_candidates = min(max_candidates, len(xs))
    candidate_inds = rng.choice(len(xs), size=num_candidates, replace=False)
    candidates = np.stack((xs[candidate_inds], ys[candidate_inds]), axis=1).astype(np.float32)

    scale = np.array([max(1, mask.shape[1]), max(1, mask.shape[0])], dtype=np.float32)
    norm_candidates = candidates / scale

    selected = [int(rng.integers(0, len(candidates)))]
    min_dist2 = np.sum((norm_candidates - norm_candidates[selected[0]]) ** 2, axis=1)
    for _ in range(1, num_points):
        next_idx = int(np.argmax(min_dist2))
        selected.append(next_idx)
        dist2 = np.sum((norm_candidates - norm_candidates[next_idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

    return candidates[selected]


def sample_warp_and_overlap(
    preds: dict[str, torch.Tensor],
    points_xy: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points_xy) == 0:
        return points_xy.copy(), np.zeros(0, dtype=np.float32)

    warp = preds["warp_AB"]
    overlap = preds["overlap_AB"]
    device = warp.device

    points = torch.as_tensor(points_xy, dtype=torch.float32, device=device)
    points_norm = to_normalized(points, H=image_height, W=image_width)
    grid = points_norm.view(1, 1, -1, 2)

    sampled_norm = F.grid_sample(
        warp.permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        align_corners=False,
    )[0, :, 0].T
    sampled_overlap = F.grid_sample(
        overlap.permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        align_corners=False,
    )[0, 0, 0]

    sampled_xy = to_pixel(sampled_norm, H=image_height, W=image_width)
    return sampled_xy.detach().cpu().numpy(), sampled_overlap.detach().cpu().numpy()


def in_mask(points_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    valid = np.zeros(len(points_xy), dtype=bool)
    for i, (x, y) in enumerate(points_xy):
        xi, yi = int(round(float(x))), int(round(float(y)))
        valid[i] = 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0
    return valid


def load_frame(image_dir: Path, mask_dir: Path, frame_id: int, instance_id: int) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    return load_masked_image(
        image_path(image_dir, frame_id),
        mask_path(mask_dir, frame_id),
        instance_id,
    )


def track_color(track_id: int) -> tuple[int, int, int]:
    return (
        int((37 * (track_id + 3)) % 255),
        int((91 * (track_id + 7)) % 255),
        int((53 * (track_id + 13)) % 255),
    )


def save_initial_points_image(
    output_file: Path,
    first_arr: np.ndarray,
    first_mask: np.ndarray,
    initial_points: np.ndarray,
    max_side: int,
) -> None:
    image = Image.fromarray(first_arr).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    overlay_arr[first_mask > 0] = np.array([0, 255, 120, 60], dtype=np.uint8)
    image = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay_arr)).convert("RGB")

    draw = ImageDraw.Draw(image)
    radius = max(6, int(round(max(image.size) / 350)))
    for track_id, (x, y) in enumerate(initial_points):
        color = track_color(track_id)
        x_f, y_f = float(x), float(y)
        draw.ellipse(
            [(x_f - radius, y_f - radius), (x_f + radius, y_f + radius)],
            fill=color,
            outline=(255, 255, 255),
            width=max(2, radius // 3),
        )
        draw.text((x_f + radius + 3, y_f - radius - 3), str(track_id), fill=(255, 255, 255))

    if max_side > 0 and max(image.size) > max_side:
        scale = max_side / max(image.size)
        new_size = (int(round(image.size[0] * scale)), int(round(image.size[1] * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_file)


def write_track_txt(
    output_file: Path,
    tracks_xy: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    source: np.ndarray,
    frame_ids: np.ndarray,
    track_id: int,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("# frame_id x y valid confidence source\n")
        for frame_idx, frame_id in enumerate(frame_ids):
            x, y = tracks_xy[frame_idx, track_id]
            f.write(
                f"{int(frame_id)} {x:.6f} {y:.6f} "
                f"{int(valid[frame_idx, track_id])} {confidence[frame_idx, track_id]:.6f} "
                f"{source[frame_idx, track_id]}\n"
            )


def save_curve_png(
    output_file: Path,
    frame_ids: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    title: str,
    y_label: str,
    color: tuple[int, int, int],
    size: tuple[int, int] = (1100, 650),
) -> None:
    width, height = size
    margin_left, margin_right = 95, 35
    margin_top, margin_bottom = 60, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    valid_values = values[valid & np.isfinite(values)]
    if len(valid_values) == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min = float(np.nanmin(valid_values))
        y_max = float(np.nanmax(valid_values))
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        pad = max(1.0, 0.05 * (y_max - y_min))
        y_min -= pad
        y_max += pad

    x_min = int(frame_ids[0])
    x_max = int(frame_ids[-1])
    x_span = max(1, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)

    def map_point(frame_id: int, value: float) -> tuple[float, float]:
        x = margin_left + (int(frame_id) - x_min) / x_span * plot_w
        y = margin_top + (y_max - float(value)) / y_span * plot_h
        return x, y

    axis_color = (60, 60, 60)
    grid_color = (225, 225, 225)
    text_color = (25, 25, 25)

    draw.rectangle(
        [(margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h)],
        outline=axis_color,
        width=1,
    )
    draw.text((margin_left, 20), title, fill=text_color)
    draw.text((margin_left, height - 42), "frame_id", fill=text_color)
    draw.text((18, margin_top + plot_h // 2), y_label, fill=text_color)

    for tick in range(6):
        x = margin_left + tick / 5 * plot_w
        frame_tick = int(round(x_min + tick / 5 * x_span))
        draw.line([(x, margin_top), (x, margin_top + plot_h)], fill=grid_color, width=1)
        draw.text((x - 28, margin_top + plot_h + 12), str(frame_tick), fill=text_color)

        y = margin_top + tick / 5 * plot_h
        value_tick = y_max - tick / 5 * y_span
        draw.line([(margin_left, y), (margin_left + plot_w, y)], fill=grid_color, width=1)
        draw.text((8, y - 8), f"{value_tick:.1f}", fill=text_color)

    segment: list[tuple[float, float]] = []
    for frame_id, value, is_valid in zip(frame_ids, values, valid):
        if is_valid and np.isfinite(value):
            segment.append(map_point(int(frame_id), float(value)))
        else:
            if len(segment) >= 2:
                draw.line(segment, fill=color, width=3)
            elif len(segment) == 1:
                x, y = segment[0]
                draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=color)
            segment = []
    if len(segment) >= 2:
        draw.line(segment, fill=color, width=3)
    elif len(segment) == 1:
        x, y = segment[0]
        draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=color)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_file)


def save_per_track_outputs(
    output_dir: Path,
    tracks_xy: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    source: np.ndarray,
    frame_ids: np.ndarray,
) -> None:
    for track_id in range(tracks_xy.shape[1]):
        track_dir = output_dir / f"point_{track_id:02d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        write_track_txt(
            track_dir / "track.txt",
            tracks_xy,
            valid,
            confidence,
            source,
            frame_ids,
            track_id,
        )
        color = track_color(track_id)
        save_curve_png(
            track_dir / "frame_id_x_curve.png",
            frame_ids,
            tracks_xy[:, track_id, 0],
            valid[:, track_id],
            title=f"point_{track_id:02d} x coordinate over time",
            y_label="x pixel",
            color=color,
        )
        save_curve_png(
            track_dir / "frame_id_y_curve.png",
            frame_ids,
            tracks_xy[:, track_id, 1],
            valid[:, track_id],
            title=f"point_{track_id:02d} y coordinate over time",
            y_label="y pixel",
            color=color,
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, type=str)
    parser.add_argument("--mask_dir", default=DEFAULT_MASK_DIR, type=str)
    parser.add_argument("--output_dir", default="long_time_track_output", type=str)
    parser.add_argument("--start_frame", default=501, type=int)
    parser.add_argument("--end_frame", default=2001, type=int)
    parser.add_argument("--instance_id", default=1, type=int, help="beanstalk instance id in mask PNG")
    parser.add_argument("--num_points", default=10, type=int, help="Number of initial points sampled in the first beanstalk mask")
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--max_candidates", default=20000, type=int)
    parser.add_argument("--setting", default="precise", choices=["turbo", "fast", "base", "precise"], type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--download_retries", default=3, type=int)
    parser.add_argument("--retry_sleep_sec", default=2.0, type=float)
    parser.add_argument("--correction_interval", default=25, type=int)
    parser.add_argument("--correction_weight", default=0.65, type=float)
    parser.add_argument("--max_correction_distance", default=80.0, type=float)
    parser.add_argument("--min_overlap", default=0.05, type=float)
    parser.add_argument("--overview_max_side", default=1600, type=int, help="Resize initial point overview image to this max side; <=0 keeps full size")
    parser.add_argument("--save_npz", action="store_true", help="Also save complete track arrays as tracks.npz")
    args = parser.parse_args()

    if args.end_frame <= args.start_frame:
        raise ValueError("--end_frame must be greater than --start_frame")
    if not 0.0 <= args.correction_weight <= 1.0:
        raise ValueError("--correction_weight must be in [0, 1]")

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[INFO] CUDA unavailable or driver mismatch detected; RoMaV2 will run on CPU (slower).")

    model = build_model_with_resilient_weights(
        checkpoint_path=args.checkpoint_path,
        download_retries=args.download_retries,
        retry_sleep_sec=args.retry_sleep_sec,
    )
    model.apply_setting(args.setting)
    model.eval()

    first_image, first_arr, first_mask = load_frame(image_dir, mask_dir, args.start_frame, args.instance_id)
    h, w = first_arr.shape[:2]
    current_points = choose_uniform_random_points(
        first_mask,
        num_points=args.num_points,
        seed=args.seed,
        max_candidates=args.max_candidates,
    )
    overview_file = output_dir / "initial_points.png"
    save_initial_points_image(
        overview_file,
        np.array(first_image.convert("RGB")),
        first_mask,
        current_points,
        max_side=args.overview_max_side,
    )
    print(f"[INFO] Saved initial point overview: {overview_file}")

    frame_ids = np.arange(args.start_frame, args.end_frame + 1, dtype=np.int32)
    tracks_xy = np.full((len(frame_ids), args.num_points, 2), np.nan, dtype=np.float32)
    valid = np.zeros((len(frame_ids), args.num_points), dtype=bool)
    confidence = np.zeros((len(frame_ids), args.num_points), dtype=np.float32)
    source = np.full((len(frame_ids), args.num_points), "lost", dtype=object)

    tracks_xy[0] = current_points
    valid[0] = True
    confidence[0] = 1.0
    source[0] = "initial"

    current_arr = first_arr
    active_valid = np.ones(args.num_points, dtype=bool)
    total_steps = args.end_frame - args.start_frame
    print(
        f"[INFO] Tracking {args.num_points} initial beanstalk points from "
        f"{frame_name(args.start_frame)} to {frame_name(args.end_frame)}"
    )
    print(f"[INFO] Correction interval: {args.correction_interval} frames")

    for step, frame_a in enumerate(range(args.start_frame, args.end_frame), start=1):
        frame_b = frame_a + 1
        _, next_arr, next_mask = load_frame(image_dir, mask_dir, frame_b, args.instance_id)

        propagation_input = np.nan_to_num(current_points, nan=0.0)
        with torch.inference_mode():
            adjacent_preds = model.match(current_arr, next_arr)
            propagated, prop_overlap = sample_warp_and_overlap(adjacent_preds, propagation_input, h, w)

        frame_valid = active_valid & in_mask(propagated, next_mask) & (prop_overlap >= args.min_overlap)
        frame_source = np.full(args.num_points, "propagated", dtype=object)
        frame_conf = prop_overlap.astype(np.float32)
        next_points = propagated.astype(np.float32)

        use_correction = args.correction_interval > 0 and step % args.correction_interval == 0
        if use_correction:
            with torch.inference_mode():
                correction_preds = model.match(first_arr, next_arr)
                corrected, corr_overlap = sample_warp_and_overlap(correction_preds, tracks_xy[0], h, w)

            corrected_valid = in_mask(corrected, next_mask) & (corr_overlap >= args.min_overlap)
            dist = np.linalg.norm(corrected - propagated, axis=1)
            close = dist <= args.max_correction_distance
            can_blend = frame_valid & corrected_valid & close
            can_replace = (~frame_valid) & corrected_valid

            next_points[can_blend] = (
                (1.0 - args.correction_weight) * propagated[can_blend]
                + args.correction_weight * corrected[can_blend]
            )
            frame_conf[can_blend] = np.maximum(frame_conf[can_blend], corr_overlap[can_blend])
            frame_source[can_blend] = "corrected_blend"

            next_points[can_replace] = corrected[can_replace]
            frame_conf[can_replace] = corr_overlap[can_replace]
            frame_valid[can_replace] = True
            frame_source[can_replace] = "corrected_replace"

            corrected_only = frame_valid & corrected_valid & (~close) & (corr_overlap > frame_conf)
            next_points[corrected_only] = corrected[corrected_only]
            frame_conf[corrected_only] = corr_overlap[corrected_only]
            frame_source[corrected_only] = "corrected_jump"

        frame_valid = frame_valid & in_mask(next_points, next_mask)
        next_points[~frame_valid] = np.nan
        frame_source[~frame_valid] = "lost"

        tracks_xy[step] = next_points
        valid[step] = frame_valid
        confidence[step] = frame_conf
        source[step] = frame_source

        current_points = np.where(frame_valid[:, None], next_points, np.nan)
        active_valid = frame_valid
        current_arr = next_arr

        if step == 1 or step == total_steps or step % 10 == 0:
            print(
                f"[{step}/{total_steps}] {frame_name(frame_a)}->{frame_name(frame_b)} "
                f"valid={int(frame_valid.sum())}/{args.num_points}"
            )

    save_per_track_outputs(output_dir, tracks_xy, valid, confidence, source, frame_ids)
    print(f"[INFO] Saved per-point track folders under: {output_dir}")

    if args.save_npz:
        npz_file = output_dir / "tracks.npz"
        np.savez_compressed(
            npz_file,
            frame_ids=frame_ids,
            tracks_xy=tracks_xy,
            valid=valid,
            confidence=confidence,
            source=source.astype(str),
            initial_points=tracks_xy[0],
            start_frame=np.array(args.start_frame, dtype=np.int32),
            end_frame=np.array(args.end_frame, dtype=np.int32),
            instance_id=np.array(args.instance_id, dtype=np.int32),
        )
        print(f"[INFO] Saved track arrays: {npz_file}")


if __name__ == "__main__":
    main()
