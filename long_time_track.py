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
    num_candidates = min(max(max_candidates, num_points), len(xs))
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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.astype(np.float32)

    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.astype(np.float32)

    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")


def axis_monotonic_score(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    smooth_window: int,
    local_tolerance: float,
    min_axis_displacement: float,
) -> tuple[float, float, int, float, float]:
    use = valid & np.isfinite(values)
    series = values[use].astype(np.float32)
    if len(series) < 3:
        return 0.0, 0.0, 0, 0.0, 1.0

    smooth = moving_average(series, smooth_window)
    tail = max(1, min(10, len(smooth) // 5))
    start_value = float(np.median(smooth[:tail]))
    end_value = float(np.median(smooth[-tail:]))
    net = end_value - start_value
    displacement = abs(net)
    if displacement < min_axis_displacement:
        return 1.0, 1.0, 0, displacement, 0.0

    direction = 1 if net > 0 else -1
    signed_diffs = direction * np.diff(smooth)
    forward = float(np.clip(signed_diffs, 0.0, None).sum())
    backward = float(np.clip(-signed_diffs, 0.0, None).sum())
    backward_ratio = backward / max(1e-6, forward + backward)
    monotonic_score = max(0.0, 1.0 - backward_ratio)
    consistency = float(np.mean(signed_diffs >= -local_tolerance)) if len(signed_diffs) else 1.0
    return monotonic_score, consistency, direction, displacement, backward_ratio


def evaluate_track_monotonicity(
    tracks_xy: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    *,
    min_valid_ratio: float,
    min_monotonic_score: float,
    min_direction_consistency: float,
    smooth_window: int,
    local_tolerance: float,
    min_axis_displacement: float,
) -> list[dict[str, float | int | bool]]:
    results: list[dict[str, float | int | bool]] = []
    num_frames, num_tracks = valid.shape
    for track_id in range(num_tracks):
        track_valid = valid[:, track_id]
        valid_ratio = float(track_valid.sum() / max(1, num_frames))
        mean_confidence = float(np.mean(confidence[track_valid, track_id])) if track_valid.any() else 0.0

        x_score, x_consistency, x_direction, x_disp, x_backward = axis_monotonic_score(
            tracks_xy[:, track_id, 0],
            track_valid,
            smooth_window=smooth_window,
            local_tolerance=local_tolerance,
            min_axis_displacement=min_axis_displacement,
        )
        y_score, y_consistency, y_direction, y_disp, y_backward = axis_monotonic_score(
            tracks_xy[:, track_id, 1],
            track_valid,
            smooth_window=smooth_window,
            local_tolerance=local_tolerance,
            min_axis_displacement=min_axis_displacement,
        )

        is_valid_track = (
            valid_ratio >= min_valid_ratio
            and x_score >= min_monotonic_score
            and y_score >= min_monotonic_score
            and x_consistency >= min_direction_consistency
            and y_consistency >= min_direction_consistency
        )
        rank_score = (
            0.45 * ((x_score + y_score) / 2.0)
            + 0.35 * valid_ratio
            + 0.20 * min(1.0, mean_confidence)
        )
        results.append(
            {
                "track_id": track_id,
                "is_valid": is_valid_track,
                "rank_score": rank_score,
                "valid_ratio": valid_ratio,
                "mean_confidence": mean_confidence,
                "x_score": x_score,
                "y_score": y_score,
                "x_consistency": x_consistency,
                "y_consistency": y_consistency,
                "x_direction": x_direction,
                "y_direction": y_direction,
                "x_displacement": x_disp,
                "y_displacement": y_disp,
                "x_backward_ratio": x_backward,
                "y_backward_ratio": y_backward,
            }
        )
    return results


def save_selected_track_outputs(
    output_dir: Path,
    selected_track_ids: list[int],
    tracks_xy: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    source: np.ndarray,
    frame_ids: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for track_id in selected_track_ids:
        track_dir = output_dir / f"point_{track_id:02d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        write_track_txt(track_dir / "track.txt", tracks_xy, valid, confidence, source, frame_ids, track_id)
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


def write_valid_summary(output_file: Path, results: list[dict[str, float | int | bool]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "track_id",
        "is_valid",
        "rank_score",
        "valid_ratio",
        "mean_confidence",
        "x_score",
        "y_score",
        "x_consistency",
        "y_consistency",
        "x_direction",
        "y_direction",
        "x_displacement",
        "y_displacement",
        "x_backward_ratio",
        "y_backward_ratio",
    ]
    with output_file.open("w", encoding="utf-8") as f:
        f.write(" ".join(keys) + "\n")
        for item in sorted(results, key=lambda x: int(x["track_id"])):
            f.write(" ".join(str(item[key]) for key in keys) + "\n")


def select_video_track_ids(
    valid_results: list[dict[str, float | int | bool]],
    initial_points: np.ndarray,
    *,
    top_k: int,
    image_width: int,
    image_height: int,
    min_monotonic_score: float,
    min_direction_consistency: float,
    diversity_weight: float,
    candidate_multiplier: int,
) -> list[int]:
    if top_k <= 0:
        return []

    strict_candidates = [
        item
        for item in valid_results
        if float(item["x_score"]) >= min_monotonic_score
        and float(item["y_score"]) >= min_monotonic_score
        and float(item["x_consistency"]) >= min_direction_consistency
        and float(item["y_consistency"]) >= min_direction_consistency
    ]
    strict_candidates.sort(key=lambda item: float(item["rank_score"]), reverse=True)

    if candidate_multiplier > 0:
        pool_size = max(top_k, top_k * candidate_multiplier)
        strict_candidates = strict_candidates[:pool_size]

    if len(strict_candidates) <= top_k:
        return [int(item["track_id"]) for item in strict_candidates]

    track_ids = np.array([int(item["track_id"]) for item in strict_candidates], dtype=np.int32)
    quality = np.array([float(item["rank_score"]) for item in strict_candidates], dtype=np.float32)
    quality_span = float(quality.max() - quality.min())
    if quality_span > 1e-6:
        quality_norm = (quality - quality.min()) / quality_span
    else:
        quality_norm = np.ones_like(quality)

    scale = np.array([max(1, image_width), max(1, image_height)], dtype=np.float32)
    positions = initial_points[track_ids] / scale

    selected_indices = [int(np.argmax(quality_norm))]
    remaining = set(range(len(track_ids)))
    remaining.remove(selected_indices[0])
    diversity_weight = float(np.clip(diversity_weight, 0.0, 1.0))

    while len(selected_indices) < top_k and remaining:
        selected_pos = positions[selected_indices]
        best_idx = None
        best_score = -np.inf
        for idx in remaining:
            min_dist = float(np.linalg.norm(positions[idx][None] - selected_pos, axis=1).min())
            combined_score = (1.0 - diversity_weight) * float(quality_norm[idx]) + diversity_weight * min_dist
            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx
        assert best_idx is not None
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [int(track_ids[idx]) for idx in selected_indices]


def write_video_selection_summary(
    output_file: Path,
    selected_track_ids: list[int],
    results: list[dict[str, float | int | bool]],
) -> None:
    result_by_id = {int(item["track_id"]): item for item in results}
    keys = [
        "track_id",
        "rank_score",
        "valid_ratio",
        "mean_confidence",
        "x_score",
        "y_score",
        "x_consistency",
        "y_consistency",
        "x_backward_ratio",
        "y_backward_ratio",
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(" ".join(keys) + "\n")
        for track_id in selected_track_ids:
            item = result_by_id[track_id]
            f.write(" ".join(str(item[key]) for key in keys) + "\n")


def draw_tracks_on_frame(
    image: Image.Image,
    frame_idx: int,
    selected_track_ids: list[int],
    tracks_xy: np.ndarray,
    valid: np.ndarray,
    max_side: int,
) -> Image.Image:
    vis = image.convert("RGB")
    scale = 1.0
    if max_side > 0 and max(vis.size) > max_side:
        scale = max_side / max(vis.size)
        new_size = (int(round(vis.size[0] * scale)), int(round(vis.size[1] * scale)))
        vis = vis.resize(new_size, Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(vis)
    radius = max(4, int(round(max(vis.size) / 260)))
    for track_id in selected_track_ids:
        if not valid[frame_idx, track_id]:
            continue
        x, y = tracks_xy[frame_idx, track_id]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x *= scale
        y *= scale
        color = track_color(track_id)
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=color,
            outline=(255, 255, 255),
            width=max(1, radius // 3),
        )
        draw.text((x + radius + 2, y - radius - 2), str(track_id), fill=(255, 255, 255))
    return vis


def save_valid_video(
    output_file: Path,
    image_dir: Path,
    frame_ids: np.ndarray,
    selected_track_ids: list[int],
    tracks_xy: np.ndarray,
    valid: np.ndarray,
    *,
    fps: float,
    max_side: int,
    stride: int,
) -> Path | None:
    if not selected_track_ids:
        return None

    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        first_frame_id = int(frame_ids[0])
        first_image = Image.open(image_path(image_dir, first_frame_id)).convert("RGB")
        first_vis = draw_tracks_on_frame(first_image, 0, selected_track_ids, tracks_xy, valid, max_side)
        first = np.array(first_vis)
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(output_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for frame_idx in range(max(1, stride), len(frame_ids), max(1, stride)):
            frame_id = int(frame_ids[frame_idx])
            image = Image.open(image_path(image_dir, frame_id)).convert("RGB")
            frame = draw_tracks_on_frame(image, frame_idx, selected_track_ids, tracks_xy, valid, max_side)
            writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        writer.release()
        return output_file
    except Exception as exc:
        frames = []
        for frame_idx in range(0, len(frame_ids), max(1, stride)):
            frame_id = int(frame_ids[frame_idx])
            image = Image.open(image_path(image_dir, frame_id)).convert("RGB")
            frames.append(draw_tracks_on_frame(image, frame_idx, selected_track_ids, tracks_xy, valid, max_side))
        gif_file = output_file.with_suffix(".gif")
        duration_ms = int(round(1000.0 / max(1e-6, fps)))
        frames[0].save(
            gif_file,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"[WARN] MP4 writing failed ({type(exc).__name__}: {exc}); saved GIF instead: {gif_file}")
        return gif_file


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
    parser.add_argument("--min_valid_ratio", default=0.70, type=float, help="Minimum valid-frame ratio for valid tracks")
    parser.add_argument("--min_monotonic_score", default=0.88, type=float, help="Minimum overall monotonic score for x and y curves")
    parser.add_argument("--min_direction_consistency", default=0.80, type=float, help="Minimum fraction of locally monotonic steps after smoothing")
    parser.add_argument("--monotonic_smooth_window", default=9, type=int, help="Moving-average window used before monotonic filtering")
    parser.add_argument("--local_tolerance", default=2.0, type=float, help="Pixel tolerance for small local reverse motion")
    parser.add_argument("--min_axis_displacement", default=5.0, type=float, help="Treat an axis as stable if robust displacement is below this")
    parser.add_argument("--top_video_points", default=32, type=int)
    parser.add_argument("--video_min_monotonic_score", default=0.92, type=float, help="Stricter x/y monotonic score required only for video points")
    parser.add_argument("--video_min_direction_consistency", default=0.88, type=float, help="Stricter x/y local consistency required only for video points")
    parser.add_argument("--video_diversity_weight", default=0.35, type=float, help="Higher values spread video points more across the first-frame ROI")
    parser.add_argument("--video_candidate_multiplier", default=8, type=int, help="Diversity selection pool size is top_video_points times this; <=0 uses all strict candidates")
    parser.add_argument("--video_fps", default=20.0, type=float)
    parser.add_argument("--video_max_side", default=1280, type=int)
    parser.add_argument("--video_stride", default=1, type=int, help="Use every Nth frame in the visualization video")
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

    monotonic_results = evaluate_track_monotonicity(
        tracks_xy,
        valid,
        confidence,
        min_valid_ratio=args.min_valid_ratio,
        min_monotonic_score=args.min_monotonic_score,
        min_direction_consistency=args.min_direction_consistency,
        smooth_window=args.monotonic_smooth_window,
        local_tolerance=args.local_tolerance,
        min_axis_displacement=args.min_axis_displacement,
    )
    valid_results = [item for item in monotonic_results if bool(item["is_valid"])]
    valid_results.sort(key=lambda item: float(item["rank_score"]), reverse=True)
    valid_track_ids = [int(item["track_id"]) for item in valid_results]

    valid_dir = output_dir / "valid"
    save_selected_track_outputs(valid_dir, valid_track_ids, tracks_xy, valid, confidence, source, frame_ids)
    write_valid_summary(valid_dir / "valid_summary.txt", monotonic_results)
    print(f"[INFO] Valid monotonic tracks: {len(valid_track_ids)}/{args.num_points}")
    print(f"[INFO] Saved valid track folders and summary under: {valid_dir}")

    video_track_ids = select_video_track_ids(
        valid_results,
        tracks_xy[0],
        top_k=max(0, args.top_video_points),
        image_width=w,
        image_height=h,
        min_monotonic_score=args.video_min_monotonic_score,
        min_direction_consistency=args.video_min_direction_consistency,
        diversity_weight=args.video_diversity_weight,
        candidate_multiplier=args.video_candidate_multiplier,
    )
    write_video_selection_summary(valid_dir / "video_selection_summary.txt", video_track_ids, monotonic_results)
    print(
        f"[INFO] Video tracks selected after strict x/y monotonic and diversity filtering: "
        f"{len(video_track_ids)}/{args.top_video_points}"
    )
    video_path = save_valid_video(
        valid_dir / "top_valid_tracks.mp4",
        image_dir,
        frame_ids,
        video_track_ids,
        tracks_xy,
        valid,
        fps=args.video_fps,
        max_side=args.video_max_side,
        stride=args.video_stride,
    )
    if video_path is not None:
        print(f"[INFO] Saved valid-track visualization video: {video_path}")
    else:
        print("[INFO] No valid tracks available for visualization video.")

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
            monotonic_results=np.array([str(item) for item in monotonic_results]),
            valid_track_ids=np.array(valid_track_ids, dtype=np.int32),
        )
        print(f"[INFO] Saved track arrays: {npz_file}")


if __name__ == "__main__":
    main()
