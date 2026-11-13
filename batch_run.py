from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from npz2png import draw_matches
from romav2 import RoMaV2
from run import build_model_with_resilient_weights


DEFAULT_IMAGE_DIR = "/lustre/home/acct-cwj/cwj-user1/sam3-main/assets/videos/3072-3072"
DEFAULT_MASK_DIR = "/lustre/home/acct-cwj/cwj-user1/sam3-main/assets/videos/output/masks/combined_instance"


def frame_name(frame_id: int) -> str:
    return f"frame_{frame_id:06d}"


def image_path(image_dir: Path, frame_id: int) -> Path:
    return image_dir / f"{frame_name(frame_id)}.png"


def mask_path(mask_dir: Path, frame_id: int) -> Path:
    return mask_dir / f"{frame_name(frame_id)}_instance_mask.png"


def load_instance_mask(
    mask_file: Path, instance_id: int, expected_size: tuple[int, int], *, allow_empty: bool = False
) -> np.ndarray:
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask not found: {mask_file}")

    mask_arr = np.array(Image.open(mask_file))
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]

    expected_w, expected_h = expected_size
    if mask_arr.shape[:2] != (expected_h, expected_w):
        raise ValueError(
            f"Mask size mismatch for {mask_file}: got {mask_arr.shape[:2]}, "
            f"expected {(expected_h, expected_w)}"
        )

    mask = (mask_arr == instance_id).astype(np.uint8)
    if not allow_empty and not mask.any():
        raise ValueError(f"Instance id {instance_id} is empty in mask: {mask_file}")
    return mask


def load_masked_image(
    image_file: Path, mask_file: Path, instance_id: int, *, allow_empty: bool = False
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    image = Image.open(image_file).convert("RGB")
    mask = load_instance_mask(mask_file, instance_id, image.size, allow_empty=allow_empty)
    arr = np.array(image)
    arr[mask == 0] = 0
    return image, arr, mask


def keep_matches_in_masks(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> np.ndarray:
    h_a, w_a = mask_a.shape[:2]
    h_b, w_b = mask_b.shape[:2]
    keep = np.zeros(len(xy_a), dtype=bool)

    for i, ((xa, ya), (xb, yb)) in enumerate(zip(xy_a, xy_b)):
        xa_i, ya_i = int(round(float(xa))), int(round(float(ya)))
        xb_i, yb_i = int(round(float(xb))), int(round(float(yb)))
        inside_a = 0 <= ya_i < h_a and 0 <= xa_i < w_a and mask_a[ya_i, xa_i] > 0
        inside_b = 0 <= yb_i < h_b and 0 <= xb_i < w_b and mask_b[yb_i, xb_i] > 0
        keep[i] = inside_a and inside_b

    return keep


def run_pair(
    *,
    model: RoMaV2,
    image_dir: Path,
    mask_dir: Path,
    npz_dir: Path,
    png_dir: Path,
    frame_a: int,
    frame_b: int,
    instance_id: int,
    num_matches: int,
    max_draw: int,
    point_radius: int,
    line_width: int,
    save_masks_in_npz: bool,
) -> int:
    im_a_path = image_path(image_dir, frame_a)
    im_b_path = image_path(image_dir, frame_b)
    mask_a_path = mask_path(mask_dir, frame_a)
    mask_b_path = mask_path(mask_dir, frame_b)

    im_a, arr_a, mask_a = load_masked_image(im_a_path, mask_a_path, instance_id)
    im_b, arr_b, mask_b = load_masked_image(im_b_path, mask_b_path, instance_id)

    with torch.inference_mode():
        preds = model.match(arr_a, arr_b)
        matches, conf, _, _ = model.sample(preds, num_corresp=num_matches)

    h_a, w_a = arr_a.shape[:2]
    h_b, w_b = arr_b.shape[:2]
    xy_a, xy_b = RoMaV2.to_pixel_coordinates(matches, H_A=h_a, W_A=w_a, H_B=h_b, W_B=w_b)
    xy_a = xy_a.cpu().numpy()
    xy_b = xy_b.cpu().numpy()
    conf = conf.cpu().numpy()

    keep = keep_matches_in_masks(xy_a, xy_b, mask_a, mask_b)
    xy_a = xy_a[keep]
    xy_b = xy_b[keep]
    conf = conf[keep]

    stem = f"{frame_name(frame_a)}_{frame_name(frame_b)}"
    npz_file = npz_dir / f"{stem}.npz"
    png_file = png_dir / f"{stem}.png"

    payload = {
        "xy_A": xy_a,
        "xy_B": xy_b,
        "confidence": conf,
        "frame_A": np.array(frame_a, dtype=np.int32),
        "frame_B": np.array(frame_b, dtype=np.int32),
        "instance_id": np.array(instance_id, dtype=np.int32),
        "image_A_path": np.array(str(im_a_path)),
        "image_B_path": np.array(str(im_b_path)),
        "mask_A_path": np.array(str(mask_a_path)),
        "mask_B_path": np.array(str(mask_b_path)),
    }
    if save_masks_in_npz:
        payload["mask_A"] = mask_a
        payload["mask_B"] = mask_b

    np.savez_compressed(npz_file, **payload)

    vis = draw_matches(
        image_a=im_a,
        image_b=im_b,
        xy_a=xy_a,
        xy_b=xy_b,
        confidence=conf,
        max_draw=max_draw,
        point_radius=point_radius,
        line_width=line_width,
    )
    vis.save(png_file)

    return len(xy_a)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, type=str)
    parser.add_argument("--mask_dir", default=DEFAULT_MASK_DIR, type=str)
    parser.add_argument("--output_dir", default="batch_output", type=str)
    parser.add_argument("--start_frame", default=501, type=int)
    parser.add_argument("--end_frame", default=2001, type=int)
    parser.add_argument("--instance_id", default=1, type=int, help="beanstalk instance id in mask PNG")
    parser.add_argument("--num_matches", default=2000, type=int)
    parser.add_argument("--setting", default="precise", choices=["turbo", "fast", "base", "precise"], type=str)
    parser.add_argument("--max_draw", default=500, type=int, help="How many matches to draw per PNG (<=0 means all)")
    parser.add_argument("--point_radius", default=3, type=int)
    parser.add_argument("--line_width", default=1, type=int)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--download_retries", default=3, type=int)
    parser.add_argument("--retry_sleep_sec", default=2.0, type=float)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--save_masks_in_npz", action="store_true", help="Store full masks in every NPZ; disabled by default to save disk")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    npz_dir = output_dir / "npz"
    png_dir = output_dir / "png"
    npz_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    if args.end_frame <= args.start_frame:
        raise ValueError("--end_frame must be greater than --start_frame")

    if not torch.cuda.is_available():
        print("[INFO] CUDA unavailable or driver mismatch detected; RoMaV2 will run on CPU (slower).")

    model = build_model_with_resilient_weights(
        checkpoint_path=args.checkpoint_path,
        download_retries=args.download_retries,
        retry_sleep_sec=args.retry_sleep_sec,
    )
    model.apply_setting(args.setting)
    model.eval()

    total_pairs = args.end_frame - args.start_frame
    print(
        f"[INFO] Processing {total_pairs} adjacent pairs: "
        f"{frame_name(args.start_frame)}-{frame_name(args.start_frame + 1)} ... "
        f"{frame_name(args.end_frame - 1)}-{frame_name(args.end_frame)}"
    )
    print(f"[INFO] Output NPZ: {npz_dir}")
    print(f"[INFO] Output PNG: {png_dir}")

    for idx, frame_a in enumerate(range(args.start_frame, args.end_frame), start=1):
        frame_b = frame_a + 1
        stem = f"{frame_name(frame_a)}_{frame_name(frame_b)}"
        npz_file = npz_dir / f"{stem}.npz"
        png_file = png_dir / f"{stem}.png"
        if args.skip_existing and npz_file.exists() and png_file.exists():
            print(f"[{idx}/{total_pairs}] Skip existing {stem}")
            continue

        try:
            kept = run_pair(
                model=model,
                image_dir=image_dir,
                mask_dir=mask_dir,
                npz_dir=npz_dir,
                png_dir=png_dir,
                frame_a=frame_a,
                frame_b=frame_b,
                instance_id=args.instance_id,
                num_matches=args.num_matches,
                max_draw=args.max_draw,
                point_radius=args.point_radius,
                line_width=args.line_width,
                save_masks_in_npz=args.save_masks_in_npz,
            )
            print(f"[{idx}/{total_pairs}] Saved {stem}: {kept} beanstalk matches")
        except Exception as exc:
            print(f"[{idx}/{total_pairs}] [ERROR] {stem}: {type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
