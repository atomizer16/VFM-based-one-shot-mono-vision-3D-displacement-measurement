#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM 3 video mask-prompt tracking.

Fill the CONFIG section, or pass the same values from the command line.
The mask prompts can be binary masks or instance/class-id masks. For class-id
masks, TARGET_MASK_IDS selects the objects to track.
"""

import argparse
import os
# Must be set before importing torch. Helps long video runs avoid allocator
# fragmentation when masks/features are repeatedly allocated and released.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


# ============================== CONFIG ==============================

# "video" for one complete video file, "frames" for an image-frame directory.
INPUT_MODE = "video"

# Mode 1: complete video path, for example r"D:\data\beanstalk.mp4".
VIDEO_PATH = r""

# Mode 2: directory containing frame_0000001.jpg, frame_0000002.jpg, ...
FRAMES_DIR = r""

# Mask manually annotated on the first frame. Supports binary masks and
# instance/class-id masks. For your LabelMe conversion, beanstalk should be id 1.
FIRST_FRAME_MASK_PATH = r""

# Optional correction masks for long videos. If CORRECTION_MASK_DIR is set, the
# script will load masks whose frame number matches input frames. You can either
# list exact frame numbers, e.g. "601,701,801,901,1001", or leave it empty to
# use every image mask found in the correction directory.
CORRECTION_MASK_DIR = r""
CORRECTION_FRAME_NUMBERS = r""
CORRECTION_MASK_PATHS = r""

# Output directory. It will contain masks/, overlays/, and optionally overlay.mp4.
OUTPUT_DIR = r"./sam3_maskprompt_outputs"

# Use one GPU for the direct mask-prompt path. Multi-GPU high-level requests do
# not expose mask prompts in this SAM 3 version.
GPU_ID = 0

# Backward-compatible single-object id. For multi-object instance masks, object
# ids default to the same values as TARGET_MASK_IDS.
OBJ_ID = 1

# If your mask file is white background and black object, set this to True.
INVERT_MASK = False

# "auto" supports both binary masks and class-id masks. Use "label_id" to force
# extracting TARGET_MASK_IDS, or "binary" to force non-zero/threshold extraction.
MASK_MODE = "auto"
TARGET_MASK_ID = 1
TARGET_MASK_IDS = "1,2,3"
TARGET_NAMES = "beanstalk,aruco1,aruco2"
OBJ_IDS = ""

# Keep only the largest connected component in the input mask.
KEEP_LARGEST_COMPONENT = True

# Set Hugging Face offline/cache options if you use local model files.
HF_HUB_OFFLINE = "1"
HF_HOME = r""

# If input is a frame directory, this FPS is used for overlay.mp4.
FRAMES_MODE_FPS = 30.0

# Whether to write an overlay video in addition to per-frame PNG files.
WRITE_OVERLAY_VIDEO = True
VISUAL_FRAME_COUNT = 10

# Keep decoded/normalized video frames on CPU. This is important for long,
# high-resolution sequences on 32 GB GPUs.
OFFLOAD_VIDEO_TO_CPU = True
ASYNC_LOADING_FRAMES = False

# During propagation, SAM3 caches video-resolution masks on GPU. Clear each
# frame's cache after copying the postprocessed numpy output to avoid OOM.
CLEAR_FRAME_CACHE_DURING_PROPAGATION = True
CLEAR_CUDA_CACHE_EVERY_N_FRAMES = 10

# ============================ IO helpers ============================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use binary or class-id ROI masks as SAM 3 video prompts."
    )
    parser.add_argument("--input-mode", choices=["video", "frames"], default=INPUT_MODE)
    parser.add_argument("--video-path", default=VIDEO_PATH)
    parser.add_argument("--frames-dir", default=FRAMES_DIR)
    parser.add_argument("--mask-path", default=FIRST_FRAME_MASK_PATH)
    parser.add_argument("--correction-mask-dir", default=CORRECTION_MASK_DIR)
    parser.add_argument("--correction-frame-numbers", default=CORRECTION_FRAME_NUMBERS)
    parser.add_argument("--correction-mask-paths", default=CORRECTION_MASK_PATHS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--gpu-id", type=int, default=GPU_ID)
    parser.add_argument("--obj-id", type=int, default=OBJ_ID)
    parser.add_argument("--obj-ids", default=OBJ_IDS)
    parser.add_argument("--invert-mask", action="store_true", default=INVERT_MASK)
    parser.add_argument(
        "--mask-mode",
        choices=["auto", "binary", "label_id"],
        default=MASK_MODE,
    )
    parser.add_argument("--target-mask-id", type=int, default=TARGET_MASK_ID)
    parser.add_argument("--target-mask-ids", default=TARGET_MASK_IDS)
    parser.add_argument("--target-names", default=TARGET_NAMES)
    parser.add_argument(
        "--keep-largest-component",
        action=argparse.BooleanOptionalAction,
        default=KEEP_LARGEST_COMPONENT,
    )
    parser.add_argument("--hf-home", default=HF_HOME)
    parser.add_argument("--hf-offline", default=HF_HUB_OFFLINE)
    parser.add_argument("--frames-mode-fps", type=float, default=FRAMES_MODE_FPS)
    parser.add_argument(
        "--write-overlay-video",
        action=argparse.BooleanOptionalAction,
        default=WRITE_OVERLAY_VIDEO,
    )
    parser.add_argument("--visual-frame-count", type=int, default=VISUAL_FRAME_COUNT)
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=OFFLOAD_VIDEO_TO_CPU,
    )
    parser.add_argument(
        "--async-loading-frames",
        action=argparse.BooleanOptionalAction,
        default=ASYNC_LOADING_FRAMES,
    )
    parser.add_argument(
        "--clear-frame-cache-during-propagation",
        action=argparse.BooleanOptionalAction,
        default=CLEAR_FRAME_CACHE_DURING_PROPAGATION,
    )
    parser.add_argument(
        "--clear-cuda-cache-every-n-frames",
        type=int,
        default=CLEAR_CUDA_CACHE_EVERY_N_FRAMES,
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = str(args.hf_offline)
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home


def resolve_resource_path(args: argparse.Namespace) -> Path:
    if args.input_mode == "video":
        path = Path(args.video_path)
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            raise FileNotFoundError(f"Invalid video path: {path}")
        return path

    path = Path(args.frames_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Invalid frame directory: {path}")
    if not list_frame_paths(path):
        raise FileNotFoundError(f"No image frames found in: {path}")
    return path

def start_session_with_options(predictor, resource_path: Path, args: argparse.Namespace) -> str:
    session_id = str(uuid.uuid4())
    inference_state = predictor.model.init_state(
        resource_path=str(resource_path),
        offload_video_to_cpu=args.offload_video_to_cpu,
        async_loading_frames=args.async_loading_frames,
        video_loader_type=getattr(predictor, "video_loader_type", "cv2"),
    )
    predictor._ALL_INFERENCE_STATES[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
    }
    return session_id

def frame_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits), path.name
    return 10**18, path.name


def list_frame_paths(frame_dir: Path) -> List[Path]:
    paths = [p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths, key=frame_sort_key)


def parse_frame_number(path_or_name) -> Optional[int]:
    stem = Path(path_or_name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def get_frame_stems(resource_path: Path, input_mode: str, num_frames: int) -> List[str]:
    if input_mode == "frames":
        return [path.stem for path in list_frame_paths(resource_path)]
    return [f"frame_{idx + 1:07d}" for idx in range(num_frames)]


def build_frame_number_to_index(frame_stems: List[str]) -> Dict[int, int]:
    frame_number_to_index = {}
    for idx, stem in enumerate(frame_stems):
        frame_number = parse_frame_number(stem)
        if frame_number is None:
            continue
        if frame_number in frame_number_to_index:
            raise ValueError(f"Duplicate frame number in input frames: {frame_number}")
        frame_number_to_index[frame_number] = idx
    return frame_number_to_index


def parse_int_list(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_target_specs(args: argparse.Namespace) -> List[dict]:
    target_mask_ids = parse_int_list(args.target_mask_ids)
    if not target_mask_ids:
        target_mask_ids = [args.target_mask_id]

    obj_ids = parse_int_list(args.obj_ids)
    if not obj_ids:
        obj_ids = target_mask_ids.copy()
    if len(obj_ids) != len(target_mask_ids):
        raise ValueError("--obj-ids must have the same length as --target-mask-ids.")

    target_names = parse_str_list(args.target_names)
    if target_names and len(target_names) != len(target_mask_ids):
        raise ValueError("--target-names must have the same length as --target-mask-ids.")
    if not target_names:
        target_names = [f"target_{target_id}" for target_id in target_mask_ids]

    specs = []
    for target_mask_id, obj_id, target_name in zip(target_mask_ids, obj_ids, target_names):
        specs.append(
            {
                "target_mask_id": int(target_mask_id),
                "obj_id": int(obj_id),
                "name": target_name,
            }
        )
    return specs


def parse_path_list(raw: str) -> List[Path]:
    if not raw:
        return []
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def find_mask_for_frame_number(mask_dir: Path, frame_number: int) -> Path:
    candidates = [
        mask_dir / f"frame_{frame_number:06d}.png",
        mask_dir / f"frame_{frame_number:07d}.png",
        mask_dir / f"{frame_number:06d}.png",
        mask_dir / f"{frame_number:07d}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matching_paths = [
        path
        for path in mask_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTS and parse_frame_number(path) == frame_number
    ]
    if len(matching_paths) == 1:
        return matching_paths[0]
    if len(matching_paths) > 1:
        names = ", ".join(path.name for path in sorted(matching_paths))
        raise ValueError(f"Multiple masks match frame {frame_number}: {names}")
    raise FileNotFoundError(f"No correction mask found for frame {frame_number} in {mask_dir}")


def resolve_mask_schedule(
    args: argparse.Namespace,
    resource_path: Path,
    input_mode: str,
    num_frames: int,
) -> List[Tuple[int, Path]]:
    frame_stems = get_frame_stems(resource_path, input_mode, num_frames)
    frame_number_to_index = build_frame_number_to_index(frame_stems)

    first_mask_path = Path(args.mask_path)
    if not first_mask_path.is_file():
        raise FileNotFoundError(f"Invalid first-frame mask path: {first_mask_path}")

    first_frame_number = parse_frame_number(first_mask_path)
    first_frame_idx = 0
    if first_frame_number in frame_number_to_index:
        first_frame_idx = frame_number_to_index[first_frame_number]

    schedule_by_idx = {first_frame_idx: first_mask_path}

    for mask_path in parse_path_list(args.correction_mask_paths):
        if not mask_path.is_file():
            raise FileNotFoundError(f"Invalid correction mask path: {mask_path}")
        frame_number = parse_frame_number(mask_path)
        if frame_number is None or frame_number not in frame_number_to_index:
            raise ValueError(
                f"Cannot map correction mask to an input frame by filename: {mask_path}"
            )
        schedule_by_idx[frame_number_to_index[frame_number]] = mask_path

    correction_mask_dir = Path(args.correction_mask_dir) if args.correction_mask_dir else None
    if args.correction_frame_numbers and correction_mask_dir is None:
        raise ValueError("--correction-frame-numbers requires --correction-mask-dir.")

    if correction_mask_dir is not None:
        if not correction_mask_dir.is_dir():
            raise FileNotFoundError(f"Invalid correction mask directory: {correction_mask_dir}")

        correction_frame_numbers = parse_int_list(args.correction_frame_numbers)
        if correction_frame_numbers:
            for frame_number in correction_frame_numbers:
                if frame_number not in frame_number_to_index:
                    raise ValueError(f"Correction frame {frame_number} is not in input frames.")
                schedule_by_idx[frame_number_to_index[frame_number]] = find_mask_for_frame_number(
                    correction_mask_dir, frame_number
                )
        else:
            for mask_path in sorted(
                [p for p in correction_mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS],
                key=frame_sort_key,
            ):
                frame_number = parse_frame_number(mask_path)
                if frame_number in frame_number_to_index:
                    schedule_by_idx[frame_number_to_index[frame_number]] = mask_path

    schedule = sorted(schedule_by_idx.items())
    if not schedule:
        raise RuntimeError("No mask prompts were resolved.")
    return schedule


def iter_frames(
    resource_path: Path,
    input_mode: str,
) -> Iterable[Tuple[int, np.ndarray, str]]:
    if input_mode == "frames":
        for idx, path in enumerate(list_frame_paths(resource_path)):
            frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Failed to read frame: {path}")
            yield idx, frame_bgr, path.stem
        return

    cap = cv2.VideoCapture(str(resource_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {resource_path}")
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        yield idx, frame_bgr, f"frame_{idx + 1:07d}"
        idx += 1
    cap.release()


def get_video_fps(resource_path: Path, input_mode: str, fallback_fps: float) -> float:
    if input_mode == "frames":
        return fallback_fps
    cap = cv2.VideoCapture(str(resource_path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    if fps is None or fps <= 1e-3 or np.isnan(fps):
        return fallback_fps
    return float(fps)


def load_binary_mask(
    mask_path: Path,
    target_hw: Tuple[int, int],
    invert: bool,
    keep_largest_component: bool,
    mask_mode: str = "auto",
    target_mask_id: int = 1,
    allow_empty: bool = False,
) -> np.ndarray:
    raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")

    if raw.ndim == 3 and raw.shape[2] == 4 and np.any(raw[:, :, 3] < 255):
        gray = raw[:, :, 3]
    elif raw.ndim == 3:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw

    if gray.ndim != 2:
        raise ValueError(f"Unsupported mask shape {gray.shape}: {mask_path}")

    if gray.shape[:2] != target_hw:
        gray = cv2.resize(
            gray,
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    unique_values = np.unique(gray)
    nonzero_values = unique_values[unique_values != 0]
    if mask_mode == "auto":
        if (
            len(nonzero_values) > 0
            and gray.dtype.kind in ("u", "i")
            and int(nonzero_values.max()) <= 32
        ):
            selected_mask_mode = "label_id"
        else:
            selected_mask_mode = "binary"
    else:
        selected_mask_mode = mask_mode

    if selected_mask_mode == "label_id":
        mask = gray == target_mask_id
    elif selected_mask_mode == "binary":
        threshold = 0 if gray.max() <= 1 else 127
        mask = gray > threshold
    else:
        raise ValueError(f"Unsupported mask_mode={mask_mode}")

    if invert:
        mask = ~mask
    if keep_largest_component and mask.any():
        mask = keep_largest_component_only(mask)

    if not mask.any():
        if allow_empty:
            return mask
        values_preview = ",".join(str(int(v)) for v in unique_values[:20])
        raise ValueError(
            f"The mask is empty after extraction: {mask_path}; "
            f"mode={selected_mask_mode}, target_mask_id={target_mask_id}, "
            f"unique_values(first20)=[{values_preview}]"
        )
    return mask


def keep_largest_component_only(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 2:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    return labels == largest_label


# ========================== SAM 3 mask prompt ==========================

@torch.inference_mode()
def add_mask_prompt(
    predictor,
    session_id: str,
    frame_idx: int,
    obj_id: int,
    binary_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Add a binary mask prompt through SAM 3's internal tracker path."""
    if getattr(predictor, "world_size", 1) != 1:
        raise RuntimeError(
            "Mask prompt is used through an internal tracker API in this SAM 3 "
            "version, so run this script with a single GPU."
        )

    session = predictor._get_session(session_id)
    inference_state = session["state"]
    model = predictor.model

    if inference_state["tracker_metadata"] == {}:
        inference_state["tracker_metadata"].update(model._initialize_metadata())

    num_frames = inference_state["num_frames"]
    if not 0 <= frame_idx < num_frames:
        raise ValueError(f"frame_idx={frame_idx} is outside [0, {num_frames})")

    tracker_metadata = inference_state["tracker_metadata"]
    obj_rank = model._get_gpu_id_by_obj_id(inference_state, obj_id)

    model._prepare_backbone_feats(inference_state, frame_idx, reverse=False)

    if obj_rank is None:
        num_prev_obj = int(np.sum(tracker_metadata["num_obj_per_gpu"]))
        if num_prev_obj >= model.max_num_objects:
            raise RuntimeError(
                f"Cannot add object {obj_id}; max_num_objects={model.max_num_objects}."
            )

        obj_rank = model._assign_new_det_to_gpus(
            new_det_num=1,
            prev_workload_per_gpu=tracker_metadata["num_obj_per_gpu"],
        )[0]

        if model.rank == obj_rank:
            tracker_state = model._init_new_tracker_state(inference_state)
            inference_state["tracker_inference_states"].append(tracker_state)

        tracker_metadata["obj_ids_per_gpu"][obj_rank] = np.concatenate(
            [
                tracker_metadata["obj_ids_per_gpu"][obj_rank],
                np.array([obj_id], dtype=np.int64),
            ]
        )
        tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
            tracker_metadata["obj_ids_per_gpu"][obj_rank]
        )
        tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
            tracker_metadata["obj_ids_per_gpu"]
        )
        tracker_metadata["max_obj_id"] = max(tracker_metadata["max_obj_id"], obj_id)
        model.add_action_history(
            inference_state, "add", frame_idx=frame_idx, obj_ids=[obj_id]
        )
    else:
        if model.rank == obj_rank:
            tracker_states = model._get_tracker_inference_states_by_obj_ids(
                inference_state, [obj_id]
            )
            if len(tracker_states) != 1:
                raise RuntimeError(
                    f"Expected one tracker state for object {obj_id}, got {len(tracker_states)}."
                )
            tracker_state = tracker_states[0]
        model.add_action_history(
            inference_state, "refine", frame_idx=frame_idx, obj_ids=[obj_id]
        )

    tracker_metadata["obj_id_to_score"][obj_id] = 1.0
    tracker_metadata["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = 1.0

    if model.rank == 0:
        rank0_metadata = tracker_metadata.get("rank0_metadata", {})
        rank0_metadata.get("removed_obj_ids", set()).discard(obj_id)
        for hidden_ids in rank0_metadata.get("suppressed_obj_ids", {}).values():
            hidden_ids.discard(obj_id)
        ensure_confirmation_arrays(rank0_metadata, tracker_metadata, obj_id, model)

    mask_tensor = torch.as_tensor(binary_mask.astype(np.float32), device=model.device)
    if model.rank == obj_rank:
        _, obj_ids, _, video_res_masks = model.tracker.add_new_mask(
            inference_state=tracker_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask_tensor,
        )
        model.tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
        new_mask_data = (video_res_masks[obj_ids.index(obj_id)] > 0.0).to(torch.bool)
    else:
        new_mask_data = None

    if model.rank == 0:
        if new_mask_data is None:
            raise RuntimeError(f"No mask data returned for object {obj_id}.")

        # The interactivity propagation path merges Tracker masks into
        # cached_frame_outputs on every frame. Mask-only initialization has no
        # prior detector/VG cache, so create empty placeholders for all frames.
        for cache_frame_idx in range(num_frames):
            inference_state["cached_frame_outputs"].setdefault(cache_frame_idx, {})

        if frame_idx in inference_state["cached_frame_outputs"]:
            obj_id_to_mask = model._build_tracker_output(
                inference_state,
                frame_idx,
                {obj_id: new_mask_data},
            )
        else:
            obj_id_to_mask = {obj_id: new_mask_data}
        suppressed_obj_ids = tracker_metadata["rank0_metadata"]["suppressed_obj_ids"][
            frame_idx
        ]
        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": tracker_metadata["obj_id_to_score"],
            "obj_id_to_tracker_score": tracker_metadata[
                "obj_id_to_tracker_score_frame_wise"
            ][frame_idx],
        }
        model._cache_frame_outputs(
            inference_state,
            frame_idx,
            obj_id_to_mask,
            suppressed_obj_ids=suppressed_obj_ids,
        )
        inference_state["previous_stages_out"][frame_idx] = "_MASK_PROMPT_OUTPUT_"
        return model._postprocess_output(
            inference_state, out, suppressed_obj_ids=suppressed_obj_ids
        )

    return {}


def ensure_confirmation_arrays(rank0_metadata, tracker_metadata, obj_id: int, model) -> None:
    if "masklet_confirmation" not in rank0_metadata:
        return
    confirmation = rank0_metadata["masklet_confirmation"]
    target_len = len(tracker_metadata["obj_ids_all_gpu"])
    for key in ("status", "consecutive_det_num"):
        arr = confirmation[key]
        if len(arr) < target_len:
            pad = np.zeros(target_len - len(arr), dtype=np.int64)
            confirmation[key] = np.concatenate([arr, pad])

    obj_indices = np.where(tracker_metadata["obj_ids_all_gpu"] == obj_id)[0]
    if len(obj_indices) == 0:
        return
    obj_idx = int(obj_indices[0])
    confirmation["status"][obj_idx] = 1
    confirmation["consecutive_det_num"][
        obj_idx
    ] = model.masklet_confirmation_consecutive_det_thresh


def propagate_in_video(
    predictor,
    session_id: str,
    start_frame_idx: int,
    end_frame_idx: Optional[int] = None,
    clear_frame_cache: bool = True,
    clear_cuda_cache_every_n_frames: int = 10,
) -> Dict[int, dict]:
    outputs_per_frame = {}
    max_frame_num_to_track = None
    if end_frame_idx is not None:
        if end_frame_idx < start_frame_idx:
            return outputs_per_frame
        max_frame_num_to_track = end_frame_idx - start_frame_idx

    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "forward",
        "start_frame_index": start_frame_idx,
        "max_frame_num_to_track": max_frame_num_to_track,
    }
    session = predictor._get_session(session_id)
    inference_state = session["state"]
    for response in predictor.handle_stream_request(request=request):
        frame_idx = response["frame_index"]
        outputs_per_frame[frame_idx] = response["outputs"]

        if clear_frame_cache:
            inference_state["cached_frame_outputs"][frame_idx] = {}
        if (
            clear_cuda_cache_every_n_frames > 0
            and (frame_idx - start_frame_idx + 1) % clear_cuda_cache_every_n_frames == 0
        ):
            torch.cuda.empty_cache()
    return outputs_per_frame


# ============================== outputs ==============================

TARGET_COLORS_BGR = [
    (0, 255, 80),
    (0, 96, 255),
    (255, 80, 0),
    (255, 0, 200),
    (0, 220, 220),
    (180, 120, 255),
]


def get_object_mask(outputs: dict, obj_id: int, fallback_largest: bool = False) -> Optional[np.ndarray]:
    if outputs is None:
        return None
    out_obj_ids = outputs.get("out_obj_ids", [])
    out_masks = outputs.get("out_binary_masks", [])
    if len(out_obj_ids) == 0 or len(out_masks) == 0:
        return None

    for idx, current_obj_id in enumerate(out_obj_ids):
        if int(current_obj_id) == obj_id:
            return out_masks[idx].astype(bool)

    if not fallback_largest:
        return None

    areas = [int(mask.sum()) for mask in out_masks]
    if not areas or max(areas) == 0:
        return None
    return out_masks[int(np.argmax(areas))].astype(bool)


def build_instance_mask(
    outputs: dict,
    target_specs: List[dict],
    frame_hw: Tuple[int, int],
) -> np.ndarray:
    instance_mask = np.zeros(frame_hw, dtype=np.uint8)
    for spec in target_specs:
        mask = get_object_mask(outputs, spec["obj_id"])
        if mask is None:
            continue
        if mask.shape[:2] != frame_hw:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame_hw[1], frame_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        instance_mask[mask] = spec["target_mask_id"]
    return instance_mask


def color_for_target(target_specs: List[dict], target_mask_id: int) -> Tuple[int, int, int]:
    for idx, spec in enumerate(target_specs):
        if spec["target_mask_id"] == target_mask_id:
            return TARGET_COLORS_BGR[idx % len(TARGET_COLORS_BGR)]
    return (255, 255, 255)


def colorize_instance_mask(instance_mask: np.ndarray, target_specs: List[dict]) -> np.ndarray:
    color_mask = np.zeros((*instance_mask.shape[:2], 3), dtype=np.uint8)
    for spec in target_specs:
        color_mask[instance_mask == spec["target_mask_id"]] = color_for_target(
            target_specs, spec["target_mask_id"]
        )
    return color_mask


def overlay_instance_mask(
    frame_bgr: np.ndarray,
    instance_mask: np.ndarray,
    target_specs: List[dict],
    alpha: float = 0.45,
) -> np.ndarray:
    if instance_mask.shape[:2] != frame_bgr.shape[:2]:
        instance_mask = cv2.resize(
            instance_mask,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    overlay = frame_bgr.copy()
    color_mask = colorize_instance_mask(instance_mask, target_specs)
    foreground = instance_mask > 0
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, color_mask, alpha, 0)
    overlay[foreground] = blended[foreground]

    for spec in target_specs:
        binary = (instance_mask == spec["target_mask_id"]).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            overlay,
            contours,
            -1,
            color_for_target(target_specs, spec["target_mask_id"]),
            2,
        )
    return overlay


def visual_frame_indices(total_frames: int, count: int) -> set:
    if count <= 0:
        return set()
    head = set(range(min(count, total_frames)))
    tail = set(range(max(0, total_frames - count), total_frames))
    return head | tail


def save_outputs(
    outputs_per_frame: Dict[int, dict],
    resource_path: Path,
    input_mode: str,
    output_dir: Path,
    target_specs: List[dict],
    fps: float,
    write_overlay_video: bool,
    visual_count: int,
) -> None:
    masks_dir = output_dir / "masks"
    combined_dir = masks_dir / "combined_instance"
    visual_dir = output_dir / "visual"
    masks_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    for spec in target_specs:
        (masks_dir / spec["name"]).mkdir(parents=True, exist_ok=True)

    total_frames = max(outputs_per_frame.keys(), default=-1) + 1
    visual_indices = visual_frame_indices(total_frames, visual_count)

    writer = None
    try:
        for frame_idx, frame_bgr, stem in tqdm(
            iter_frames(resource_path, input_mode), desc="writing outputs"
        ):
            frame_hw = frame_bgr.shape[:2]
            instance_mask = build_instance_mask(
                outputs_per_frame.get(frame_idx), target_specs, frame_hw
            )
            cv2.imwrite(str(combined_dir / f"{stem}_instance_mask.png"), instance_mask)

            for spec in target_specs:
                target_binary = (instance_mask == spec["target_mask_id"]).astype(np.uint8) * 255
                cv2.imwrite(str(masks_dir / spec["name"] / f"{stem}_mask.png"), target_binary)

            overlay = overlay_instance_mask(frame_bgr, instance_mask, target_specs)

            if frame_idx in visual_indices:
                color_mask = colorize_instance_mask(instance_mask, target_specs)
                cv2.imwrite(str(visual_dir / f"{stem}_color_mask.png"), color_mask)
                cv2.imwrite(str(visual_dir / f"{stem}_overlay.png"), overlay)

            if write_overlay_video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_path = output_dir / "overlay.mp4"
                    h, w = frame_bgr.shape[:2]
                    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                writer.write(overlay)
    finally:
        if writer is not None:
            writer.release()


def main() -> None:
    args = parse_args()
    configure_environment(args)

    resource_path = resolve_resource_path(args)

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 video inference requires CUDA in this project.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"gpu_id={args.gpu_id} is invalid; available GPUs: {torch.cuda.device_count()}"
        )

    from sam3.model_builder import build_sam3_video_predictor

    target_specs = resolve_target_specs(args)
    print("Resolved target specs:")
    for spec in target_specs:
        print(
            f"  name={spec['name']}, target_mask_id={spec['target_mask_id']}, "
            f"obj_id={spec['obj_id']}"
        )

    print(f"Building SAM 3 video predictor on GPU {args.gpu_id}...")
    predictor = build_sam3_video_predictor(gpus_to_use=[args.gpu_id])

    session_id = None
    try:
        print(f"Starting session: {resource_path}")
        session_id = start_session_with_options(
            predictor=predictor,
            resource_path=resource_path,
            args=args,
        )
        inference_state = predictor._get_session(session_id)["state"]
        target_hw = (inference_state["orig_height"], inference_state["orig_width"])
        num_frames = inference_state["num_frames"]

        mask_schedule = resolve_mask_schedule(
            args=args,
            resource_path=resource_path,
            input_mode=args.input_mode,
            num_frames=num_frames,
        )
        print("Resolved mask prompt schedule:")
        for prompt_frame_idx, prompt_mask_path in mask_schedule:
            print(f"  frame_index={prompt_frame_idx:04d}: {prompt_mask_path}")

        outputs_per_frame = {}
        for schedule_idx, (prompt_frame_idx, prompt_mask_path) in enumerate(mask_schedule):
            next_prompt_frame_idx = (
                mask_schedule[schedule_idx + 1][0]
                if schedule_idx + 1 < len(mask_schedule)
                else None
            )
            segment_end_idx = (
                next_prompt_frame_idx - 1
                if next_prompt_frame_idx is not None
                else num_frames - 1
            )

            action_name = "Adding" if schedule_idx == 0 else "Refining with"
            prompted_outputs = None
            for spec in target_specs:
                prompt_mask = load_binary_mask(
                    mask_path=prompt_mask_path,
                    target_hw=target_hw,
                    invert=args.invert_mask,
                    keep_largest_component=args.keep_largest_component,
                    mask_mode=args.mask_mode,
                    target_mask_id=spec["target_mask_id"],
                    allow_empty=True,
                )
                if not prompt_mask.any():
                    print(
                        f"Warning: {prompt_mask_path} has no pixels for "
                        f"{spec['name']} (target_mask_id={spec['target_mask_id']}); skipped."
                    )
                    continue

                print(
                    f"{action_name} mask prompt on frame_index={prompt_frame_idx}; "
                    f"target={spec['name']} id={spec['target_mask_id']} "
                    f"pixels={int(prompt_mask.sum())}, frame size={target_hw[1]}x{target_hw[0]}"
                )
                prompted_outputs = add_mask_prompt(
                    predictor=predictor,
                    session_id=session_id,
                    frame_idx=prompt_frame_idx,
                    obj_id=spec["obj_id"],
                    binary_mask=prompt_mask,
                )

            if prompted_outputs is None:
                raise RuntimeError(f"No valid target masks found in prompt file: {prompt_mask_path}")

            outputs_per_frame[prompt_frame_idx] = prompted_outputs

            print(
                f"Propagating frame_index={prompt_frame_idx} "
                f"through frame_index={segment_end_idx}..."
            )
            segment_outputs = propagate_in_video(
                predictor=predictor,
                session_id=session_id,
                start_frame_idx=prompt_frame_idx,
                end_frame_idx=segment_end_idx,
                clear_frame_cache=args.clear_frame_cache_during_propagation,
                clear_cuda_cache_every_n_frames=args.clear_cuda_cache_every_n_frames,
            )
            outputs_per_frame.update(segment_outputs)
            outputs_per_frame[prompt_frame_idx] = prompted_outputs

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fps = get_video_fps(resource_path, args.input_mode, args.frames_mode_fps)
        save_outputs(
            outputs_per_frame=outputs_per_frame,
            resource_path=resource_path,
            input_mode=args.input_mode,
            output_dir=output_dir,
            target_specs=target_specs,
            fps=fps,
            write_overlay_video=args.write_overlay_video,
            visual_count=args.visual_frame_count,
        )

        print(f"Done. Masks and overlays saved to: {output_dir.resolve()}")
    finally:
        if session_id is not None:
            predictor.handle_request(
                request={"type": "close_session", "session_id": session_id}
            )
        predictor.shutdown()


if __name__ == "__main__":
    main()