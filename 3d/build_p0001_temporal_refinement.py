#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from reconstruct_coronary_tree import (
    load_binary_mask,
    process_mask,
    save_binary,
)


LEFT_SERIES = ("00000001", "00000002", "00000003")


@dataclass
class TemporalFrame:
    frame_id: str
    image_path: Path
    mask_path: Path
    image: np.ndarray
    mask: np.ndarray


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def shift_array(arr: np.ndarray, shift_rc: tuple[float, float], order: int) -> np.ndarray:
    return ndi.shift(arr, shift=shift_rc, order=order, mode="nearest", prefilter=False)


def phase_correlation_shift(
    moving: np.ndarray,
    fixed: np.ndarray,
    roi_mask: np.ndarray | None = None,
    max_shift: int = 45,
) -> tuple[int, int]:
    # Use a downsampled brute-force NCC search instead of FFT phase correlation.
    # It is slower but stable in this environment and sufficient for 800x800 angiography frames.
    ds = 4
    moving_ds = moving[::ds, ::ds]
    fixed_ds = fixed[::ds, ::ds]
    if roi_mask is None:
        roi_ds = np.ones_like(moving_ds, dtype=np.float32)
    else:
        roi_ds = roi_mask[::ds, ::ds].astype(np.float32)
    max_shift_ds = max(1, max_shift // ds)

    def overlap_score(dr: int, dc: int) -> float:
        rs = max(0, dr)
        re = min(fixed_ds.shape[0], fixed_ds.shape[0] + dr)
        cs = max(0, dc)
        ce = min(fixed_ds.shape[1], fixed_ds.shape[1] + dc)
        srs = max(0, -dr)
        sre = srs + (re - rs)
        scs = max(0, -dc)
        sce = scs + (ce - cs)
        if re <= rs or ce <= cs:
            return -1e18
        a = fixed_ds[rs:re, cs:ce]
        b = moving_ds[srs:sre, scs:sce]
        m = roi_ds[rs:re, cs:ce]
        if m.sum() < 100:
            return -1e18
        a = a[m > 0]
        b = b[m > 0]
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-6:
            return -1e18
        return float(np.dot(a, b) / denom)

    best = (-1e18, 0, 0)
    for dr in range(-max_shift_ds, max_shift_ds + 1):
        for dc in range(-max_shift_ds, max_shift_ds + 1):
            score = overlap_score(dr, dc)
            if score > best[0]:
                best = (score, dr, dc)
    return int(best[1] * ds), int(best[2] * ds)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return ndi.maximum_filter(mask.astype(np.uint8), size=size, mode="nearest") > 0


def save_float_image(arr: np.ndarray, path: Path) -> None:
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="L").save(path)


def save_color_time_map(frame_idx_map: np.ndarray, valid_mask: np.ndarray, path: Path) -> None:
    colors = np.zeros((*frame_idx_map.shape, 3), dtype=np.uint8)
    if valid_mask.any():
        vals = frame_idx_map[valid_mask].astype(np.float32)
        vals = vals - vals.min()
        denom = max(vals.max(), 1.0)
        vals = vals / denom
        hue = vals
        colors[..., 0][valid_mask] = (255 * (1.0 - hue)).astype(np.uint8)
        colors[..., 1][valid_mask] = (255 * np.clip(1.0 - np.abs(hue - 0.5) * 1.6, 0.0, 1.0)).astype(np.uint8)
        colors[..., 2][valid_mask] = (255 * hue).astype(np.uint8)
    Image.fromarray(colors, mode="RGB").save(path)


def compute_mask_temporal_maps(aligned_masks_arr: np.ndarray) -> dict[str, np.ndarray]:
    num_frames = aligned_masks_arr.shape[0]
    counts = aligned_masks_arr.sum(axis=0).astype(np.float32)
    any_seen = counts > 0
    support_fraction = counts / max(num_frames, 1)

    first_seen = np.argmax(aligned_masks_arr, axis=0).astype(np.int32)
    reversed_first = np.argmax(aligned_masks_arr[::-1], axis=0).astype(np.int32)
    last_seen = (num_frames - 1 - reversed_first).astype(np.int32)
    frame_idx = np.arange(num_frames, dtype=np.float32)[:, None, None]
    time_centroid = np.divide(
        (aligned_masks_arr.astype(np.float32) * frame_idx).sum(axis=0),
        np.maximum(counts, 1.0),
    )
    time_span = (last_seen - first_seen).astype(np.float32)

    first_seen[~any_seen] = 0
    last_seen[~any_seen] = 0
    time_centroid[~any_seen] = 0.0
    time_span[~any_seen] = 0.0
    return {
        "support_fraction": support_fraction,
        "first_seen": first_seen.astype(np.float32),
        "last_seen": last_seen.astype(np.float32),
        "time_centroid": time_centroid.astype(np.float32),
        "time_span": time_span.astype(np.float32),
        "any_seen": any_seen,
    }


def refine_series(series: str, base_dir: Path, output_dir: Path) -> dict[str, object]:
    image_paths = sorted((base_dir / "p0001_unique" / "images").glob(f"p0001_{series}_f*.png"))
    mask_paths = {path.stem: base_dir / "p0001_unique" / "masks" / path.name for path in image_paths}
    frames = [
        TemporalFrame(
            frame_id=path.stem,
            image_path=path,
            mask_path=mask_paths[path.stem],
            image=load_gray(path),
            mask=load_binary_mask(mask_paths[path.stem]),
        )
        for path in image_paths
    ]

    reference = max(frames, key=lambda frame: int(frame.mask.sum()))
    roi_mask = dilate(reference.mask, 100).astype(np.float32)

    aligned_images = []
    aligned_masks = []
    shifts = []
    for frame in frames:
        dr, dc = phase_correlation_shift(frame.image, reference.image, roi_mask=roi_mask, max_shift=45)
        shifts.append({"frame": frame.frame_id, "shift_dr": dr, "shift_dc": dc})
        aligned_images.append(shift_array(frame.image, (dr, dc), order=1))
        aligned_masks.append(shift_array(frame.mask.astype(np.float32), (dr, dc), order=0) > 0.5)

    aligned_images_arr = np.stack(aligned_images, axis=0)
    aligned_masks_arr = np.stack(aligned_masks, axis=0)
    mask_temporal = compute_mask_temporal_maps(aligned_masks_arr)

    baseline_count = max(2, min(3, len(frames) // 3))
    baseline = aligned_images_arr[:baseline_count].mean(axis=0)
    enhancement = np.clip(baseline[None, ...] - aligned_images_arr, 0.0, None)

    vessel_roi = dilate(np.logical_or.reduce(aligned_masks_arr), 60)
    enhancement_roi = enhancement * vessel_roi[None, ...]
    max_enhancement = enhancement_roi.max(axis=0)
    sum_enhancement = enhancement_roi.sum(axis=0)

    peak_idx = enhancement_roi.argmax(axis=0)
    valid_temporal = max_enhancement > max(np.percentile(max_enhancement[vessel_roi], 55), 1.0)

    support_union = np.logical_or.reduce(aligned_masks_arr)
    stable = aligned_masks_arr.sum(axis=0) >= max(2, len(frames) // 3)
    temporal_mask_support = mask_temporal["support_fraction"] >= 0.18
    temporal_mask_stable = mask_temporal["support_fraction"] >= 0.35
    temporally_ordered_mask = mask_temporal["any_seen"] & (mask_temporal["first_seen"] <= max(len(frames) - 2, 1))

    core_threshold = np.percentile(max_enhancement[support_union], 58) if support_union.any() else 0.0
    expand_threshold = np.percentile(max_enhancement[vessel_roi], 88) if vessel_roi.any() else 0.0

    contrast_core = max_enhancement > core_threshold
    contrast_expand = max_enhancement > expand_threshold

    # Conservative refinement:
    # keep mask-supported regions that carry temporal contrast signal,
    # and only add small nearby expansions with very strong enhancement.
    candidate = support_union & dilate(contrast_core, 4)
    new_regions = contrast_expand & dilate(support_union, 14)
    mask_temporal_candidate = temporal_mask_support & dilate(contrast_core, 3)
    refined_mask = candidate | stable | new_regions | temporal_mask_stable | (mask_temporal_candidate & temporally_ordered_mask)
    refined_mask = ndi.binary_opening(refined_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    refined_mask = ndi.binary_closing(refined_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)

    labels, n = ndi.label(refined_mask)
    if n > 0:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        refined_mask = labels == counts.argmax()

    refined_mask_path = output_dir / f"p0001_{series}_temporal_refined_mask.png"
    enhancement_path = output_dir / f"p0001_{series}_temporal_max_enhancement.png"
    enhancement_sum_path = output_dir / f"p0001_{series}_temporal_sum_enhancement.png"
    peak_time_path = output_dir / f"p0001_{series}_temporal_peak_time.png"
    mask_support_path = output_dir / f"p0001_{series}_mask_temporal_support.png"
    mask_first_seen_path = output_dir / f"p0001_{series}_mask_first_seen.png"
    mask_last_seen_path = output_dir / f"p0001_{series}_mask_last_seen.png"
    mask_time_centroid_path = output_dir / f"p0001_{series}_mask_time_centroid.png"
    mask_first_seen_gray_path = output_dir / f"p0001_{series}_mask_first_seen_gray.png"
    mask_last_seen_gray_path = output_dir / f"p0001_{series}_mask_last_seen_gray.png"
    mask_time_centroid_gray_path = output_dir / f"p0001_{series}_mask_time_centroid_gray.png"
    shifts_path = output_dir / f"p0001_{series}_temporal_shifts.json"

    save_binary(refined_mask, refined_mask_path)
    save_float_image(max_enhancement, enhancement_path)
    save_float_image(sum_enhancement, enhancement_sum_path)
    save_color_time_map(peak_idx, valid_temporal, peak_time_path)
    save_float_image(mask_temporal["support_fraction"], mask_support_path)
    save_color_time_map(mask_temporal["first_seen"], mask_temporal["any_seen"], mask_first_seen_path)
    save_color_time_map(mask_temporal["last_seen"], mask_temporal["any_seen"], mask_last_seen_path)
    save_color_time_map(mask_temporal["time_centroid"], mask_temporal["any_seen"], mask_time_centroid_path)
    save_float_image(mask_temporal["first_seen"], mask_first_seen_gray_path)
    save_float_image(mask_temporal["last_seen"], mask_last_seen_gray_path)
    save_float_image(mask_temporal["time_centroid"], mask_time_centroid_gray_path)
    shifts_path.write_text(json.dumps(shifts, ensure_ascii=False, indent=2), encoding="utf-8")

    tree = process_mask(
        f"p0001_{series}_temporal_refined",
        "left",
        refined_mask,
        refined_mask_path,
        refined_mask_path.stem,
        output_dir,
    )

    return {
        "series": series,
        "reference_frame": reference.frame_id,
        "refined_mask": str(refined_mask_path),
        "temporal_max_enhancement": str(enhancement_path),
        "temporal_sum_enhancement": str(enhancement_sum_path),
        "temporal_peak_time": str(peak_time_path),
        "mask_temporal_support": str(mask_support_path),
        "mask_first_seen": str(mask_first_seen_path),
        "mask_last_seen": str(mask_last_seen_path),
        "mask_time_centroid": str(mask_time_centroid_path),
        "mask_first_seen_gray": str(mask_first_seen_gray_path),
        "mask_last_seen_gray": str(mask_last_seen_gray_path),
        "mask_time_centroid_gray": str(mask_time_centroid_gray_path),
        "shifts": str(shifts_path),
        "preview": str(tree.preview_path),
        "mesh": str(tree.obj_path),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs_p0001_temporal"
    output_dir.mkdir(exist_ok=True)

    summary = {
        "series": [refine_series(series, base_dir, output_dir) for series in LEFT_SERIES],
        "note": (
            "Motion-compensated temporal refinement uses original angiography frames, "
            "baseline subtraction and mask-guided accumulation to recover vessels as contrast fills over time."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
