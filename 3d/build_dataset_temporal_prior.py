#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from build_p0001_projective_recon import extract_view
from build_p0001_temporal_refinement import (
    compute_mask_temporal_maps,
    dilate,
    load_gray,
    phase_correlation_shift,
    save_color_time_map,
    save_float_image,
    shift_array,
)
from reconstruct_coronary_tree import load_binary_mask, preprocess_mask, save_binary


DATASET_DIR = Path("все/our_data_with_dublicates_297img")
SERIES_RE = re.compile(
    r"^(?P<patient>p\d+)_(?P<series>\d{8})_(?P<frame>f\d{4})_(?P<annotator>\d+)\.png$"
)


@dataclass
class AnnotationEntry:
    annotator: str
    image_path: Path
    mask_path: Path


@dataclass
class UniqueFrame:
    patient: str
    series: str
    frame: str
    frame_index: int
    image_path: Path
    image_hashes: list[str]
    annotators: list[str]
    consensus_mask: np.ndarray
    mask_confidence: np.ndarray
    annotation_count: int


def file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_entries(images_dir: Path, masks_dir: Path) -> dict[tuple[str, str, str], list[AnnotationEntry]]:
    grouped: dict[tuple[str, str, str], list[AnnotationEntry]] = {}
    for image_path in sorted(images_dir.glob("*.png")):
        match = SERIES_RE.match(image_path.name)
        if not match:
            continue
        mask_path = masks_dir / image_path.name
        if not mask_path.exists():
            continue
        key = (match["patient"], match["series"], match["frame"])
        grouped.setdefault(key, []).append(
            AnnotationEntry(
                annotator=match["annotator"],
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return grouped


def build_unique_frames(grouped: dict[tuple[str, str, str], list[AnnotationEntry]]) -> dict[tuple[str, str], list[UniqueFrame]]:
    per_series: dict[tuple[str, str], list[UniqueFrame]] = {}
    for (patient, series, frame), entries in sorted(grouped.items()):
        image_hashes = sorted({file_sha1(item.image_path) for item in entries})
        masks = np.stack([load_binary_mask(item.mask_path).astype(np.float32) for item in entries], axis=0)
        confidence = masks.mean(axis=0)
        consensus = confidence >= 0.5
        if not consensus.any():
            consensus = confidence > 0.0
        consensus = preprocess_mask(consensus)
        per_series.setdefault((patient, series), []).append(
            UniqueFrame(
                patient=patient,
                series=series,
                frame=frame,
                frame_index=int(frame[1:]),
                image_path=entries[0].image_path,
                image_hashes=image_hashes,
                annotators=sorted(item.annotator for item in entries),
                consensus_mask=consensus,
                mask_confidence=confidence,
                annotation_count=len(entries),
            )
        )
    for frames in per_series.values():
        frames.sort(key=lambda item: item.frame_index)
    return per_series


def save_frame_consensus(frame: UniqueFrame, output_dir: Path) -> dict[str, object]:
    frame_dir = ensure_dir(output_dir / frame.patient / frame.series / "frames")
    consensus_path = frame_dir / f"{frame.patient}_{frame.series}_{frame.frame}_consensus_mask.png"
    confidence_path = frame_dir / f"{frame.patient}_{frame.series}_{frame.frame}_mask_confidence.png"
    save_binary(frame.consensus_mask, consensus_path)
    save_float_image(frame.mask_confidence, confidence_path)
    return {
        "frame": frame.frame,
        "frame_index": frame.frame_index,
        "image_path": str(frame.image_path),
        "consensus_mask": str(consensus_path),
        "mask_confidence": str(confidence_path),
        "annotation_count": frame.annotation_count,
        "annotators": frame.annotators,
        "distinct_image_hashes": len(frame.image_hashes),
    }


def refine_temporal_series(
    patient: str,
    series: str,
    frames: list[UniqueFrame],
    output_dir: Path,
) -> dict[str, object]:
    series_dir = ensure_dir(output_dir / patient / series)
    if len(frames) < 3:
        return {
            "patient": patient,
            "series": series,
            "status": "skipped",
            "reason": "less_than_3_unique_frames",
            "num_frames": len(frames),
        }

    images = [load_gray(frame.image_path) for frame in frames]
    masks = [frame.consensus_mask for frame in frames]
    reference_idx = int(np.argmax([int(mask.sum()) for mask in masks]))
    reference_image = images[reference_idx]
    reference_mask = masks[reference_idx]
    roi_mask = dilate(reference_mask, 100).astype(np.float32)

    aligned_images = []
    aligned_masks = []
    shifts = []
    for frame, image, mask in zip(frames, images, masks):
        dr, dc = phase_correlation_shift(image, reference_image, roi_mask=roi_mask, max_shift=45)
        shifts.append({"frame": frame.frame, "shift_dr": int(dr), "shift_dc": int(dc)})
        aligned_images.append(shift_array(image, (dr, dc), order=1))
        aligned_masks.append(shift_array(mask.astype(np.float32), (dr, dc), order=0) > 0.5)

    aligned_images_arr = np.stack(aligned_images, axis=0)
    aligned_masks_arr = np.stack(aligned_masks, axis=0)
    mask_temporal = compute_mask_temporal_maps(aligned_masks_arr)

    baseline_count = max(2, min(3, len(frames) // 3))
    baseline = aligned_images_arr[:baseline_count].mean(axis=0)
    enhancement = np.clip(baseline[None, ...] - aligned_images_arr, 0.0, None)

    support_union = np.logical_or.reduce(aligned_masks_arr)
    vessel_roi = dilate(support_union, 60)
    enhancement_roi = enhancement * vessel_roi[None, ...]
    max_enhancement = enhancement_roi.max(axis=0)
    sum_enhancement = enhancement_roi.sum(axis=0)
    peak_idx = enhancement_roi.argmax(axis=0)
    valid_temporal = max_enhancement > max(np.percentile(max_enhancement[vessel_roi], 55), 1.0) if vessel_roi.any() else np.zeros_like(max_enhancement, dtype=bool)

    stable = aligned_masks_arr.sum(axis=0) >= max(2, len(frames) // 3)
    temporal_mask_support = mask_temporal["support_fraction"] >= 0.18
    temporal_mask_stable = mask_temporal["support_fraction"] >= 0.35
    temporally_ordered_mask = mask_temporal["any_seen"] & (mask_temporal["first_seen"] <= max(len(frames) - 2, 1))
    core_threshold = np.percentile(max_enhancement[support_union], 58) if support_union.any() else 0.0
    expand_threshold = np.percentile(max_enhancement[vessel_roi], 88) if vessel_roi.any() else 0.0
    contrast_core = max_enhancement > core_threshold
    contrast_expand = max_enhancement > expand_threshold

    mask_temporal_candidate = temporal_mask_support & dilate(contrast_core, 3)
    refined_mask = (
        (support_union & dilate(contrast_core, 4))
        | stable
        | temporal_mask_stable
        | (contrast_expand & dilate(support_union, 14))
        | (mask_temporal_candidate & temporally_ordered_mask)
    )
    refined_mask = ndi.binary_opening(refined_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    refined_mask = ndi.binary_closing(refined_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labels, num_labels = ndi.label(refined_mask)
    if num_labels > 0:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        refined_mask = labels == counts.argmax()

    refined_mask_path = series_dir / f"{patient}_{series}_temporal_refined_mask.png"
    enhancement_path = series_dir / f"{patient}_{series}_temporal_max_enhancement.png"
    enhancement_sum_path = series_dir / f"{patient}_{series}_temporal_sum_enhancement.png"
    peak_time_path = series_dir / f"{patient}_{series}_temporal_peak_time.png"
    mask_support_path = series_dir / f"{patient}_{series}_mask_temporal_support.png"
    mask_first_seen_path = series_dir / f"{patient}_{series}_mask_first_seen.png"
    mask_last_seen_path = series_dir / f"{patient}_{series}_mask_last_seen.png"
    mask_time_centroid_path = series_dir / f"{patient}_{series}_mask_time_centroid.png"
    mask_first_seen_gray_path = series_dir / f"{patient}_{series}_mask_first_seen_gray.png"
    mask_last_seen_gray_path = series_dir / f"{patient}_{series}_mask_last_seen_gray.png"
    mask_time_centroid_gray_path = series_dir / f"{patient}_{series}_mask_time_centroid_gray.png"
    shifts_path = series_dir / f"{patient}_{series}_temporal_shifts.json"

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

    view = extract_view(
        f"{patient}_{series}",
        refined_mask_path,
        enhancement_path=enhancement_path,
        peak_time_path=peak_time_path,
    )
    proximal_bifurcations = sorted(
        [
            {
                "node": int(item.node),
                "bifurcation_level": int(item.bifurcation_level),
                "path_length": round(float(item.path_length), 3),
                "subtree_leaves": int(item.subtree_leaves),
                "child_count": int(item.child_count),
                "root_branch": int(item.root_branch),
                "signature": [round(float(x), 6) for x in item.signature.tolist()],
            }
            for item in view.bifurcations
            if item.bifurcation_level <= 3
        ],
        key=lambda item: (item["bifurcation_level"], -item["subtree_leaves"], item["path_length"]),
    )
    descriptor = {
        "patient": patient,
        "series": series,
        "status": "ok",
        "reference_frame": frames[reference_idx].frame,
        "num_frames": len(frames),
        "mean_annotation_count": round(float(np.mean([frame.annotation_count for frame in frames])), 3),
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
        "temporal_shifts": str(shifts_path),
        "mask_area": int(refined_mask.sum()),
        "num_leaves": len(view.leaves),
        "num_bifurcations": len(view.bifurcations),
        "proximal_bifurcations": proximal_bifurcations,
    }
    (series_dir / f"{patient}_{series}_descriptor.json").write_text(
        json.dumps(descriptor, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return descriptor


def summarize_prior(series_descriptors: list[dict[str, object]]) -> dict[str, object]:
    ok_series = [item for item in series_descriptors if item["status"] == "ok"]
    level_vectors: dict[int, list[np.ndarray]] = {}
    rank_vectors: dict[int, list[np.ndarray]] = {}
    for item in ok_series:
        proximal = item["proximal_bifurcations"]
        for rank, bif in enumerate(proximal[:4]):
            vec = np.array(bif["signature"], dtype=float)
            level = int(bif["bifurcation_level"])
            level_vectors.setdefault(level, []).append(vec)
            rank_vectors.setdefault(rank, []).append(vec)

    by_level = {}
    for level, vectors in sorted(level_vectors.items()):
        arr = np.stack(vectors, axis=0)
        by_level[str(level)] = {
            "count": int(len(vectors)),
            "mean_signature": [round(float(x), 6) for x in arr.mean(axis=0)],
            "std_signature": [round(float(x), 6) for x in arr.std(axis=0)],
            "mean_peak_time_feature": round(float(arr[:, -1].mean()), 6),
            "mean_local_enhancement_feature": round(float(arr[:, -2].mean()), 6),
        }

    by_rank = {}
    for rank, vectors in sorted(rank_vectors.items()):
        arr = np.stack(vectors, axis=0)
        by_rank[str(rank)] = {
            "count": int(len(vectors)),
            "mean_signature": [round(float(x), 6) for x in arr.mean(axis=0)],
            "std_signature": [round(float(x), 6) for x in arr.std(axis=0)],
        }

    return {
        "num_series_total": len(series_descriptors),
        "num_series_refined": len(ok_series),
        "prior_by_bifurcation_level": by_level,
        "prior_by_proximal_rank": by_rank,
        "note": (
            "Dataset-level proximal prior is built from temporal-refined series descriptors. "
            "Signature dimensions include path depth, bifurcation level, subtree size, child count, "
            "PCA position, parent/child directions, child segment lengths, local enhancement and local peak time."
        ),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    images_dir = base_dir / DATASET_DIR / "images"
    masks_dir = base_dir / DATASET_DIR / "masks"
    outputs_dir = ensure_dir(base_dir / "outputs_dataset_temporal_prior")

    grouped = parse_entries(images_dir, masks_dir)
    unique_per_series = build_unique_frames(grouped)

    index_summary = {
        "patients": {},
        "series": [],
    }
    descriptors = []

    for (patient, series), frames in sorted(unique_per_series.items()):
        patient_entry = index_summary["patients"].setdefault(
            patient,
            {"num_series": 0, "series": []},
        )
        patient_entry["num_series"] = int(patient_entry["num_series"]) + 1

        frame_exports = [save_frame_consensus(frame, outputs_dir) for frame in frames]
        descriptor = refine_temporal_series(patient, series, frames, outputs_dir)
        descriptors.append(descriptor)

        series_summary = {
            "patient": patient,
            "series": series,
            "num_unique_frames": len(frames),
            "num_annotations_total": int(sum(frame.annotation_count for frame in frames)),
            "frame_exports": frame_exports,
            "temporal_descriptor": descriptor,
        }
        patient_entry["series"].append({"series": series, "num_unique_frames": len(frames), "status": descriptor["status"]})
        index_summary["series"].append(series_summary)

    cohort_prior = summarize_prior(descriptors)
    (outputs_dir / "dataset_index.json").write_text(
        json.dumps(index_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (outputs_dir / "series_descriptors.json").write_text(
        json.dumps(descriptors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (outputs_dir / "cohort_proximal_prior.json").write_text(
        json.dumps(cohort_prior, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "dataset_index": str(outputs_dir / "dataset_index.json"),
        "series_descriptors": str(outputs_dir / "series_descriptors.json"),
        "cohort_proximal_prior": str(outputs_dir / "cohort_proximal_prior.json"),
        "num_patients": len(index_summary["patients"]),
        "num_series": len(index_summary["series"]),
        "num_series_refined": cohort_prior["num_series_refined"],
    }
    (outputs_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
