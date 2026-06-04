#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
PAIRS_PATH = ROOT / "outputs_dicom_projection_pairs" / "projection_pairs_angle_ge25.json"
OUTPUT_DIR = ROOT / "reconstruction_cases"
TOP_K_PHASE_PAIRS = 5


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def image_to_float_array(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def mask_to_binary_array(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return (np.asarray(image, dtype=np.float32) > 0).astype(np.float32)


def save_png_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_case_id(pair: dict[str, Any]) -> str:
    return f"{pair['patient']}__{pair['series_a']}__{pair['series_b']}"


def export_phase_pair(case_dir: Path, pair_index: int, phase_pair: dict[str, Any]) -> dict[str, Any]:
    pair_dir = case_dir / "phase_pairs" / f"pair_{pair_index:02d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    image_a_src = Path(phase_pair["image_a_path"])
    image_b_src = Path(phase_pair["image_b_path"])
    mask_a_src = Path(phase_pair["mask_a_path"])
    mask_b_src = Path(phase_pair["mask_b_path"])

    image_a_dst = pair_dir / "image_a.png"
    image_b_dst = pair_dir / "image_b.png"
    mask_a_dst = pair_dir / "mask_a.png"
    mask_b_dst = pair_dir / "mask_b.png"

    for src, dst in [
        (image_a_src, image_a_dst),
        (image_b_src, image_b_dst),
        (mask_a_src, mask_a_dst),
        (mask_b_src, mask_b_dst),
    ]:
        save_png_copy(src, dst)

    image_stack = np.stack(
        [
            image_to_float_array(image_a_src),
            image_to_float_array(image_b_src),
        ],
        axis=0,
    )[None, ...]
    mask_stack = np.stack(
        [
            mask_to_binary_array(mask_a_src),
            mask_to_binary_array(mask_b_src),
        ],
        axis=0,
    )[None, ...]

    np.save(pair_dir / "images.npy", image_stack)
    np.save(pair_dir / "masks.npy", mask_stack)

    pair_manifest = {
        "frame_a": phase_pair["frame_a"],
        "frame_a_index": phase_pair["frame_a_index"],
        "image_a_path": str(image_a_dst.relative_to(case_dir).as_posix()),
        "mask_a_path": str(mask_a_dst.relative_to(case_dir).as_posix()),
        "phase_a": phase_pair["phase_a"],
        "frame_b": phase_pair["frame_b"],
        "frame_b_index": phase_pair["frame_b_index"],
        "image_b_path": str(image_b_dst.relative_to(case_dir).as_posix()),
        "mask_b_path": str(mask_b_dst.relative_to(case_dir).as_posix()),
        "phase_b": phase_pair["phase_b"],
        "phase_gap": phase_pair["phase_gap"],
        "images_npy": str((pair_dir / "images.npy").relative_to(case_dir).as_posix()),
        "masks_npy": str((pair_dir / "masks.npy").relative_to(case_dir).as_posix()),
    }
    (pair_dir / "pair_manifest.json").write_text(json.dumps(pair_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return pair_manifest


def export_case(pair: dict[str, Any]) -> dict[str, Any]:
    case_id = build_case_id(pair)
    case_dir = OUTPUT_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    phase_manifests = [
        export_phase_pair(case_dir, idx + 1, phase_pair)
        for idx, phase_pair in enumerate(pair["phase_pair_examples"][:TOP_K_PHASE_PAIRS])
    ]

    geometry = {
        "image_size": {
            "rows_a": None,
            "cols_a": None,
            "rows_b": None,
            "cols_b": None,
        },
        "projection_a": {
            "series": pair["series_a"],
            "dicom_path": pair["dicom_a_path"],
            "positioner_primary_angle": pair["positioner_primary_angle_a"],
            "positioner_secondary_angle": pair["positioner_secondary_angle_a"],
            "distance_source_to_detector": pair["distance_source_to_detector_a"],
            "distance_source_to_patient": pair["distance_source_to_patient_a"],
            "imager_pixel_spacing": pair["imager_pixel_spacing_a"],
            "frame_time_ms": pair["frame_time_ms_a"],
            "cine_rate_fps": pair["cine_rate_fps_a"],
            "acquisition_time": pair["acquisition_time_a"],
            "number_of_frames": pair["number_of_frames_a"],
        },
        "projection_b": {
            "series": pair["series_b"],
            "dicom_path": pair["dicom_b_path"],
            "positioner_primary_angle": pair["positioner_primary_angle_b"],
            "positioner_secondary_angle": pair["positioner_secondary_angle_b"],
            "distance_source_to_detector": pair["distance_source_to_detector_b"],
            "distance_source_to_patient": pair["distance_source_to_patient_b"],
            "imager_pixel_spacing": pair["imager_pixel_spacing_b"],
            "frame_time_ms": pair["frame_time_ms_b"],
            "cine_rate_fps": pair["cine_rate_fps_b"],
            "acquisition_time": pair["acquisition_time_b"],
            "number_of_frames": pair["number_of_frames_b"],
        },
        "angle_delta": pair["angles"],
    }

    best_phase_pair = phase_manifests[0] if phase_manifests else None
    case_manifest = {
        "case_id": case_id,
        "patient": pair["patient"],
        "dicom_patient_id": pair["dicom_patient_id"],
        "study_date": pair["study_date"],
        "series_a": pair["series_a"],
        "series_b": pair["series_b"],
        "geometry": geometry,
        "mask_match_score_a": pair["mask_match_score_a"],
        "mask_match_score_b": pair["mask_match_score_b"],
        "num_mask_frames_a": pair["num_mask_frames_a"],
        "num_mask_frames_b": pair["num_mask_frames_b"],
        "num_phase_pairs_exported": len(phase_manifests),
        "best_phase_pair": best_phase_pair,
        "phase_pairs": phase_manifests,
    }

    (case_dir / "case_manifest.json").write_text(json.dumps(case_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_manifest


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_json(PAIRS_PATH)
    cases = [export_case(pair) for pair in pairs]
    summary = {
        "num_cases": len(cases),
        "top_k_phase_pairs": TOP_K_PHASE_PAIRS,
        "cases": [
            {
                "case_id": case["case_id"],
                "patient": case["patient"],
                "series_a": case["series_a"],
                "series_b": case["series_b"],
                "angle_delta": case["geometry"]["angle_delta"]["delta_magnitude"],
                "case_manifest": str((OUTPUT_DIR / case["case_id"] / "case_manifest.json").as_posix()),
            }
            for case in cases
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
