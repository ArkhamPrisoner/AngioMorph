#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATASET_INDEX_PATH = ROOT / "outputs_dataset_temporal_prior" / "dataset_index.json"
OUTPUT_DIR = ROOT / "outputs_dicom_projection_pairs"
MIN_ANGLE_DEGREES = 25.0
MATCH_SCORE_THRESHOLD = 0.95


@dataclass
class DicomSeries:
    root_name: str
    study_folder: str
    dicom_path: Path
    patient_id: str
    study_date: str | None
    acquisition_time: str | None
    content_time: str | None
    rows: int
    cols: int
    number_of_frames: int
    cine_rate: float | None
    frame_time_ms: float | None
    primary_angle: float | None
    secondary_angle: float | None
    source_to_detector: float | None
    source_to_patient: float | None
    imager_pixel_spacing: list[float] | None
    magnification: float | None
    field_of_view_dimensions: list[float] | None
    field_of_view_shape: str | None
    detector_type: str | None
    intensifier_size: float | None

    @property
    def angle_vector(self) -> tuple[float, float] | None:
        if self.primary_angle is None or self.secondary_angle is None:
            return None
        return (self.primary_angle, self.secondary_angle)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def to_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(x) for x in value]
    except Exception:
        return None


def build_dicom_index() -> list[DicomSeries]:
    roots = [ROOT.parent / "Ангио", ROOT.parent / "Ангио_2", ROOT.parent / "Ангио_24032026"]
    series: list[DicomSeries] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "DICOMDIR":
                continue
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if getattr(ds, "Modality", None) != "XA":
                continue
            num_frames = getattr(ds, "NumberOfFrames", None)
            if num_frames is None:
                continue
            series.append(
                DicomSeries(
                    root_name=root.name,
                    study_folder=str(path.parent.relative_to(root)),
                    dicom_path=path,
                    patient_id=str(getattr(ds, "PatientID", "")),
                    study_date=getattr(ds, "StudyDate", None),
                    acquisition_time=getattr(ds, "AcquisitionTime", None),
                    content_time=getattr(ds, "ContentTime", None),
                    rows=int(ds.Rows),
                    cols=int(ds.Columns),
                    number_of_frames=int(num_frames),
                    cine_rate=to_float(getattr(ds, "CineRate", None)),
                    frame_time_ms=to_float(getattr(ds, "FrameTime", None)),
                    primary_angle=to_float(getattr(ds, "PositionerPrimaryAngle", None)),
                    secondary_angle=to_float(getattr(ds, "PositionerSecondaryAngle", None)),
                    source_to_detector=to_float(getattr(ds, "DistanceSourceToDetector", None)),
                    source_to_patient=to_float(getattr(ds, "DistanceSourceToPatient", None)),
                    imager_pixel_spacing=to_float_list(getattr(ds, "ImagerPixelSpacing", None)),
                    magnification=to_float(getattr(ds, "EstimatedRadiographicMagnificationFactor", None)),
                    field_of_view_dimensions=to_float_list(getattr(ds, "FieldOfViewDimensions", None)),
                    field_of_view_shape=getattr(ds, "FieldOfViewShape", None),
                    detector_type=getattr(ds, "DetectorType", None),
                    intensifier_size=to_float(getattr(ds, "IntensifierSize", None)),
                )
            )
    return series


def prep_png(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr


def frame_corr(dicom_frame: np.ndarray, png_frame: np.ndarray) -> float:
    frame = np.array(Image.fromarray(dicom_frame).resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    frame = (frame - frame.mean()) / (frame.std() + 1e-6)
    return float((frame * png_frame).mean())


def load_dataset_series() -> list[dict[str, Any]]:
    index = load_json(DATASET_INDEX_PATH)
    return index["series"]


def build_series_samples(frame_exports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picks = []
    for pos in [0, len(frame_exports) // 2, len(frame_exports) - 1]:
        frame_export = frame_exports[pos]
        picks.append(
            {
                "frame_index": frame_export["frame_index"],
                "image_path": Path(frame_export["image_path"]),
                "mask_path": Path(frame_export["consensus_mask"]),
                "mask_confidence_path": Path(frame_export["mask_confidence"]),
            }
        )
    unique = []
    seen: set[int] = set()
    for item in picks:
        if item["frame_index"] not in seen:
            unique.append(item)
            seen.add(item["frame_index"])
    return unique


def match_mask_series_to_dicom(mask_series: dict[str, Any], dicom_series_list: list[DicomSeries]) -> dict[str, Any] | None:
    samples = build_series_samples(mask_series["frame_exports"])
    first_image = Image.open(samples[0]["image_path"])
    image_shape = first_image.size[1], first_image.size[0]
    max_frame_index = max(item["frame_index"] for item in samples)

    png_cache = {item["frame_index"]: prep_png(item["image_path"]) for item in samples}
    eligible = [
        item
        for item in dicom_series_list
        if item.rows == image_shape[0] and item.cols == image_shape[1] and item.number_of_frames >= max_frame_index
    ]
    if not eligible:
        return None

    scored: list[dict[str, Any]] = []
    for candidate in eligible:
        ds = pydicom.dcmread(candidate.dicom_path, force=True)
        try:
            pixel_array = ds.pixel_array
        except Exception:
            continue
        scores = []
        for item in samples:
            frame_idx = item["frame_index"] - 1
            scores.append(frame_corr(pixel_array[frame_idx], png_cache[item["frame_index"]]))
        mean_score = float(sum(scores) / len(scores))
        scored.append(
            {
                "dicom_series": candidate,
                "score_mean": mean_score,
                "score_samples": scores,
            }
        )

    if not scored:
        return None

    scored.sort(key=lambda x: x["score_mean"], reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    best_series: DicomSeries = best["dicom_series"]

    return {
        "patient": mask_series["patient"],
        "series": mask_series["series"],
        "num_mask_frames": mask_series["num_unique_frames"],
        "best_match_score": round(best["score_mean"], 6),
        "best_match_scores_per_sample": [round(x, 6) for x in best["score_samples"]],
        "second_match_score": round(second["score_mean"], 6) if second else None,
        "accepted": best["score_mean"] >= MATCH_SCORE_THRESHOLD,
        "dicom": {
            "root_name": best_series.root_name,
            "study_folder": best_series.study_folder,
            "dicom_path": str(best_series.dicom_path.resolve()),
            "patient_id": best_series.patient_id,
            "study_date": best_series.study_date,
            "acquisition_time": best_series.acquisition_time,
            "content_time": best_series.content_time,
            "rows": best_series.rows,
            "cols": best_series.cols,
            "number_of_frames": best_series.number_of_frames,
            "cine_rate": best_series.cine_rate,
            "frame_time_ms": best_series.frame_time_ms,
            "positioner_primary_angle": best_series.primary_angle,
            "positioner_secondary_angle": best_series.secondary_angle,
            "distance_source_to_detector": best_series.source_to_detector,
            "distance_source_to_patient": best_series.source_to_patient,
            "imager_pixel_spacing": best_series.imager_pixel_spacing,
            "estimated_radiographic_magnification_factor": best_series.magnification,
            "field_of_view_dimensions": best_series.field_of_view_dimensions,
            "field_of_view_shape": best_series.field_of_view_shape,
            "detector_type": best_series.detector_type,
            "intensifier_size": best_series.intensifier_size,
        },
        "sample_frames": [
            {
                "frame_index": item["frame_index"],
                "image_path": str(item["image_path"]),
                "mask_path": str(item["mask_path"]),
                "mask_confidence_path": str(item["mask_confidence_path"]),
            }
            for item in samples
        ],
        "all_mask_frames": [
            {
                "frame": item["frame"],
                "frame_index": item["frame_index"],
                "image_path": item["image_path"],
                "mask_path": item["consensus_mask"],
                "mask_confidence_path": item["mask_confidence"],
                "normalized_phase": round(
                    (item["frame_index"] - 1) / max(1, best_series.number_of_frames - 1),
                    6,
                ),
            }
            for item in mask_series["frame_exports"]
        ],
    }


def angle_delta_degrees(series_a: dict[str, Any], series_b: dict[str, Any]) -> dict[str, float] | None:
    a1 = series_a["dicom"]["positioner_primary_angle"]
    a2 = series_a["dicom"]["positioner_secondary_angle"]
    b1 = series_b["dicom"]["positioner_primary_angle"]
    b2 = series_b["dicom"]["positioner_secondary_angle"]
    if None in [a1, a2, b1, b2]:
        return None
    d_primary = float(b1 - a1)
    d_secondary = float(b2 - a2)
    magnitude = math.hypot(d_primary, d_secondary)
    return {
        "delta_primary": round(d_primary, 6),
        "delta_secondary": round(d_secondary, 6),
        "delta_magnitude": round(magnitude, 6),
    }


def build_phase_matches(series_a: dict[str, Any], series_b: dict[str, Any], max_pairs: int = 12) -> list[dict[str, Any]]:
    frames_a = series_a["all_mask_frames"]
    frames_b = series_b["all_mask_frames"]
    phase_pairs: list[dict[str, Any]] = []
    for item_a in frames_a:
        best_b = min(frames_b, key=lambda x: abs(x["normalized_phase"] - item_a["normalized_phase"]))
        phase_gap = abs(best_b["normalized_phase"] - item_a["normalized_phase"])
        phase_pairs.append(
            {
                "frame_a": item_a["frame"],
                "frame_a_index": item_a["frame_index"],
                "image_a_path": item_a["image_path"],
                "mask_a_path": item_a["mask_path"],
                "phase_a": item_a["normalized_phase"],
                "frame_b": best_b["frame"],
                "frame_b_index": best_b["frame_index"],
                "image_b_path": best_b["image_path"],
                "mask_b_path": best_b["mask_path"],
                "phase_b": best_b["normalized_phase"],
                "phase_gap": round(phase_gap, 6),
            }
        )
    phase_pairs.sort(key=lambda x: x["phase_gap"])
    return phase_pairs[:max_pairs]


def build_pair_examples(matched_series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for item in matched_series:
        if not item["accepted"]:
            continue
        by_patient.setdefault(item["patient"], []).append(item)

    pairs: list[dict[str, Any]] = []
    for patient, patient_series in sorted(by_patient.items()):
        patient_series.sort(key=lambda x: x["series"])
        for i in range(len(patient_series)):
            for j in range(i + 1, len(patient_series)):
                a = patient_series[i]
                b = patient_series[j]
                angle = angle_delta_degrees(a, b)
                if angle is None or angle["delta_magnitude"] < MIN_ANGLE_DEGREES:
                    continue
                phase_pairs = build_phase_matches(a, b)
                pairs.append(
                    {
                        "patient": patient,
                        "series_a": a["series"],
                        "series_b": b["series"],
                        "dicom_a_path": a["dicom"]["dicom_path"],
                        "dicom_b_path": b["dicom"]["dicom_path"],
                        "dicom_root_a": a["dicom"]["root_name"],
                        "dicom_root_b": b["dicom"]["root_name"],
                        "dicom_patient_id": a["dicom"]["patient_id"],
                        "study_date": a["dicom"]["study_date"],
                        "acquisition_time_a": a["dicom"]["acquisition_time"],
                        "acquisition_time_b": b["dicom"]["acquisition_time"],
                        "angles": angle,
                        "num_mask_frames_a": a["num_mask_frames"],
                        "num_mask_frames_b": b["num_mask_frames"],
                        "number_of_frames_a": a["dicom"]["number_of_frames"],
                        "number_of_frames_b": b["dicom"]["number_of_frames"],
                        "cine_rate_fps_a": a["dicom"]["cine_rate"],
                        "cine_rate_fps_b": b["dicom"]["cine_rate"],
                        "frame_time_ms_a": a["dicom"]["frame_time_ms"],
                        "frame_time_ms_b": b["dicom"]["frame_time_ms"],
                        "positioner_primary_angle_a": a["dicom"]["positioner_primary_angle"],
                        "positioner_secondary_angle_a": a["dicom"]["positioner_secondary_angle"],
                        "positioner_primary_angle_b": b["dicom"]["positioner_primary_angle"],
                        "positioner_secondary_angle_b": b["dicom"]["positioner_secondary_angle"],
                        "distance_source_to_detector_a": a["dicom"]["distance_source_to_detector"],
                        "distance_source_to_detector_b": b["dicom"]["distance_source_to_detector"],
                        "distance_source_to_patient_a": a["dicom"]["distance_source_to_patient"],
                        "distance_source_to_patient_b": b["dicom"]["distance_source_to_patient"],
                        "imager_pixel_spacing_a": a["dicom"]["imager_pixel_spacing"],
                        "imager_pixel_spacing_b": b["dicom"]["imager_pixel_spacing"],
                        "estimated_radiographic_magnification_factor_a": a["dicom"]["estimated_radiographic_magnification_factor"],
                        "estimated_radiographic_magnification_factor_b": b["dicom"]["estimated_radiographic_magnification_factor"],
                        "mask_match_score_a": a["best_match_score"],
                        "mask_match_score_b": b["best_match_score"],
                        "phase_pair_examples": phase_pairs,
                    }
                )
    pairs.sort(key=lambda x: (x["patient"], -x["angles"]["delta_magnitude"], x["series_a"], x["series_b"]))
    return pairs


def write_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient",
                "series_a",
                "series_b",
                "dicom_patient_id",
                "study_date",
                "acquisition_time_a",
                "acquisition_time_b",
                "positioner_primary_angle_a",
                "positioner_secondary_angle_a",
                "positioner_primary_angle_b",
                "positioner_secondary_angle_b",
                "delta_primary",
                "delta_secondary",
                "delta_magnitude",
                "num_mask_frames_a",
                "num_mask_frames_b",
                "number_of_frames_a",
                "number_of_frames_b",
                "frame_time_ms_a",
                "frame_time_ms_b",
                "distance_source_to_detector_a",
                "distance_source_to_detector_b",
                "distance_source_to_patient_a",
                "distance_source_to_patient_b",
                "imager_pixel_spacing_a",
                "imager_pixel_spacing_b",
                "mask_match_score_a",
                "mask_match_score_b",
                "dicom_a_path",
                "dicom_b_path",
            ],
        )
        writer.writeheader()
        for item in pairs:
            writer.writerow(
                {
                    "patient": item["patient"],
                    "series_a": item["series_a"],
                    "series_b": item["series_b"],
                    "dicom_patient_id": item["dicom_patient_id"],
                    "study_date": item["study_date"],
                    "acquisition_time_a": item["acquisition_time_a"],
                    "acquisition_time_b": item["acquisition_time_b"],
                    "positioner_primary_angle_a": item["positioner_primary_angle_a"],
                    "positioner_secondary_angle_a": item["positioner_secondary_angle_a"],
                    "positioner_primary_angle_b": item["positioner_primary_angle_b"],
                    "positioner_secondary_angle_b": item["positioner_secondary_angle_b"],
                    "delta_primary": item["angles"]["delta_primary"],
                    "delta_secondary": item["angles"]["delta_secondary"],
                    "delta_magnitude": item["angles"]["delta_magnitude"],
                    "num_mask_frames_a": item["num_mask_frames_a"],
                    "num_mask_frames_b": item["num_mask_frames_b"],
                    "number_of_frames_a": item["number_of_frames_a"],
                    "number_of_frames_b": item["number_of_frames_b"],
                    "frame_time_ms_a": item["frame_time_ms_a"],
                    "frame_time_ms_b": item["frame_time_ms_b"],
                    "distance_source_to_detector_a": item["distance_source_to_detector_a"],
                    "distance_source_to_detector_b": item["distance_source_to_detector_b"],
                    "distance_source_to_patient_a": item["distance_source_to_patient_a"],
                    "distance_source_to_patient_b": item["distance_source_to_patient_b"],
                    "imager_pixel_spacing_a": item["imager_pixel_spacing_a"],
                    "imager_pixel_spacing_b": item["imager_pixel_spacing_b"],
                    "mask_match_score_a": item["mask_match_score_a"],
                    "mask_match_score_b": item["mask_match_score_b"],
                    "dicom_a_path": item["dicom_a_path"],
                    "dicom_b_path": item["dicom_b_path"],
                }
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dicom_series = build_dicom_index()
    dataset_series = load_dataset_series()
    matched = [match_mask_series_to_dicom(item, dicom_series) for item in dataset_series]
    matched = [item for item in matched if item is not None]
    pairs = build_pair_examples(matched)

    summary = {
        "note": (
            "Pairs are built from mask-bearing series that were automatically matched "
            "to XA DICOM cine series by direct image correlation. "
            f"Pairs are kept when angular separation sqrt(dPrimary^2 + dSecondary^2) >= {MIN_ANGLE_DEGREES} degrees."
        ),
        "match_score_threshold": MATCH_SCORE_THRESHOLD,
        "min_angle_degrees": MIN_ANGLE_DEGREES,
        "num_mask_series": len(dataset_series),
        "num_matched_series": sum(1 for item in matched if item["accepted"]),
        "num_pairs": len(pairs),
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "series_to_dicom_matches.json").write_text(json.dumps(matched, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "projection_pairs_angle_ge25.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUTPUT_DIR / "projection_pairs_angle_ge25.csv", pairs)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
