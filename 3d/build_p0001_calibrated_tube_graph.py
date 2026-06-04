#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from build_p0001_projective_recon import (
    LEFT_SERIES,
    build_left_cohort_prior,
    build_projective_edges_from_segments,
    build_proximal_segment_matches,
    evaluate_proximal_geometry,
    extract_view,
    match_bifurcations,
    resample_curve,
)
from reconstruct_coronary_tree import render_preview, write_centerlines_obj, write_json, write_obj_mesh


COLORS = [
    (220, 64, 64),
    (64, 160, 255),
    (80, 190, 120),
    (255, 180, 60),
    (190, 100, 220),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_case_id(series_a: str, series_b: str) -> str:
    return f"p0001__{series_a}__{series_b}"


def sample_distance_profile(distance_map: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    rr = np.clip(np.round(points_xy[:, 1]).astype(int), 0, distance_map.shape[0] - 1)
    cc = np.clip(np.round(points_xy[:, 0]).astype(int), 0, distance_map.shape[1] - 1)
    return distance_map[rr, cc].astype(np.float32)


def estimate_isocenter_spacing_mm(geometry: dict, which: str) -> float:
    proj = geometry[f"projection_{which}"]
    detector_spacing = float(proj["imager_pixel_spacing"][0])
    dso = float(proj["distance_source_to_patient"])
    dsd = float(proj["distance_source_to_detector"])
    return detector_spacing * dso / dsd


def polyline_length_3d(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def draw_polyline_overlay(image_path: Path, polylines: list[dict], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for item in polylines:
        color = item["color"]
        points = [tuple(map(float, pt)) for pt in item["points"]]
        width = max(2, int(round(item["width_px"])))
        if len(points) >= 2:
            draw.line(points, fill=color + (220,), width=width, joint="curve")
        for pt in points[:: max(1, len(points) // 6)]:
            r = max(2, int(round(width / 2)))
            x, y = pt
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color + (180,), outline=(255, 255, 255, 180))
    image.save(out_path)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / "outputs_p0001_calibrated_tube_graph"
    outputs_dir.mkdir(exist_ok=True)

    projective_summary = load_json(base_dir / "outputs_p0001_projective" / "summary.json")
    series_a, series_b = projective_summary["best_pair"]["series"]
    assert series_a in LEFT_SERIES and series_b in LEFT_SERIES

    temporal_dir = base_dir / "outputs_p0001_temporal"
    cohort_prior = build_left_cohort_prior(base_dir, exclude_patient="p0001")
    views = {
        series: extract_view(
            series,
            temporal_dir / f"p0001_{series}_temporal_refined_mask.png"
            if (temporal_dir / f"p0001_{series}_temporal_refined_mask.png").exists()
            else base_dir / "outputs_p0001_multiseries" / f"p0001_{series}_fused_mask.png",
            enhancement_path=temporal_dir / f"p0001_{series}_temporal_max_enhancement.png",
            peak_time_path=temporal_dir / f"p0001_{series}_temporal_peak_time.png",
            mask_support_path=temporal_dir / f"p0001_{series}_mask_temporal_support.png",
            mask_time_path=temporal_dir / f"p0001_{series}_mask_time_centroid_gray.png",
            cohort_prior=cohort_prior,
        )
        for series in LEFT_SERIES
    }

    view_a = views[series_a]
    view_b = views[series_b]
    bif_pairs, _bif_meta, _ = match_bifurcations(view_a, view_b)
    segments = build_proximal_segment_matches(view_a, view_b, bif_pairs)
    geom_eval = evaluate_proximal_geometry(segments, n_samples=12)
    if geom_eval is None:
        raise RuntimeError("failed to build proximal calibrated scaffold")
    edges = build_projective_edges_from_segments(geom_eval["F"], segments)

    case_id = normalize_case_id(series_a, series_b)
    case_manifest = load_json(base_dir / "reconstruction_cases" / case_id / "case_manifest.json")
    phase_pair = case_manifest["best_phase_pair"]
    geometry = case_manifest["geometry"]
    spacing_a_mm = estimate_isocenter_spacing_mm(geometry, "a")
    spacing_b_mm = estimate_isocenter_spacing_mm(geometry, "b")

    phase_mask_a = np.array(Image.open(base_dir / "reconstruction_cases" / case_id / phase_pair["mask_a_path"]).convert("L"), dtype=np.uint8) > 0
    phase_mask_b = np.array(Image.open(base_dir / "reconstruction_cases" / case_id / phase_pair["mask_b_path"]).convert("L"), dtype=np.uint8) > 0
    dt_a = ndi.distance_transform_edt(phase_mask_a)
    dt_b = ndi.distance_transform_edt(phase_mask_b)

    desired_lengths_mm = []
    current_lengths = []
    overlay_a = []
    overlay_b = []
    segment_summaries = []
    for idx, (segment, edge) in enumerate(zip(segments, edges)):
        n = len(edge.points3d)
        samples_a = resample_curve(segment.points_a, n)
        samples_b = resample_curve(segment.points_b, n)
        radii_px_a = sample_distance_profile(dt_a, samples_a)
        radii_px_b = sample_distance_profile(dt_b, samples_b)
        radii_mm_a = radii_px_a * spacing_a_mm
        radii_mm_b = radii_px_b * spacing_b_mm
        radii_mm = np.clip(0.5 * (radii_mm_a + radii_mm_b), 0.45, 2.2)
        edge.points2d = samples_a.copy()
        edge.radii = radii_mm.astype(np.float32)

        desired_length_mm = 0.5 * (float(segment.length_a) * spacing_a_mm + float(segment.length_b) * spacing_b_mm)
        current_length = polyline_length_3d(edge.points3d)
        if current_length > 1e-6:
            desired_lengths_mm.append(desired_length_mm)
            current_lengths.append(current_length)

        color = COLORS[idx % len(COLORS)]
        overlay_a.append({
            "points": samples_a,
            "width_px": float(np.clip(np.mean(radii_px_a) * 2.0, 2.0, 10.0)),
            "color": color,
        })
        overlay_b.append({
            "points": samples_b,
            "width_px": float(np.clip(np.mean(radii_px_b) * 2.0, 2.0, 10.0)),
            "color": color,
        })
        segment_summaries.append({
            "index": idx,
            "bifurcation_level": int(segment.bifurcation_level),
            "length_a_px": float(segment.length_a),
            "length_b_px": float(segment.length_b),
            "radius_mm_mean": float(np.mean(radii_mm)),
            "radius_mm_min": float(np.min(radii_mm)),
            "radius_mm_max": float(np.max(radii_mm)),
        })

    scale_factor = 1.0
    if desired_lengths_mm and current_lengths:
        scale_factor = float(np.mean(np.array(desired_lengths_mm) / np.maximum(np.array(current_lengths), 1e-6)))
        for edge in edges:
            edge.points3d = edge.points3d * scale_factor

    mesh_path = outputs_dir / f"{case_id}_tube_graph.obj"
    centerlines_path = outputs_dir / f"{case_id}_centerlines.obj"
    preview_path = outputs_dir / f"{case_id}_preview.png"
    overlay_a_path = outputs_dir / f"{case_id}_overlay_a.png"
    overlay_b_path = outputs_dir / f"{case_id}_overlay_b.png"
    summary_path = outputs_dir / f"{case_id}_summary.json"

    write_obj_mesh(mesh_path, edges)
    write_centerlines_obj(centerlines_path, edges)
    render_preview(edges, preview_path, f"p0001 {series_a}-{series_b} calibrated tube scaffold")

    image_a_path = base_dir / "reconstruction_cases" / case_id / phase_pair["image_a_path"]
    image_b_path = base_dir / "reconstruction_cases" / case_id / phase_pair["image_b_path"]
    draw_polyline_overlay(image_a_path, overlay_a, overlay_a_path)
    draw_polyline_overlay(image_b_path, overlay_b, overlay_b_path)

    summary = {
        "case_id": case_id,
        "series": [series_a, series_b],
        "source_phase_pair": phase_pair,
        "reconstruction_mode": "projective_centerline_with_calibrated_radii",
        "geometry_eval": {
            "correspondences": int(geom_eval["correspondences"]),
            "inliers": int(geom_eval["inliers"]),
            "inlier_ratio": float(geom_eval["inlier_ratio"]),
            "median_sampson": float(geom_eval["median_sampson"]),
        },
        "spacing_mm": {
            "view_a_isocenter": spacing_a_mm,
            "view_b_isocenter": spacing_b_mm,
        },
        "scale_factor_to_mm": scale_factor,
        "segments": segment_summaries,
        "artifacts": {
            "tube_graph_obj": str(mesh_path.resolve()),
            "centerlines_obj": str(centerlines_path.resolve()),
            "preview_png": str(preview_path.resolve()),
            "overlay_a_png": str(overlay_a_path.resolve()),
            "overlay_b_png": str(overlay_b_path.resolve()),
        },
        "note": (
            "This is a proximal tube scaffold built from matched two-view centerlines. "
            "Radii are calibrated from 2D mask distance transforms and isocenter pixel spacing. "
            "3D centerline geometry is still projective, not fully metric CTA-equivalent anatomy."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
