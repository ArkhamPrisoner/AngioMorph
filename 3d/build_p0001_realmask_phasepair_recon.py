#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from build_p0001_projective_recon import (
    build_projective_edges,
    build_projective_edges_from_segments,
    build_proximal_segment_matches,
    evaluate_pair_geometry,
    evaluate_proximal_geometry,
    extract_view,
    match_bifurcations,
    match_leaf_paths,
    render_3d_preview,
    render_correspondence_preview,
    render_proximal_correspondence_preview,
    resample_curve,
)
from reconstruct_coronary_tree import write_centerlines_obj, write_obj_mesh


COLORS = [
    (220, 64, 64),
    (64, 160, 255),
    (80, 190, 120),
    (255, 180, 60),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def estimate_isocenter_spacing_mm(proj: dict) -> float:
    detector_spacing = float(proj['imager_pixel_spacing'][0])
    dso = float(proj['distance_source_to_patient'])
    dsd = float(proj['distance_source_to_detector'])
    return detector_spacing * dso / dsd


def sample_distance_profile(distance_map: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    rr = np.clip(np.round(points_xy[:, 1]).astype(int), 0, distance_map.shape[0] - 1)
    cc = np.clip(np.round(points_xy[:, 0]).astype(int), 0, distance_map.shape[1] - 1)
    return distance_map[rr, cc].astype(np.float32)


def draw_polyline_overlay(image_path: Path, polylines: list[dict], out_path: Path) -> None:
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image, 'RGBA')
    for item in polylines:
        color = item['color']
        points = [tuple(map(float, pt)) for pt in item['points']]
        width = max(2, int(round(item['width_px'])))
        if len(points) >= 2:
            draw.line(points, fill=color + (220,), width=width, joint='curve')
    image.save(out_path)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / 'outputs_p0001_realmask_phasepair'
    outputs_dir.mkdir(exist_ok=True)

    case_id = 'p0001__00000001__00000002'
    case_dir = base_dir / 'reconstruction_cases' / case_id
    case_manifest = load_json(case_dir / 'case_manifest.json')
    phase_pair = case_manifest['best_phase_pair']
    geometry = case_manifest['geometry']

    mask_a_path = case_dir / phase_pair['mask_a_path']
    mask_b_path = case_dir / phase_pair['mask_b_path']
    image_a_path = case_dir / phase_pair['image_a_path']
    image_b_path = case_dir / phase_pair['image_b_path']

    view_a = extract_view('00000001', mask_a_path)
    view_b = extract_view('00000002', mask_b_path)

    bif_pairs, bif_meta, _ = match_bifurcations(view_a, view_b)
    bif_map = {int(a.node): int(b.node) for a, b in bif_pairs}
    segments = build_proximal_segment_matches(view_a, view_b, bif_pairs)
    leaf_pairs, match_meta, _ = match_leaf_paths(view_a, view_b, bifurcation_map=bif_map)

    proximal_eval = evaluate_proximal_geometry(segments, n_samples=12) if segments else None
    leaf_eval = evaluate_pair_geometry(leaf_pairs, n_samples=10) if leaf_pairs else None

    if proximal_eval is not None:
        reconstruction_mode = 'proximal_segments'
        edges = build_projective_edges_from_segments(proximal_eval['F'], segments)
        render_proximal_correspondence_preview(view_a, view_b, segments, outputs_dir / 'matches.png')
        geom_eval = proximal_eval
    elif leaf_eval is not None:
        reconstruction_mode = 'leaf_paths'
        edges = build_projective_edges(leaf_eval['F'], leaf_pairs)
        render_correspondence_preview(view_a, view_b, leaf_pairs, outputs_dir / 'matches.png')
        geom_eval = leaf_eval
    else:
        raise RuntimeError('No viable reconstruction from real phase masks only')

    spacing_a_mm = estimate_isocenter_spacing_mm(geometry['projection_a'])
    spacing_b_mm = estimate_isocenter_spacing_mm(geometry['projection_b'])
    phase_mask_a = np.array(Image.open(mask_a_path).convert('L'), dtype=np.uint8) > 0
    phase_mask_b = np.array(Image.open(mask_b_path).convert('L'), dtype=np.uint8) > 0
    dt_a = ndi.distance_transform_edt(phase_mask_a)
    dt_b = ndi.distance_transform_edt(phase_mask_b)

    overlay_a = []
    overlay_b = []
    segment_summaries = []
    if reconstruction_mode == 'proximal_segments':
        source_units = segments
        for idx, (segment, edge) in enumerate(zip(segments, edges)):
            n = len(edge.points3d)
            samples_a = resample_curve(segment.points_a, n)
            samples_b = resample_curve(segment.points_b, n)
            radii_px_a = sample_distance_profile(dt_a, samples_a)
            radii_px_b = sample_distance_profile(dt_b, samples_b)
            radii_mm = np.clip(0.5 * (radii_px_a * spacing_a_mm + radii_px_b * spacing_b_mm), 0.45, 2.2)
            edge.points2d = samples_a.copy()
            edge.radii = radii_mm.astype(np.float32)
            color = COLORS[idx % len(COLORS)]
            overlay_a.append({'points': samples_a, 'width_px': float(np.clip(np.mean(radii_px_a) * 2.0, 2.0, 10.0)), 'color': color})
            overlay_b.append({'points': samples_b, 'width_px': float(np.clip(np.mean(radii_px_b) * 2.0, 2.0, 10.0)), 'color': color})
            segment_summaries.append({
                'index': idx,
                'bifurcation_level': int(segment.bifurcation_level),
                'length_a_px': float(segment.length_a),
                'length_b_px': float(segment.length_b),
                'radius_mm_mean': float(np.mean(radii_mm)),
            })
    else:
        source_units = leaf_pairs
        for idx, ((leaf_a, leaf_b), edge) in enumerate(zip(leaf_pairs, edges)):
            n = len(edge.points3d)
            samples_a = resample_curve(leaf_a.points, n)
            samples_b = resample_curve(leaf_b.points, n)
            radii_px_a = sample_distance_profile(dt_a, samples_a)
            radii_px_b = sample_distance_profile(dt_b, samples_b)
            radii_mm = np.clip(0.5 * (radii_px_a * spacing_a_mm + radii_px_b * spacing_b_mm), 0.45, 2.2)
            edge.points2d = samples_a.copy()
            edge.radii = radii_mm.astype(np.float32)
            color = COLORS[idx % len(COLORS)]
            overlay_a.append({'points': samples_a, 'width_px': float(np.clip(np.mean(radii_px_a) * 2.0, 2.0, 10.0)), 'color': color})
            overlay_b.append({'points': samples_b, 'width_px': float(np.clip(np.mean(radii_px_b) * 2.0, 2.0, 10.0)), 'color': color})
            segment_summaries.append({
                'index': idx,
                'leaf_a': int(leaf_a.leaf),
                'leaf_b': int(leaf_b.leaf),
                'length_a_px': float(leaf_a.length),
                'length_b_px': float(leaf_b.length),
                'radius_mm_mean': float(np.mean(radii_mm)),
            })

    mesh_path = outputs_dir / f'{case_id}_realmask_tube_graph.obj'
    centerlines_path = outputs_dir / f'{case_id}_realmask_centerlines.obj'
    preview_path = outputs_dir / f'{case_id}_realmask_preview.png'
    overlay_a_path = outputs_dir / f'{case_id}_realmask_overlay_a.png'
    overlay_b_path = outputs_dir / f'{case_id}_realmask_overlay_b.png'
    write_obj_mesh(mesh_path, edges)
    write_centerlines_obj(centerlines_path, edges)
    render_3d_preview(edges, preview_path, f'{case_id} real-mask {reconstruction_mode}')
    draw_polyline_overlay(image_a_path, overlay_a, overlay_a_path)
    draw_polyline_overlay(image_b_path, overlay_b, overlay_b_path)

    summary = {
        'case_id': case_id,
        'source_phase_pair': phase_pair,
        'inputs': {
            'mask_a': str(mask_a_path.resolve()),
            'mask_b': str(mask_b_path.resolve()),
            'image_a': str(image_a_path.resolve()),
            'image_b': str(image_b_path.resolve()),
        },
        'reconstruction_mode': reconstruction_mode,
        'match_counts': {
            'bifurcations': len(bif_pairs),
            'proximal_segments': len(segments),
            'leaf_pairs': len(leaf_pairs),
        },
        'matching_meta': {
            'bifurcation_meta': bif_meta,
            'leaf_match_meta': match_meta,
        },
        'geometry_eval': {
            'correspondences': int(geom_eval['correspondences']),
            'inliers': int(geom_eval['inliers']),
            'inlier_ratio': float(geom_eval['inlier_ratio']),
            'median_sampson': float(geom_eval['median_sampson']),
        },
        'spacing_mm': {
            'view_a_isocenter': spacing_a_mm,
            'view_b_isocenter': spacing_b_mm,
        },
        'segments': segment_summaries,
        'artifacts': {
            'tube_graph_obj': str(mesh_path.resolve()),
            'centerlines_obj': str(centerlines_path.resolve()),
            'preview_png': str(preview_path.resolve()),
            'overlay_a_png': str(overlay_a_path.resolve()),
            'overlay_b_png': str(overlay_b_path.resolve()),
            'matches_png': str((outputs_dir / 'matches.png').resolve()),
        },
        'note': 'Matching and reconstruction are computed only from the real selected phase-pair masks, without temporal_refined or fused masks.'
    }
    (outputs_dir / f'{case_id}_realmask_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
