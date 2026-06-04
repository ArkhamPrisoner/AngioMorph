#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from reconstruct_coronary_tree import (
    EdgeModel,
    TreeModel,
    load_binary_mask,
    process_mask,
    register_mask_translation,
    save_binary,
    save_confidence_map,
    transform_edges,
)


PATIENT_ID = "p0001"
LEFT_SERIES = ("00000001", "00000002", "00000003")
RIGHT_SERIES = ("00000004",)


@dataclass
class UniqueFrame:
    series: str
    frame: str
    variant: int
    image_path: Path
    mask_path: Path
    confidence_path: Path | None
    annotation_ids: list[str]
    annotation_mask_paths: list[Path]
    area: int


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def parse_name(path: Path) -> tuple[str, str, str, str]:
    patient, series, frame, annotator = path.stem.split("_")
    return patient, series, frame, annotator


def annotation_union(mask_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    masks = [load_binary_mask(path) for path in mask_paths]
    counts = np.zeros(masks[0].shape, dtype=np.uint16)
    for mask in masks:
        counts += mask.astype(np.uint16)
    consensus = counts >= 1
    return consensus, counts


def build_unique_patient_set(base_dir: Path, patient_id: str) -> tuple[list[UniqueFrame], Path]:
    dataset_dir = base_dir / "все" / "our_data_with_dublicates_297img"
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    unique_dir = base_dir / f"{patient_id}_unique"
    unique_image_dir = unique_dir / "images"
    unique_mask_dir = unique_dir / "masks"
    unique_conf_dir = unique_dir / "mask_confidence"
    unique_image_dir.mkdir(parents=True, exist_ok=True)
    unique_mask_dir.mkdir(parents=True, exist_ok=True)
    unique_conf_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[tuple[Path, Path, str]]] = defaultdict(list)
    for image_path in sorted(image_dir.glob(f"{patient_id}_*.png")):
        _, series, frame, annotator = parse_name(image_path)
        grouped[(series, frame)].append((image_path, mask_dir / image_path.name, annotator))

    unique_frames: list[UniqueFrame] = []
    summary: list[dict[str, object]] = []

    for (series, frame), items in sorted(grouped.items()):
        by_hash: dict[str, list[tuple[Path, Path, str]]] = defaultdict(list)
        for image_path, local_mask_path, annotator in items:
            by_hash[md5(image_path)].append((image_path, local_mask_path, annotator))

        for variant_idx, hash_group in enumerate(sorted(by_hash.values(), key=lambda rows: rows[0][0].name), start=1):
            rep_image_path = hash_group[0][0]
            mask_paths = [row[1] for row in hash_group]
            annotation_ids = [row[2] for row in hash_group]
            consensus_mask, counts = annotation_union(mask_paths)
            area = int(consensus_mask.sum())

            suffix = f"_v{variant_idx}" if len(by_hash) > 1 else ""
            stem = f"{patient_id}_{series}_{frame}{suffix}"
            out_image_path = unique_image_dir / f"{stem}.png"
            out_mask_path = unique_mask_dir / f"{stem}.png"
            out_conf_path = unique_conf_dir / f"{stem}.png" if len(mask_paths) > 1 else None

            out_image_path.write_bytes(rep_image_path.read_bytes())
            save_binary(consensus_mask, out_mask_path)
            if out_conf_path is not None:
                save_confidence_map(counts, out_conf_path)

            unique_frames.append(
                UniqueFrame(
                    series=series,
                    frame=frame,
                    variant=variant_idx,
                    image_path=out_image_path,
                    mask_path=out_mask_path,
                    confidence_path=out_conf_path,
                    annotation_ids=annotation_ids,
                    annotation_mask_paths=mask_paths,
                    area=area,
                )
            )
            summary.append(
                {
                    "series": series,
                    "frame": frame,
                    "variant": variant_idx,
                    "source_image": str(rep_image_path),
                    "source_masks": [str(path) for path in mask_paths],
                    "annotation_ids": annotation_ids,
                    "consensus_mask": str(out_mask_path),
                    "confidence_map": str(out_conf_path) if out_conf_path else None,
                    "area": area,
                }
            )

    (unique_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return unique_frames, unique_dir


def mask_pca_angle(mask: np.ndarray) -> float:
    coords = np.argwhere(mask)
    if len(coords) < 3:
        return 0.0
    pts = coords[:, ::-1].astype(float)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, np.argmax(values)]
    return math.degrees(math.atan2(major[1], major[0]))


def process_fused_mask_paths(
    name: str,
    side: str,
    mask_paths: list[Path],
    output_dir: Path,
) -> tuple[TreeModel, dict[str, Path | list[dict[str, object]]]]:
    masks = {path: load_binary_mask(path) for path in mask_paths}
    areas = {path: int(mask.sum()) for path, mask in masks.items()}
    ref_path = max(mask_paths, key=lambda path: areas[path])
    reference = masks[ref_path]
    max_area = areas[ref_path]

    counts = np.zeros(reference.shape, dtype=np.uint16)
    alignment: list[dict[str, object]] = []
    selected = [path for path in mask_paths if areas[path] >= 0.55 * max_area]
    for path in selected:
        score, dr, dc, shifted = register_mask_translation(masks[path], reference, search_radius=20)
        shift_norm = float(math.hypot(dr, dc))
        keep = path == ref_path or (score >= 0.18 and shift_norm <= 120.0)
        if keep:
            counts += shifted.astype(np.uint16)
        alignment.append(
            {
                "mask": path.name,
                "area": areas[path],
                "iou_after_shift": round(score, 4),
                "shift_dr": dr,
                "shift_dc": dc,
                "shift_norm": round(shift_norm, 3),
                "kept": keep,
            }
        )

    fused_mask = np.logical_or(reference, counts >= 2)
    fused_mask_path = output_dir / f"{name}_mask.png"
    confidence_path = output_dir / f"{name}_confidence.png"
    alignment_path = output_dir / f"{name}_alignment.json"

    save_binary(fused_mask, fused_mask_path)
    save_confidence_map(counts, confidence_path)
    alignment_path.write_text(json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8")
    tree = process_mask(name, side, fused_mask, fused_mask_path, f"fused_from_{ref_path.stem}", output_dir)
    return tree, {
        "mask": fused_mask_path,
        "confidence": confidence_path,
        "alignment_json": alignment_path,
    }


def choose_best_mask_path(mask_paths: list[Path]) -> Path:
    scored = [(int(load_binary_mask(path).sum()), path) for path in mask_paths]
    return max(scored)[1]


def render_bundle_preview(bundle_edges: list[tuple[str, list[EdgeModel], str]], out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    all_points = []
    for label, edges, color in bundle_edges:
        first = True
        for edge in edges:
            pts = edge.points3d
            all_points.append(pts)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=color,
                linewidth=1.4,
                alpha=0.9,
                label=label if first else None,
            )
            first = False
    ax.set_title(title)
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-61)
    points = np.concatenate(all_points, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(maxs - mins) * 0.58
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_bundle_obj(path: Path, all_edges: list[EdgeModel]) -> None:
    from reconstruct_coronary_tree import write_obj_mesh

    write_obj_mesh(path, all_edges)


def build_left_multiseries_bundle(series_trees: dict[str, TreeModel], output_dir: Path) -> dict[str, str]:
    reference_series = LEFT_SERIES[0]
    reference_tree = series_trees[reference_series]
    reference_mask = load_binary_mask(reference_tree.mask_path)
    reference_angle = mask_pca_angle(reference_mask)

    colors = {
        "00000001": "#b81414",
        "00000002": "#0f8b8d",
        "00000003": "#d17b0f",
    }

    bundle_edges: list[tuple[str, list[EdgeModel], str]] = []
    all_edges: list[EdgeModel] = []
    alignment_summary: list[dict[str, object]] = []

    for series in LEFT_SERIES:
        tree = series_trees[series]
        mask = load_binary_mask(tree.mask_path)
        angle = mask_pca_angle(mask)
        rotation = reference_angle - angle
        transformed = transform_edges(
            tree.edges,
            translate=np.zeros(3, dtype=float),
            root_xyz=tree.root_xyz,
            rotation_deg=rotation,
        )
        bundle_edges.append((series, transformed, colors[series]))
        all_edges.extend(transformed)
        alignment_summary.append(
            {
                "series": series,
                "mask": str(tree.mask_path),
                "pca_angle_deg": round(angle, 3),
                "rotation_to_reference_deg": round(rotation, 3),
            }
        )

    preview_path = output_dir / "p0001_left_multiseries_bundle_preview.png"
    obj_path = output_dir / "p0001_left_multiseries_bundle.obj"
    alignment_path = output_dir / "p0001_left_multiseries_bundle_alignment.json"
    render_bundle_preview(bundle_edges, preview_path, "p0001 left multiseries bundle")
    write_bundle_obj(obj_path, all_edges)
    alignment_path.write_text(json.dumps(alignment_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "preview": str(preview_path),
        "mesh": str(obj_path),
        "alignment_json": str(alignment_path),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs_p0001_multiseries"
    output_dir.mkdir(exist_ok=True)

    unique_frames, unique_dir = build_unique_patient_set(base_dir, PATIENT_ID)
    by_series: dict[str, list[UniqueFrame]] = defaultdict(list)
    for frame in unique_frames:
        by_series[frame.series].append(frame)

    series_models: dict[str, dict[str, object]] = {}
    left_fused_trees: dict[str, TreeModel] = {}
    right_fused_trees: dict[str, TreeModel] = {}

    for series in LEFT_SERIES + RIGHT_SERIES:
        local_frames = sorted(by_series[series], key=lambda item: item.frame)
        mask_paths = [item.mask_path for item in local_frames]
        side = "left" if series in LEFT_SERIES else "right"

        best_mask_path = choose_best_mask_path(mask_paths)
        best_tree = process_mask(
            f"{PATIENT_ID}_{series}_best",
            side,
            load_binary_mask(best_mask_path),
            best_mask_path,
            best_mask_path.stem,
            output_dir,
        )
        fused_tree, fused_assets = process_fused_mask_paths(
            f"{PATIENT_ID}_{series}_fused",
            side,
            mask_paths,
            output_dir,
        )
        series_models[series] = {
            "side": side,
            "unique_frames": [item.frame for item in local_frames],
            "best_tree": {
                "frame": best_tree.frame,
                "mask": str(best_tree.mask_path),
                "preview": str(best_tree.preview_path),
                "mesh": str(best_tree.obj_path),
            },
            "fused_tree": {
                "frame": fused_tree.frame,
                "mask": str(fused_tree.mask_path),
                "preview": str(fused_tree.preview_path),
                "mesh": str(fused_tree.obj_path),
                "confidence_map": str(fused_assets["confidence"]),
                "alignment_json": str(fused_assets["alignment_json"]),
            },
        }
        if series in LEFT_SERIES:
            left_fused_trees[series] = fused_tree
        else:
            right_fused_trees[series] = fused_tree

    left_bundle = build_left_multiseries_bundle(left_fused_trees, output_dir)

    summary = {
        "patient": PATIENT_ID,
        "unique_data_dir": str(unique_dir),
        "series": series_models,
        "left_multiseries_bundle": left_bundle,
        "note": (
            "Series 00000001/00000002/00000003 are treated as three separate left-coronary views. "
            "This remains an uncalibrated pseudo-3D bundle because C-arm geometry is unavailable."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
