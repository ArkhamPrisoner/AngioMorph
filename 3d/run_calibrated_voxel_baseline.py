#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required for the calibrated baseline") from exc


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_case_path(case_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return case_root / path
    if path.exists():
        return path
    parts = list(path.parts)
    if "reconstruction_cases" in parts:
        idx = parts.index("reconstruction_cases")
        tail = parts[idx + 2 :]
        if tail:
            return case_root / Path(*tail)
    return path


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    return (load_gray(path) > 0.5).astype(np.float32)


def rotation_x(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rotation_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rotation_z(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def build_rotation(primary: float, secondary: float, order: str) -> np.ndarray:
    primary_rot = rotation_y(primary)
    secondary_rot = rotation_x(secondary)
    if order == "xy":
        return secondary_rot @ primary_rot
    if order == "yx":
        return primary_rot @ secondary_rot
    if order == "xz":
        return rotation_z(secondary) @ primary_rot
    raise ValueError(f"Unknown rotation order: {order}")


@dataclass
class ProjectionGeometry:
    primary_angle: float
    secondary_angle: float
    source_to_detector_mm: float
    source_to_patient_mm: float
    pixel_spacing_mm: float
    rows: int
    cols: int
    rotation_order: str
    flip_u: bool
    flip_v: bool

    def project(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotation = build_rotation(self.primary_angle, self.secondary_angle, self.rotation_order)
        camera = points_xyz @ rotation.T
        camera[:, 2] += self.source_to_patient_mm

        valid = camera[:, 2] > 1e-3
        u = np.full(points_xyz.shape[0], -1.0, dtype=np.float32)
        v = np.full(points_xyz.shape[0], -1.0, dtype=np.float32)

        scale = self.source_to_detector_mm / np.maximum(camera[:, 2], 1e-3)
        mm_u = camera[:, 0] * scale
        mm_v = camera[:, 1] * scale

        if self.flip_u:
            mm_u = -mm_u
        if self.flip_v:
            mm_v = -mm_v

        u = mm_u / self.pixel_spacing_mm + (self.cols - 1) / 2.0
        v = mm_v / self.pixel_spacing_mm + (self.rows - 1) / 2.0
        inside = valid & (u >= 0) & (u < self.cols) & (v >= 0) & (v < self.rows)
        return u, v, inside


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h, w = image.shape
    values = np.zeros_like(u, dtype=np.float32)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return values

    uu = u[idx]
    vv = v[idx]
    x0 = np.floor(uu).astype(np.int32)
    y0 = np.floor(vv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    dx = uu - x0
    dy = vv - y0

    values[idx] = (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x1] * dx * (1 - dy)
        + image[y1, x0] * (1 - dx) * dy
        + image[y1, x1] * dx * dy
    )
    return values


def make_soft_mask(mask: np.ndarray, sigma: float, dt_scale: float) -> np.ndarray:
    if sigma > 0:
        mask = ndi.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    dt = ndi.distance_transform_edt(mask > 0)
    if dt.max() > 0:
        dt = np.clip(dt / dt_scale, 0.0, 1.0)
    return np.maximum(mask, dt).astype(np.float32)


def generate_grid(extent_mm: float, resolution: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    axis = np.linspace(-extent_mm / 2.0, extent_mm / 2.0, resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="xy")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return points, (xx, yy, zz)


def rasterize_projection(points_xyz: np.ndarray, geometry: ProjectionGeometry, image_shape: tuple[int, int], radius_px: int) -> np.ndarray:
    u, v, valid = geometry.project(points_xyz)
    canvas = np.zeros(image_shape, dtype=np.uint8)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return canvas
    uu = np.round(u[idx]).astype(np.int32)
    vv = np.round(v[idx]).astype(np.int32)
    canvas[vv, uu] = 1
    if radius_px > 0:
        canvas = ndi.binary_dilation(canvas, iterations=radius_px).astype(np.uint8)
    return canvas


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0
    b_bin = b > 0
    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    return float(inter / union) if union > 0 else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0
    b_bin = b > 0
    inter = np.logical_and(a_bin, b_bin).sum()
    denom = a_bin.sum() + b_bin.sum()
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask, 0
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return (labeled == keep).astype(np.uint8), int(num)


def save_overlay(rendered: np.ndarray, target_mask: np.ndarray, image_gray: np.ndarray, path: Path) -> None:
    image = np.clip(image_gray * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([image, image, image], axis=-1)
    pred = rendered > 0
    gt = target_mask > 0
    rgb[np.logical_and(gt, pred)] = [0, 255, 0]
    rgb[np.logical_and(~gt, pred)] = [255, 0, 0]
    rgb[np.logical_and(gt, ~pred)] = [0, 0, 255]
    Image.fromarray(rgb).save(path)


def write_point_cloud(points_xyz: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in points_xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def evaluate_one_run(
    pair_manifest: dict[str, Any],
    case_manifest: dict[str, Any],
    case_root: Path,
    output_dir: Path,
    method_id: str,
    resolution: int,
    extent_mm: float,
    score_threshold: float,
    sigma: float,
    dt_scale: float,
    rotation_order: str,
    flip_u: bool,
    flip_v: bool,
    splat_radius: int,
) -> dict[str, Any]:
    image_a = load_gray(resolve_case_path(case_root, pair_manifest["image_a_path"]))
    image_b = load_gray(resolve_case_path(case_root, pair_manifest["image_b_path"]))
    mask_a = load_mask(resolve_case_path(case_root, pair_manifest["mask_a_path"]))
    mask_b = load_mask(resolve_case_path(case_root, pair_manifest["mask_b_path"]))
    soft_a = make_soft_mask(mask_a, sigma=sigma, dt_scale=dt_scale)
    soft_b = make_soft_mask(mask_b, sigma=sigma, dt_scale=dt_scale)

    geom = case_manifest["geometry"]
    geom_a = ProjectionGeometry(
        primary_angle=float(geom["projection_a"]["positioner_primary_angle"]),
        secondary_angle=float(geom["projection_a"]["positioner_secondary_angle"]),
        source_to_detector_mm=float(geom["projection_a"]["distance_source_to_detector"]),
        source_to_patient_mm=float(geom["projection_a"]["distance_source_to_patient"]),
        pixel_spacing_mm=float(geom["projection_a"]["imager_pixel_spacing"][0]),
        rows=mask_a.shape[0],
        cols=mask_a.shape[1],
        rotation_order=rotation_order,
        flip_u=flip_u,
        flip_v=flip_v,
    )
    geom_b = ProjectionGeometry(
        primary_angle=float(geom["projection_b"]["positioner_primary_angle"]),
        secondary_angle=float(geom["projection_b"]["positioner_secondary_angle"]),
        source_to_detector_mm=float(geom["projection_b"]["distance_source_to_detector"]),
        source_to_patient_mm=float(geom["projection_b"]["distance_source_to_patient"]),
        pixel_spacing_mm=float(geom["projection_b"]["imager_pixel_spacing"][0]),
        rows=mask_b.shape[0],
        cols=mask_b.shape[1],
        rotation_order=rotation_order,
        flip_u=flip_u,
        flip_v=flip_v,
    )

    points_xyz, _ = generate_grid(extent_mm=extent_mm, resolution=resolution)
    ua, va, valid_a = geom_a.project(points_xyz)
    ub, vb, valid_b = geom_b.project(points_xyz)
    score_a = bilinear_sample(soft_a, ua, va, valid_a)
    score_b = bilinear_sample(soft_b, ub, vb, valid_b)
    score = np.minimum(score_a, score_b)
    occupancy = (score >= score_threshold).reshape(resolution, resolution, resolution).astype(np.uint8)
    occupancy, component_count = largest_component(occupancy)
    points_keep = points_xyz[occupancy.reshape(-1) > 0]

    rendered_a = rasterize_projection(points_keep, geom_a, mask_a.shape, radius_px=splat_radius)
    rendered_b = rasterize_projection(points_keep, geom_b, mask_b.shape, radius_px=splat_radius)

    iou_a = iou(rendered_a, mask_a)
    iou_b = iou(rendered_b, mask_b)
    dice_a = dice(rendered_a, mask_a)
    dice_b = dice(rendered_b, mask_b)
    mean_iou = (iou_a + iou_b) / 2.0
    mean_dice = (dice_a + dice_b) / 2.0
    component_penalty = max(component_count - 1, 0)
    score_total = mean_iou - 0.05 * component_penalty

    run_dir = output_dir / method_id
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "volume.npz", occupancy=occupancy, resolution=resolution, extent_mm=extent_mm)
    write_point_cloud(points_keep, run_dir / "point_cloud.ply")
    save_overlay(rendered_a, mask_a, image_a, run_dir / "overlay_a.png")
    save_overlay(rendered_b, mask_b, image_b, run_dir / "overlay_b.png")

    summary = {
        "method_id": method_id,
        "phase_pair": {
            "frame_a": pair_manifest["frame_a"],
            "frame_b": pair_manifest["frame_b"],
            "phase_gap": pair_manifest["phase_gap"],
        },
        "params": {
            "resolution": resolution,
            "extent_mm": extent_mm,
            "score_threshold": score_threshold,
            "sigma": sigma,
            "dt_scale": dt_scale,
            "rotation_order": rotation_order,
            "flip_u": flip_u,
            "flip_v": flip_v,
            "splat_radius": splat_radius,
        },
        "metrics": {
            "iou_a": round(iou_a, 6),
            "iou_b": round(iou_b, 6),
            "dice_a": round(dice_a, 6),
            "dice_b": round(dice_b, 6),
            "mean_iou": round(mean_iou, 6),
            "mean_dice": round(mean_dice, 6),
            "component_count": component_count,
            "occupied_points": int(len(points_keep)),
            "score_total": round(score_total, 6),
        },
        "artifacts": {
            "volume_npz": str((run_dir / "volume.npz").resolve()),
            "point_cloud_ply": str((run_dir / "point_cloud.ply").resolve()),
            "overlay_a": str((run_dir / "overlay_a.png").resolve()),
            "overlay_b": str((run_dir / "overlay_b.png").resolve()),
        },
    }
    save_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    case_manifest = load_json(args.case_manifest)
    case_root = args.case_manifest.parent
    phase_pairs = case_manifest["phase_pairs"]

    candidates = [
        {"method_id": "hard_xy", "resolution": 96, "extent_mm": 180.0, "score_threshold": 0.95, "sigma": 0.0, "dt_scale": 6.0, "rotation_order": "xy", "flip_u": False, "flip_v": False, "splat_radius": 1},
        {"method_id": "soft_xy", "resolution": 96, "extent_mm": 180.0, "score_threshold": 0.25, "sigma": 1.0, "dt_scale": 8.0, "rotation_order": "xy", "flip_u": False, "flip_v": False, "splat_radius": 1},
        {"method_id": "soft_xy_flipu", "resolution": 96, "extent_mm": 180.0, "score_threshold": 0.25, "sigma": 1.0, "dt_scale": 8.0, "rotation_order": "xy", "flip_u": True, "flip_v": False, "splat_radius": 1},
        {"method_id": "soft_yx", "resolution": 96, "extent_mm": 180.0, "score_threshold": 0.25, "sigma": 1.0, "dt_scale": 8.0, "rotation_order": "yx", "flip_u": False, "flip_v": False, "splat_radius": 1},
        {"method_id": "soft_yx_flipu", "resolution": 96, "extent_mm": 180.0, "score_threshold": 0.25, "sigma": 1.0, "dt_scale": 8.0, "rotation_order": "yx", "flip_u": True, "flip_v": False, "splat_radius": 1},
        {"method_id": "soft_highres", "resolution": 128, "extent_mm": 180.0, "score_threshold": 0.22, "sigma": 1.2, "dt_scale": 10.0, "rotation_order": "xy", "flip_u": False, "flip_v": False, "splat_radius": 1},
    ]

    all_runs = []
    for phase_index, pair_manifest in enumerate(phase_pairs, start=1):
        phase_dir = args.output_dir / f"phase_{phase_index:02d}"
        for candidate in candidates:
            summary = evaluate_one_run(
                pair_manifest=pair_manifest,
                case_manifest=case_manifest,
                case_root=case_root,
                output_dir=phase_dir,
                **candidate,
            )
            summary["phase_index"] = phase_index
            all_runs.append(summary)

    all_runs.sort(key=lambda x: x["metrics"]["score_total"], reverse=True)
    best = all_runs[0] if all_runs else None
    result = {
        "case_id": case_manifest["case_id"],
        "num_phase_pairs": len(phase_pairs),
        "num_runs": len(all_runs),
        "best_run": best,
        "runs": all_runs,
    }
    save_json(args.output_dir / "results.json", result)
    print(json.dumps({"case_id": case_manifest["case_id"], "num_runs": len(all_runs), "best_run": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
