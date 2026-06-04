#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a prepared NeCA case by reprojecting the reconstructed volume.")
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--threshold-steps", type=int, default=64)
    return parser.parse_args()


def load_case(case_dir: Path) -> tuple[dict, dict, np.ndarray, np.ndarray | None]:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    with (case_dir / "data" / "config.yml").open("r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    target = np.load(case_dir / "data" / "CCTA_test" / "data.npy").astype(np.float32)
    hard_masks_path = case_dir / "data" / "CCTA_test" / "hard_masks.npy"
    hard_masks = np.load(hard_masks_path).astype(np.float32) if hard_masks_path.exists() else None
    return manifest, data_config, target, hard_masks


def load_latest_volume(case_dir: Path) -> tuple[Path, np.ndarray]:
    log_root = case_dir / "logs"
    eval_files = sorted(log_root.glob("*/eval/*.npy"), key=lambda p: int(p.stem))
    if not eval_files:
        raise FileNotFoundError(f"No eval npy files found under {log_root}")
    path = eval_files[-1]
    return path, np.load(path).astype(np.float32)


def build_projectors(repo_dir: Path, data_config: dict):
    import sys
    sys.path.insert(0, str(repo_dir))
    from src.render.ct_geometry_projector import ConeBeam3DProjector
    from odl.applications.tomo.util.utility import axis_rotation, rotation_matrix_from_to

    def rotation_matrix_to_axis_angle(m):
        angle = np.arccos((m[0, 0] + m[1, 1] + m[2, 2] - 1) / 2)
        denom = np.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2)
        x = (m[2, 1] - m[1, 2]) / denom
        y = (m[0, 2] - m[2, 0]) / denom
        z = (m[1, 0] - m[0, 1]) / denom
        return (x, y, z), angle

    def make_projector(primary: float, secondary: float, dso: float, dde: float):
        image_size = np.array(data_config["nVoxel"])
        image_reso = np.array(data_config["dVoxel"])
        proj_size = np.array(data_config["nDetector"])
        proj_reso = np.array(data_config["dDetector"])

        proj_angle = [-secondary, primary]
        from_source_vec = (0, -dso, 0)
        from_rot_vec = (-1, 0, 0)
        to_source_vec = axis_rotation((0, 0, 1), angle=proj_angle[0] / 180 * np.pi, vectors=from_source_vec)
        to_rot_vec = axis_rotation((0, 0, 1), angle=proj_angle[0] / 180 * np.pi, vectors=from_rot_vec)
        to_source_vec = axis_rotation(to_rot_vec[0], angle=proj_angle[1] / 180 * np.pi, vectors=to_source_vec[0])
        rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
        axis, angle = rotation_matrix_to_axis_angle(rot_mat)
        return ConeBeam3DProjector(image_size, image_reso, angle, axis, proj_size, proj_reso, dde, dso)

    first = make_projector(
        float(data_config["first_projection_angle"][0]),
        float(data_config["first_projection_angle"][1]),
        float(data_config["DSO"][0]),
        float(data_config["DDE"][0]),
    )
    second = make_projector(
        float(data_config["second_projection_angle"][0]),
        float(data_config["second_projection_angle"][1]),
        float(data_config["DSO"][1]),
        float(data_config["DDE"][1]),
    )
    return first, second


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def sweep_binary_metrics(pred: np.ndarray, target: np.ndarray, steps: int) -> dict:
    pred = normalize_image(pred)
    target = (target > 0.5).astype(np.uint8)
    best = None
    for threshold in np.linspace(0.05, 0.95, steps):
        binary = (pred >= threshold).astype(np.uint8)
        intersection = float((binary & target).sum())
        union = float((binary | target).sum())
        pred_sum = float(binary.sum())
        target_sum = float(target.sum())
        iou = intersection / union if union else 0.0
        denom = pred_sum + target_sum
        dice = (2.0 * intersection) / denom if denom else 0.0
        candidate = {
            "threshold": float(threshold),
            "iou": iou,
            "dice": dice,
            "pred_sum": pred_sum,
            "target_sum": target_sum,
        }
        if best is None or candidate["dice"] > best["dice"]:
            best = candidate
    assert best is not None
    return best


def save_overlay(path: Path, pred: np.ndarray, target: np.ndarray) -> None:
    pred_norm = normalize_image(pred)
    target = normalize_image(target)
    rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    rgb[..., 1] = np.clip(pred_norm * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 0] = np.clip(target * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def summarize_metrics(items: list[dict], prefix: str) -> dict:
    if not items:
        return {}
    keys = [
        ("mse_a", "mse_a"),
        ("mse_b", "mse_b"),
        ("mean_mse", "mean_mse"),
        ("iou_a", "iou_a"),
        ("iou_b", "iou_b"),
        ("dice_a", "dice_a"),
        ("dice_b", "dice_b"),
        ("mean_iou", "mean_iou"),
        ("mean_dice", "mean_dice"),
    ]
    out = {f"count_{prefix}": len(items)}
    for src_key, dst_key in keys:
        out[f"mean_{dst_key}_{prefix}"] = float(np.mean([item[src_key] for item in items]))
    return out


def main() -> None:
    args = parse_args()
    manifest, data_config, target, hard_masks = load_case(args.case_dir)
    volume_path, volume = load_latest_volume(args.case_dir)
    first, second = build_projectors(args.repo_dir, data_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(volume, dtype=torch.float32, device=device)[None, ...]
    proj_a = first.forward_project(tensor).detach().cpu().numpy().squeeze()
    proj_b = second.forward_project(tensor).detach().cpu().numpy().squeeze()
    proj_a_norm = normalize_image(proj_a)
    proj_b_norm = normalize_image(proj_b)

    if hard_masks is None:
        hard_masks = target

    eval_dir = args.case_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.save(eval_dir / "reprojection_a.npy", proj_a)
    np.save(eval_dir / "reprojection_b.npy", proj_b)

    hard_metrics = []
    training_metrics = []
    bundle_samples = []
    for sample_idx in range(target.shape[0]):
        train_target_a = target[sample_idx, 0]
        train_target_b = target[sample_idx, 1]
        hard_target_a = hard_masks[sample_idx, 0]
        hard_target_b = hard_masks[sample_idx, 1]

        train_mse_a = float(np.mean((proj_a_norm - train_target_a) ** 2))
        train_mse_b = float(np.mean((proj_b_norm - train_target_b) ** 2))
        hard_mse_a = float(np.mean((proj_a_norm - hard_target_a) ** 2))
        hard_mse_b = float(np.mean((proj_b_norm - hard_target_b) ** 2))

        train_best_a = sweep_binary_metrics(proj_a, train_target_a, args.threshold_steps)
        train_best_b = sweep_binary_metrics(proj_b, train_target_b, args.threshold_steps)
        hard_best_a = sweep_binary_metrics(proj_a, hard_target_a, args.threshold_steps)
        hard_best_b = sweep_binary_metrics(proj_b, hard_target_b, args.threshold_steps)

        phase_meta = manifest.get("phase_pairs", [])
        phase_payload = phase_meta[sample_idx] if sample_idx < len(phase_meta) else {"phase_index": sample_idx + 1}

        training_item = {
            "sample_index": sample_idx,
            "phase_index": int(phase_payload.get("phase_index", sample_idx + 1)),
            "mse_a": train_mse_a,
            "mse_b": train_mse_b,
            "mean_mse": (train_mse_a + train_mse_b) / 2.0,
            "iou_a": train_best_a["iou"],
            "iou_b": train_best_b["iou"],
            "dice_a": train_best_a["dice"],
            "dice_b": train_best_b["dice"],
            "mean_iou": (train_best_a["iou"] + train_best_b["iou"]) / 2.0,
            "mean_dice": (train_best_a["dice"] + train_best_b["dice"]) / 2.0,
            "threshold_a": train_best_a["threshold"],
            "threshold_b": train_best_b["threshold"],
        }
        hard_item = {
            "sample_index": sample_idx,
            "phase_index": int(phase_payload.get("phase_index", sample_idx + 1)),
            "mse_a": hard_mse_a,
            "mse_b": hard_mse_b,
            "mean_mse": (hard_mse_a + hard_mse_b) / 2.0,
            "iou_a": hard_best_a["iou"],
            "iou_b": hard_best_b["iou"],
            "dice_a": hard_best_a["dice"],
            "dice_b": hard_best_b["dice"],
            "mean_iou": (hard_best_a["iou"] + hard_best_b["iou"]) / 2.0,
            "mean_dice": (hard_best_a["dice"] + hard_best_b["dice"]) / 2.0,
            "threshold_a": hard_best_a["threshold"],
            "threshold_b": hard_best_b["threshold"],
        }
        training_metrics.append(training_item)
        hard_metrics.append(hard_item)

        overlay_a_path = eval_dir / f"overlay_a_{sample_idx + 1:02d}.png"
        overlay_b_path = eval_dir / f"overlay_b_{sample_idx + 1:02d}.png"
        save_overlay(overlay_a_path, proj_a, hard_target_a)
        save_overlay(overlay_b_path, proj_b, hard_target_b)
        bundle_samples.append({
            "sample_index": sample_idx,
            "phase_index": int(phase_payload.get("phase_index", sample_idx + 1)),
            "frame_a": phase_payload.get("frame_a"),
            "frame_b": phase_payload.get("frame_b"),
            "phase_gap": phase_payload.get("phase_gap"),
            "metrics_hard": hard_item,
            "metrics_training_target": training_item,
            "artifacts": {
                "overlay_a": str(overlay_a_path.resolve()),
                "overlay_b": str(overlay_b_path.resolve()),
            },
        })

    volume_stats = {
        "min": float(volume.min()),
        "max": float(volume.max()),
        "mean": float(volume.mean()),
        "p95": float(np.percentile(volume, 95)),
        "p99": float(np.percentile(volume, 99)),
        "p995": float(np.percentile(volume, 99.5)),
        "p999": float(np.percentile(volume, 99.9)),
    }
    threshold_values = {
        "abs_0_5": 0.5,
        "p99": volume_stats["p99"],
        "p995": volume_stats["p995"],
        "p999": volume_stats["p999"],
    }
    occupancy_summaries = {}
    for name, threshold in threshold_values.items():
        binary_volume = volume >= float(threshold)
        _, num_components = ndimage.label(binary_volume)
        occupancy_summaries[name] = {
            "threshold": float(threshold),
            "occupied_voxels": int(binary_volume.sum()),
            "component_count": int(num_components),
        }

    summary = {
        "case_id": manifest["case_id"],
        "phase_index": manifest["phase_index"],
        "selected_phase_indices": manifest.get("selected_phase_indices", [manifest["phase_index"]]),
        "bundle_size": manifest.get("bundle_size", target.shape[0]),
        "input_mode": manifest["input_mode"],
        "latest_volume": str(volume_path),
        "final_volume_shape": list(volume.shape),
        "training_target_metrics": {
            **summarize_metrics(training_metrics, "training_target"),
            "per_sample": training_metrics,
        },
        "hard_mask_metrics": {
            **summarize_metrics(hard_metrics, "hard_mask"),
            "per_sample": hard_metrics,
        },
        "volume_metrics": {
            "stats": volume_stats,
            "occupancy_by_threshold": occupancy_summaries,
        },
        "artifacts": {
            "reprojection_a": str((eval_dir / "reprojection_a.npy").resolve()),
            "reprojection_b": str((eval_dir / "reprojection_b.npy").resolve()),
        },
        "bundle_samples": bundle_samples,
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
