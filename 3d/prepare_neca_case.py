#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_DIR = ROOT / "reconstruction_cases"
DEFAULT_OUTPUT_DIR = ROOT / "neca_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a NeCA-compatible case from exported reconstruction pairs.")
    parser.add_argument("--case-id", required=True, help="Case identifier from reconstruction_cases, e.g. p0001__00000001__00000002")
    parser.add_argument("--phase-index", type=int, default=1, help="1-based phase pair index to use as bundle center")
    parser.add_argument("--input-mode", choices=("mask", "image", "masked_image", "edt_mask", "blurred_mask"), default="mask")
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--detector-size", type=int, default=256, help="Square detector resolution for NeCA")
    parser.add_argument("--volume-size", type=int, default=128, help="Cubic voxel grid resolution")
    parser.add_argument("--volume-extent-mm", type=float, default=180.0, help="Physical cubic extent in mm")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--bound", type=float, default=0.3)
    parser.add_argument("--bundle-size", type=int, default=1, help="Number of phase-aligned frame pairs to include in one training bundle")
    parser.add_argument("--blur-sigma", type=float, default=2.5, help="Gaussian sigma in pixels for blurred targets")
    parser.add_argument("--edt-falloff-px", type=float, default=3.0, help="Outside-mask exponential falloff in pixels for EDT target")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resize_stack(stack: np.ndarray, detector_size: int) -> np.ndarray:
    if stack.shape[-2:] == (detector_size, detector_size):
        return stack.astype(np.float32, copy=False)
    resized = np.zeros((stack.shape[0], stack.shape[1], detector_size, detector_size), dtype=np.float32)
    for batch_idx in range(stack.shape[0]):
        for proj_idx in range(stack.shape[1]):
            image = Image.fromarray((np.clip(stack[batch_idx, proj_idx], 0.0, 1.0) * 255.0).astype(np.uint8))
            image = image.resize((detector_size, detector_size), resample=Image.BILINEAR)
            resized[batch_idx, proj_idx] = np.asarray(image, dtype=np.float32) / 255.0
    return resized


def normalize01(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def soft_edt_target(mask: np.ndarray, edt_falloff_px: float) -> np.ndarray:
    binary = mask > 0.5
    outside = ndimage.distance_transform_edt(~binary).astype(np.float32)
    soft = np.exp(-outside / max(edt_falloff_px, 1e-3))
    soft[binary] = 1.0
    return soft.astype(np.float32)


def blurred_mask_target(mask: np.ndarray, blur_sigma: float) -> np.ndarray:
    blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=max(blur_sigma, 1e-3))
    return normalize01(blurred)


def transform_inputs(images: np.ndarray, masks: np.ndarray, input_mode: str, blur_sigma: float, edt_falloff_px: float) -> np.ndarray:
    if input_mode == "mask":
        return masks.astype(np.float32)
    if input_mode == "image":
        return images.astype(np.float32)
    if input_mode == "masked_image":
        return (images * masks).astype(np.float32)
    if input_mode == "edt_mask":
        out = np.zeros_like(masks, dtype=np.float32)
        for i in range(masks.shape[0]):
            for j in range(masks.shape[1]):
                out[i, j] = soft_edt_target(masks[i, j], edt_falloff_px)
        return out
    if input_mode == "blurred_mask":
        out = np.zeros_like(masks, dtype=np.float32)
        for i in range(masks.shape[0]):
            for j in range(masks.shape[1]):
                out[i, j] = blurred_mask_target(masks[i, j], blur_sigma)
        return out
    raise ValueError(f"Unsupported input_mode={input_mode}")


def select_phase_indices(phase_pairs: list[dict], center_phase_index: int, bundle_size: int) -> list[int]:
    if center_phase_index < 1 or center_phase_index > len(phase_pairs):
        raise ValueError(f"phase_index={center_phase_index} out of range 1..{len(phase_pairs)}")
    bundle_size = max(1, min(bundle_size, len(phase_pairs)))
    center_zero = center_phase_index - 1
    center_pair = phase_pairs[center_zero]
    center_phase = 0.5 * (float(center_pair["phase_a"]) + float(center_pair["phase_b"]))
    scored = []
    for idx, pair in enumerate(phase_pairs):
        pair_phase = 0.5 * (float(pair["phase_a"]) + float(pair["phase_b"]))
        scored.append((abs(pair_phase - center_phase), abs(idx - center_zero), idx))
    scored.sort()
    selected = sorted(idx for _, _, idx in scored[:bundle_size])
    return [idx + 1 for idx in selected]


def load_phase_stacks(case_dir: Path, phase_indices: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    images = []
    masks = []
    payloads = []
    manifest = load_json(case_dir / "case_manifest.json")
    phase_pairs = manifest["phase_pairs"]
    for phase_index in phase_indices:
        pair_dir = case_dir / "phase_pairs" / f"pair_{phase_index:02d}"
        images.append(np.load(pair_dir / "images.npy").astype(np.float32)[0])
        masks.append(np.load(pair_dir / "masks.npy").astype(np.float32)[0])
        payload = dict(phase_pairs[phase_index - 1])
        payload["phase_index"] = phase_index
        payloads.append(payload)
    return np.stack(images, axis=0), np.stack(masks, axis=0), payloads


def main() -> None:
    args = parse_args()
    case_dir = args.cases_dir / args.case_id
    manifest = load_json(case_dir / "case_manifest.json")
    phase_pairs = manifest["phase_pairs"]
    selected_phase_indices = select_phase_indices(phase_pairs, args.phase_index, args.bundle_size)
    images, masks, phase_payloads = load_phase_stacks(case_dir, selected_phase_indices)
    original_detector = images.shape[-1]

    images = resize_stack(images, args.detector_size)
    masks = resize_stack(masks, args.detector_size)
    projections = transform_inputs(images, masks, args.input_mode, args.blur_sigma, args.edt_falloff_px)

    bundle_tag = f"b{len(selected_phase_indices)}"
    out_dir = args.output_dir / args.case_id / f"phase_{args.phase_index:02d}_{bundle_tag}_{args.input_mode}_{args.detector_size}"
    data_dir = out_dir / "data" / "CCTA_test"
    config_dir = out_dir / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "data.npy"
    np.save(data_path, projections)
    np.save(data_dir / "hard_masks.npy", masks.astype(np.float32))
    np.save(data_dir / "raw_images.npy", images.astype(np.float32))

    geom = manifest["geometry"]
    proj_a = geom["projection_a"]
    proj_b = geom["projection_b"]

    scale = original_detector / args.detector_size
    pixel_spacing_a = float(proj_a["imager_pixel_spacing"][0]) * scale
    pixel_spacing_b = float(proj_b["imager_pixel_spacing"][0]) * scale
    if abs(pixel_spacing_a - pixel_spacing_b) > 1e-6:
        raise ValueError("Expected identical pixel spacing across both projections after resampling.")

    volume_spacing = args.volume_extent_mm / args.volume_size

    data_config = {
        "datadir": str(data_path.resolve()),
        "numTrain": int(projections.shape[0]),
        "DSD": [
            float(proj_a["distance_source_to_detector"]),
            float(proj_b["distance_source_to_detector"]),
        ],
        "DSO": [
            float(proj_a["distance_source_to_patient"]),
            float(proj_b["distance_source_to_patient"]),
        ],
        "DDE": [
            float(proj_a["distance_source_to_detector"]) - float(proj_a["distance_source_to_patient"]),
            float(proj_b["distance_source_to_detector"]) - float(proj_b["distance_source_to_patient"]),
        ],
        "nDetector": [args.detector_size, args.detector_size],
        "dDetector": [pixel_spacing_a, pixel_spacing_a],
        "nVoxel": [args.volume_size, args.volume_size, args.volume_size],
        "dVoxel": [volume_spacing, volume_spacing, volume_spacing],
        "first_projection_angle": [
            float(proj_a["positioner_primary_angle"]),
            float(proj_a["positioner_secondary_angle"]),
        ],
        "second_projection_angle": [
            float(proj_b["positioner_primary_angle"]),
            float(proj_b["positioner_secondary_angle"]),
        ],
    }

    train_config = {
        "exp": {
            "expname": f"{args.case_id}__phase_{args.phase_index:02d}__{bundle_tag}__{args.input_mode}_{args.detector_size}",
            "expdir": str((out_dir / "logs").resolve()),
            "dataconfig": str((out_dir / "data" / "config.yml").resolve()),
        },
        "network": {
            "net_type": "mlp",
            "num_layers": 8,
            "hidden_dim": 256,
            "skips": [4],
            "out_dim": 1,
            "last_activation": "sigmoid",
            "bound": args.bound,
        },
        "encoder": {
            "encoding": "hashgrid",
            "input_dim": 3,
            "num_levels": 16,
            "level_dim": 2,
            "base_resolution": 16,
            "log2_hashmap_size": 19,
        },
        "render": {
            "n_fine": 0,
            "netchunk": 699060,
        },
        "train": {
            "epoch": args.epochs,
            "lrate": 1e-4,
            "lrate_gamma": 0.1,
            "lrate_step": args.epochs,
            "resume": False,
        },
        "log": {
            "i_eval": args.eval_every,
            "i_save": args.save_every,
        },
    }

    with (out_dir / "data" / "config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_config, handle, sort_keys=False)
    with (config_dir / "CCTA.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(train_config, handle, sort_keys=False)

    export_manifest = {
        "case_id": args.case_id,
        "source_case_manifest": str((case_dir / "case_manifest.json").resolve()),
        "phase_index": args.phase_index,
        "selected_phase_indices": selected_phase_indices,
        "bundle_size": len(selected_phase_indices),
        "input_mode": args.input_mode,
        "input_mode_params": {
            "blur_sigma": args.blur_sigma,
            "edt_falloff_px": args.edt_falloff_px,
        },
        "detector_size": args.detector_size,
        "volume_size": args.volume_size,
        "volume_extent_mm": args.volume_extent_mm,
        "data_path": str(data_path.resolve()),
        "data_shape": list(projections.shape),
        "phase_pairs": phase_payloads,
        "geometry": manifest["geometry"],
        "train_config": str((config_dir / "CCTA.yaml").resolve()),
        "hard_masks_path": str((data_dir / "hard_masks.npy").resolve()),
        "raw_images_path": str((data_dir / "raw_images.npy").resolve()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(export_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(out_dir.resolve()),
        "data_path": str(data_path.resolve()),
        "train_config": str((config_dir / "CCTA.yaml").resolve()),
        "phase_index": args.phase_index,
        "selected_phase_indices": selected_phase_indices,
        "bundle_size": len(selected_phase_indices),
        "input_mode": args.input_mode,
        "detector_size": args.detector_size,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
