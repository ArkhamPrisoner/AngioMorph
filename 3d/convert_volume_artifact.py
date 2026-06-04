#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert volumetric artifact to PLY point cloud and OBJ voxel surface mesh.")
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest.json with volume_extent_mm / volume_size")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--percentiles", default="99.5,99.9,99.95")
    return parser.parse_args()


def load_volume(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix == ".npz":
        data = np.load(path)
        if not data.files:
            raise ValueError(f"No arrays inside {path}")
        return data[data.files[0]].astype(np.float32)
    raise ValueError(f"Unsupported volume format: {path}")


def infer_spacing(volume: np.ndarray, manifest_path: Path | None) -> tuple[float, float, float]:
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        extent = float(manifest.get("volume_extent_mm", 0.0))
        if extent > 0:
            sx = extent / float(volume.shape[0])
            sy = extent / float(volume.shape[1])
            sz = extent / float(volume.shape[2])
            return sx, sy, sz
    return 1.0, 1.0, 1.0


def voxel_centers(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    shape = np.array(mask.shape, dtype=np.float32)
    spacing_arr = np.array(spacing, dtype=np.float32)
    centers = (idx.astype(np.float32) + 0.5 - shape / 2.0) * spacing_arr
    return centers[:, [2, 1, 0]]


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def voxel_box_vertices(center_xyz: tuple[float, float, float], spacing: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    cx, cy, cz = center_xyz
    sx, sy, sz = spacing[2] / 2.0, spacing[1] / 2.0, spacing[0] / 2.0
    return [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ]


def write_voxel_surface_obj(path: Path, mask: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    occupied = np.argwhere(mask)
    shape = np.array(mask.shape, dtype=np.float32)
    face_defs = [
        ((1, 0, 0), (5, 6, 7, 8)),
        ((-1, 0, 0), (1, 2, 3, 4)),
        ((0, 1, 0), (4, 3, 7, 8)),
        ((0, -1, 0), (1, 2, 6, 5)),
        ((0, 0, 1), (2, 3, 7, 6)),
        ((0, 0, -1), (1, 4, 8, 5)),
    ]
    vertex_count = 0
    face_count = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# voxel surface OBJ\n")
        for i, j, k in occupied:
            center_ijk = (np.array([i, j, k], dtype=np.float32) + 0.5 - shape / 2.0) * np.array(spacing, dtype=np.float32)
            center_xyz = (float(center_ijk[2]), float(center_ijk[1]), float(center_ijk[0]))
            verts = voxel_box_vertices(center_xyz, spacing)
            for offset, face in face_defs:
                ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]
                inside = 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and 0 <= nk < mask.shape[2]
                if inside and mask[ni, nj, nk]:
                    continue
                start = vertex_count + 1
                for vx, vy, vz in verts:
                    handle.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
                a, b, c, d = face
                handle.write(f"f {start + a - 1} {start + b - 1} {start + c - 1}\n")
                handle.write(f"f {start + a - 1} {start + c - 1} {start + d - 1}\n")
                vertex_count += 8
                face_count += 2
    return {"vertices_written": vertex_count, "faces_written": face_count}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    volume = load_volume(args.volume)
    percentiles = [float(item.strip()) for item in args.percentiles.split(",") if item.strip()]
    spacing = infer_spacing(volume, args.manifest)

    summary = {
        "volume_path": str(args.volume.resolve()),
        "manifest_path": str(args.manifest.resolve()) if args.manifest else None,
        "shape": list(volume.shape),
        "spacing_mm": list(spacing),
        "stats": {
            "min": float(volume.min()),
            "max": float(volume.max()),
            "mean": float(volume.mean()),
        },
        "artifacts": [],
    }

    for percentile in percentiles:
        threshold = float(np.percentile(volume, percentile))
        mask = volume >= threshold
        num_components = int(ndimage.label(mask)[1])
        points = voxel_centers(mask, spacing)
        tag = f"p{str(percentile).replace('.', '_')}"
        ply_path = args.output_dir / f"{args.volume.stem}_{tag}.ply"
        obj_path = args.output_dir / f"{args.volume.stem}_{tag}.obj"
        write_ascii_ply(ply_path, points)
        mesh_info = write_voxel_surface_obj(obj_path, mask, spacing)
        summary["artifacts"].append({
            "percentile": percentile,
            "threshold": threshold,
            "occupied_voxels": int(mask.sum()),
            "component_count": num_components,
            "ply": str(ply_path.resolve()),
            "obj": str(obj_path.resolve()),
            **mesh_info,
        })

    summary_path = args.output_dir / f"{args.volume.stem}_conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
