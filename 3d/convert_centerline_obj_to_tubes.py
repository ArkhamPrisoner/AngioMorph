#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert OBJ polylines (v/l) into tube mesh OBJ.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-obj', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--radius', type=float, default=0.8)
    parser.add_argument('--sides', type=int, default=10)
    parser.add_argument('--min-segment-length', type=float, default=0.5)
    parser.add_argument('--min-point-norm', type=float, default=1.0)
    return parser.parse_args()


def parse_obj_lines(path: Path):
    vertices = []
    polylines = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        if raw.startswith('v '):
            _, x, y, z = raw.split()[:4]
            vertices.append((float(x), float(y), float(z)))
        elif raw.startswith('l '):
            ids = [int(part) - 1 for part in raw.split()[1:]]
            polylines.append(ids)
    return np.asarray(vertices, dtype=np.float32), polylines


def filter_polyline(points: np.ndarray, min_point_norm: float, min_segment_length: float) -> np.ndarray:
    kept = []
    for point in points:
        if float(np.linalg.norm(point)) < min_point_norm:
            continue
        if kept and float(np.linalg.norm(point - kept[-1])) < min_segment_length:
            continue
        kept.append(point)
    if len(kept) < 2:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(kept, dtype=np.float32)


def orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(direction, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(direction, u)
    v = v / np.linalg.norm(v)
    return u.astype(np.float32), v.astype(np.float32)


def ring_points(center: np.ndarray, direction: np.ndarray, radius: float, sides: int) -> np.ndarray:
    u, v = orthonormal_basis(direction)
    pts = []
    for i in range(sides):
        angle = 2.0 * math.pi * i / sides
        offset = math.cos(angle) * u * radius + math.sin(angle) * v * radius
        pts.append(center + offset)
    return np.asarray(pts, dtype=np.float32)


def tube_mesh(polyline: np.ndarray, radius: float, sides: int):
    verts = []
    faces = []
    rings = []
    for idx, center in enumerate(polyline):
        if idx == 0:
            direction = polyline[1] - polyline[0]
        elif idx == len(polyline) - 1:
            direction = polyline[-1] - polyline[-2]
        else:
            direction = polyline[idx + 1] - polyline[idx - 1]
        if float(np.linalg.norm(direction)) < 1e-6:
            continue
        ring = ring_points(center, direction, radius, sides)
        rings.append(ring)
    if len(rings) < 2:
        return np.zeros((0, 3), dtype=np.float32), []
    for ring in rings:
        start = len(verts) + 1
        verts.extend(ring.tolist())
        if start > 1:
            prev = start - sides
            curr = start
            for i in range(sides):
                a = prev + i
                b = prev + ((i + 1) % sides)
                c = curr + ((i + 1) % sides)
                d = curr + i
                faces.append((a, b, c))
                faces.append((a, c, d))
    return np.asarray(verts, dtype=np.float32), faces


def write_obj(path: Path, vertices: np.ndarray, faces: list[tuple[int, int, int]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        handle.write('# tube mesh from centerline obj\n')
        for x, y, z in vertices:
            handle.write(f'v {x:.6f} {y:.6f} {z:.6f}\n')
        for a, b, c in faces:
            handle.write(f'f {a} {b} {c}\n')


def main() -> None:
    args = parse_args()
    vertices, polylines = parse_obj_lines(args.input)
    all_vertices = []
    all_faces = []
    offset = 0
    kept_polylines = []
    for ids in polylines:
        poly = filter_polyline(vertices[ids], args.min_point_norm, args.min_segment_length)
        if len(poly) < 2:
            continue
        mesh_vertices, mesh_faces = tube_mesh(poly, args.radius, args.sides)
        if len(mesh_vertices) == 0:
            continue
        all_vertices.extend(mesh_vertices.tolist())
        all_faces.extend([(a + offset, b + offset, c + offset) for a, b, c in mesh_faces])
        offset += len(mesh_vertices)
        kept_polylines.append({
            'num_points': int(len(poly)),
            'start': poly[0].tolist(),
            'end': poly[-1].tolist(),
            'length': float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1))),
        })
    write_obj(args.output_obj, np.asarray(all_vertices, dtype=np.float32), all_faces)
    summary = {
        'input': str(args.input.resolve()),
        'output_obj': str(args.output_obj.resolve()),
        'radius': args.radius,
        'sides': args.sides,
        'min_segment_length': args.min_segment_length,
        'min_point_norm': args.min_point_norm,
        'num_input_vertices': int(len(vertices)),
        'num_input_polylines': int(len(polylines)),
        'num_kept_polylines': int(len(kept_polylines)),
        'num_output_vertices': int(len(all_vertices)),
        'num_output_faces': int(len(all_faces)),
        'kept_polylines': kept_polylines,
    }
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
