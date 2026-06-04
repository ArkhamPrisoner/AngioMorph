#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


@dataclass
class EdgeModel:
    source: int
    target: int
    points2d: np.ndarray
    points3d: np.ndarray
    radii: np.ndarray


@dataclass
class TreeModel:
    name: str
    side: str
    frame: str
    root_node: int
    root_xyz: np.ndarray
    mask_path: Path
    debug_mask_path: Path
    preview_path: Path
    obj_path: Path
    centerline_obj_path: Path
    json_path: Path
    edges: list[EdgeModel]


def load_binary_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr > 0


def save_binary(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def choose_best_mask(mask_dir: Path) -> Path:
    candidates = sorted(mask_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"no masks in {mask_dir}")
    scored = []
    for path in candidates:
        mask = load_binary_mask(path)
        scored.append((int(mask.sum()), path))
    return max(scored)[1]


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.array(mask.shape, dtype=float) / 2.0
    return coords.mean(axis=0)


def shift_binary_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask)
    row_start = max(0, dr)
    row_end = min(mask.shape[0], mask.shape[0] + dr)
    col_start = max(0, dc)
    col_end = min(mask.shape[1], mask.shape[1] + dc)
    src_row_start = max(0, -dr)
    src_col_start = max(0, -dc)
    src_row_end = src_row_start + (row_end - row_start)
    src_col_end = src_col_start + (col_end - col_start)
    if row_end > row_start and col_end > col_start:
        out[row_start:row_end, col_start:col_end] = mask[src_row_start:src_row_end, src_col_start:src_col_end]
    return out


def register_mask_translation(
    mask: np.ndarray,
    reference: np.ndarray,
    search_radius: int = 20,
) -> tuple[float, int, int, np.ndarray]:
    guess = np.round(mask_centroid(reference) - mask_centroid(mask)).astype(int)
    best_score = -1.0
    best_shift = (0, 0)
    best_mask = mask
    for dr in range(int(guess[0]) - search_radius, int(guess[0]) + search_radius + 1):
        for dc in range(int(guess[1]) - search_radius, int(guess[1]) + search_radius + 1):
            shifted = shift_binary_mask(mask, dr, dc)
            inter = np.logical_and(shifted, reference).sum()
            union = np.logical_or(shifted, reference).sum()
            score = inter / union if union else 0.0
            if score > best_score:
                best_score = float(score)
                best_shift = (int(dr), int(dc))
                best_mask = shifted
    return best_score, best_shift[0], best_shift[1], best_mask


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2), dtype=bool), iterations=1)
    labels, n = ndi.label(mask)
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    mask = labels == counts.argmax()
    return mask.astype(bool)


def zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    img = mask.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1, mode="constant")
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]

            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            transitions = sum(
                ((neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1)).astype(np.uint8)
                for i in range(8)
            )
            neighbor_count = sum(neighbors)
            if step == 0:
                cond = (
                    (img == 1)
                    & (neighbor_count >= 2)
                    & (neighbor_count <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p6) == 0)
                    & ((p4 * p6 * p8) == 0)
                )
            else:
                cond = (
                    (img == 1)
                    & (neighbor_count >= 2)
                    & (neighbor_count <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p8) == 0)
                    & ((p2 * p6 * p8) == 0)
                )
            if np.any(cond):
                img[cond] = 0
                changed = True
    return img.astype(bool)


def neighbor_map(skeleton: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    coords = {tuple(x) for x in np.argwhere(skeleton)}
    mapping: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        neighbors = []
        for dr, dc in NEIGHBOR_OFFSETS:
            q = (r + dr, c + dc)
            if q in coords:
                neighbors.append(q)
        mapping[(r, c)] = neighbors
    return mapping


def prune_short_branches(skeleton: np.ndarray, min_length: int = 12, rounds: int = 20) -> np.ndarray:
    skel = skeleton.copy()
    for _ in range(rounds):
        mapping = neighbor_map(skel)
        degrees = {p: len(nbs) for p, nbs in mapping.items()}
        endpoints = [p for p, d in degrees.items() if d == 1]
        to_remove: set[tuple[int, int]] = set()
        for endpoint in endpoints:
            if endpoint in to_remove:
                continue
            path = [endpoint]
            prev = None
            cur = endpoint
            while True:
                neighbors = [x for x in mapping[cur] if x != prev]
                if not neighbors:
                    break
                nxt = neighbors[0]
                path.append(nxt)
                prev, cur = cur, nxt
                if degrees[cur] != 2:
                    break
                if len(path) > min_length:
                    break
            if len(path) <= min_length and degrees.get(cur, 0) >= 3:
                to_remove.update(path[:-1])
        if not to_remove:
            break
        for r, c in to_remove:
            skel[r, c] = False
    return skel


def build_graph_from_skeleton(skeleton: np.ndarray) -> tuple[nx.Graph, dict[int, tuple[int, int]]]:
    mapping = neighbor_map(skeleton)
    degrees = {p: len(nbs) for p, nbs in mapping.items()}
    keypoints = [p for p, degree in degrees.items() if degree != 2]
    if not keypoints:
        keypoints = [min(mapping)]

    node_pixels = {idx: pixel for idx, pixel in enumerate(sorted(keypoints))}
    pixel_to_node = {pixel: idx for idx, pixel in node_pixels.items()}
    graph = nx.Graph()
    for node_id, pixel in node_pixels.items():
        graph.add_node(node_id, pixel=pixel)

    visited_segments: set[frozenset[tuple[int, int]]] = set()
    for pixel in keypoints:
        source = pixel_to_node[pixel]
        for neighbor in mapping[pixel]:
            segment_key = frozenset((pixel, neighbor))
            if segment_key in visited_segments:
                continue
            path = [pixel, neighbor]
            visited_segments.add(segment_key)
            prev = pixel
            cur = neighbor
            while cur not in pixel_to_node:
                options = [x for x in mapping[cur] if x != prev]
                if not options:
                    break
                nxt = options[0]
                visited_segments.add(frozenset((cur, nxt)))
                path.append(nxt)
                prev, cur = cur, nxt
            if cur not in pixel_to_node or cur == pixel:
                continue
            target = pixel_to_node[cur]
            polyline = np.array([(p[1], -p[0]) for p in path], dtype=float)
            length = polyline_length(polyline)
            if length <= 0:
                continue
            graph.add_edge(source, target, path=path, length=length)
    if not nx.is_tree(graph):
        graph = nx.minimum_spanning_tree(graph, weight="length")
    return graph, node_pixels


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def endpoint_root(graph: nx.Graph, node_pixels: dict[int, tuple[int, int]]) -> int:
    leaves = [node for node in graph.nodes if graph.degree[node] <= 1]
    candidates = leaves or list(graph.nodes)
    def score(node: int) -> tuple[float, float]:
        r, c = node_pixels[node]
        return (r, c)
    return min(candidates, key=score)


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.zeros(1, dtype=float)
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=float)
    return vec / norm


def tangent(points: np.ndarray, forward: bool = True) -> np.ndarray:
    if len(points) < 2:
        return np.array([1.0, 0.0], dtype=float)
    if forward:
        delta = points[min(4, len(points) - 1)] - points[0]
    else:
        delta = points[-1] - points[max(0, len(points) - 5)]
    return unit(delta)


def depth_warp(points: np.ndarray, root_xy: np.ndarray, bbox: float, side: str) -> np.ndarray:
    scale = 0.18 * bbox
    x_norm = (points[:, 0] - root_xy[0]) / bbox
    y_norm = (points[:, 1] - root_xy[1]) / bbox
    if side == "left":
        return scale * (0.55 * x_norm - 0.18 * y_norm)
    return scale * (-0.40 * x_norm - 0.14 * y_norm)


def reconstruct_tree(
    graph: nx.Graph,
    node_pixels: dict[int, tuple[int, int]],
    skeleton: np.ndarray,
    mask: np.ndarray,
    side: str,
) -> tuple[int, np.ndarray, list[EdgeModel]]:
    root = endpoint_root(graph, node_pixels)
    root_xy = np.array([node_pixels[root][1], -node_pixels[root][0]], dtype=float)
    coords = np.argwhere(mask)
    y_span = float(coords[:, 0].max() - coords[:, 0].min() + 1)
    x_span = float(coords[:, 1].max() - coords[:, 1].min() + 1)
    bbox = max(x_span, y_span, 1.0)
    dist = ndi.distance_transform_edt(mask)
    max_length = max((data["length"] for _, _, data in graph.edges(data=True)), default=1.0)

    edges: list[EdgeModel] = []

    def walk(node: int, parent: int | None, incoming_dir: np.ndarray | None, z0: float) -> None:
        for nbr in graph.neighbors(node):
            if nbr == parent:
                continue
            path = graph[node][nbr]["path"]
            if path[0] != node_pixels[node]:
                path = list(reversed(path))
            points2d = np.array([(c, -r) for r, c in path], dtype=float)
            t_start = tangent(points2d, forward=True)
            edge_length = max(polyline_length(points2d), 1.0)
            if incoming_dir is None:
                branch_bias = float(np.clip(t_start[0], -0.7, 0.7))
            else:
                cross = incoming_dir[0] * t_start[1] - incoming_dir[1] * t_start[0]
                dot = float(np.clip(np.dot(incoming_dir, t_start), -1.0, 1.0))
                angle = math.atan2(cross, dot)
                branch_bias = math.sin(angle)

            s = cumulative_lengths(points2d)
            s_norm = s / max(s[-1], 1.0)
            local_scale = 0.22 * bbox * (0.35 + 0.65 * min(edge_length / max_length, 1.0))
            warp = depth_warp(points2d, root_xy, bbox, side)
            z = z0 + (warp - warp[0]) + branch_bias * local_scale * np.power(s_norm, 1.15)
            radii = np.array([max(float(dist[r, c]), 1.0) for r, c in path], dtype=float)
            points3d = np.column_stack([points2d[:, 0], points2d[:, 1], z])
            points3d, radii = resample_polyline(points3d, radii, step=2.5)
            points3d = smooth_polyline(points3d, window=9)
            radii = smooth_radii(radii, window=9)
            edges.append(
                EdgeModel(
                    source=node,
                    target=nbr,
                    points2d=points3d[:, :2].copy(),
                    points3d=points3d,
                    radii=radii,
                )
            )
            walk(nbr, node, tangent(points2d, forward=False), float(z[-1]))

    walk(root, None, None, 0.0)
    return root, np.array([root_xy[0], root_xy[1], 0.0], dtype=float), edges


def resample_polyline(points: np.ndarray, radii: np.ndarray, step: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        return points, radii
    s = cumulative_lengths(points)
    total = s[-1]
    count = max(2, int(total / step) + 1)
    targets = np.linspace(0.0, total, count)
    out_points = np.column_stack(
        [np.interp(targets, s, points[:, i]) for i in range(points.shape[1])]
    )
    out_radii = np.interp(targets, s, radii)
    return out_points, out_radii


def smooth_polyline(points: np.ndarray, window: int = 7) -> np.ndarray:
    if len(points) < 5 or window < 3:
        return points
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    smoothed = points.copy()
    for axis in range(points.shape[1]):
        values = points[:, axis]
        padded = np.pad(values, (window // 2, window // 2), mode="edge")
        smoothed[:, axis] = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def smooth_radii(radii: np.ndarray, window: int = 7) -> np.ndarray:
    if len(radii) < 5 or window < 3:
        return radii
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(radii, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = radii[0]
    smoothed[-1] = radii[-1]
    return smoothed


def tube_mesh(points: np.ndarray, radii: np.ndarray, sides: int = 10) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    if len(points) < 2:
        return [], []
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    rings: list[list[int]] = []
    for i in range(len(points)):
        if i == 0:
            tangent_vec = points[1] - points[0]
        elif i == len(points) - 1:
            tangent_vec = points[-1] - points[-2]
        else:
            tangent_vec = points[i + 1] - points[i - 1]
        tangent_vec = tangent_vec / max(np.linalg.norm(tangent_vec), 1e-8)
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(tangent_vec, ref)) > 0.92:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = np.cross(tangent_vec, ref)
        normal = normal / max(np.linalg.norm(normal), 1e-8)
        binormal = np.cross(tangent_vec, normal)
        binormal = binormal / max(np.linalg.norm(binormal), 1e-8)
        ring: list[int] = []
        radius = max(float(radii[i]), 0.8)
        for j in range(sides):
            angle = (2.0 * math.pi * j) / sides
            offset = math.cos(angle) * normal * radius + math.sin(angle) * binormal * radius
            vertices.append(points[i] + offset)
            ring.append(len(vertices))
        rings.append(ring)

    for a, b in zip(rings[:-1], rings[1:]):
        for j in range(sides):
            a0 = a[j]
            a1 = a[(j + 1) % sides]
            b0 = b[j]
            b1 = b[(j + 1) % sides]
            faces.append((a0, b0, b1))
            faces.append((a0, b1, a1))
    return vertices, faces


def write_obj_mesh(path: Path, edges: list[EdgeModel]) -> None:
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    for edge in edges:
        points, radii = resample_polyline(edge.points3d, edge.radii, step=3.0)
        edge_vertices, edge_faces = tube_mesh(points, radii, sides=10)
        offset = len(vertices)
        vertices.extend(edge_vertices)
        faces.extend([(a + offset, b + offset, c + offset) for a, b, c in edge_faces])
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def write_centerlines_obj(path: Path, edges: list[EdgeModel]) -> None:
    vertex_offset = 1
    with path.open("w", encoding="utf-8") as f:
        for edge in edges:
            for point in edge.points3d:
                f.write(f"v {point[0]:.5f} {point[1]:.5f} {point[2]:.5f}\n")
            indices = list(range(vertex_offset, vertex_offset + len(edge.points3d)))
            f.write("l " + " ".join(map(str, indices)) + "\n")
            vertex_offset += len(edge.points3d)


def save_debug_overlay(mask: np.ndarray, skeleton: np.ndarray, path: Path) -> None:
    base = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    base[mask] = (210, 210, 210)
    base[skeleton] = (220, 30, 30)
    Image.fromarray(base, mode="RGB").save(path)


def render_preview(edges: list[EdgeModel], path: Path, title: str) -> None:
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    for edge in edges:
        pts = edge.points3d
        linewidth = float(np.clip(np.percentile(edge.radii, 85) * 0.28, 0.8, 4.2))
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#b81414", linewidth=linewidth)
    ax.set_title(title)
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-61)
    all_points = np.concatenate([edge.points3d for edge in edges], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(maxs - mins) * 0.55
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, tree: TreeModel) -> None:
    payload = {
        "name": tree.name,
        "side": tree.side,
        "frame": tree.frame,
        "root_node": tree.root_node,
        "root_xyz": tree.root_xyz.tolist(),
        "mask_path": str(tree.mask_path),
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "points3d": edge.points3d.round(4).tolist(),
                "radii": edge.radii.round(4).tolist(),
            }
            for edge in tree.edges
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_confidence_map(counts: np.ndarray, path: Path) -> None:
    if counts.max() <= 0:
        image = np.zeros((*counts.shape, 3), dtype=np.uint8)
    else:
        norm = counts.astype(float) / counts.max()
        image = np.zeros((*counts.shape, 3), dtype=np.uint8)
        image[..., 0] = np.clip(255.0 * norm, 0, 255).astype(np.uint8)
        image[..., 1] = np.clip(180.0 * np.sqrt(norm), 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def process_mask(name: str, side: str, mask: np.ndarray, mask_path: Path, frame_label: str, output_dir: Path) -> TreeModel:
    mask = preprocess_mask(mask)
    skeleton = zhang_suen_thinning(mask)
    skeleton = prune_short_branches(skeleton, min_length=10, rounds=24)
    graph, node_pixels = build_graph_from_skeleton(skeleton)
    root_node, root_xyz, edges = reconstruct_tree(graph, node_pixels, skeleton, mask, side)

    debug_mask_path = output_dir / f"{name}_mask_skeleton.png"
    preview_path = output_dir / f"{name}_preview.png"
    obj_path = output_dir / f"{name}.obj"
    centerline_obj_path = output_dir / f"{name}_centerlines.obj"
    json_path = output_dir / f"{name}.json"

    save_debug_overlay(mask, skeleton, debug_mask_path)
    render_preview(edges, preview_path, f"{name} pseudo-3D")
    write_obj_mesh(obj_path, edges)
    write_centerlines_obj(centerline_obj_path, edges)

    tree = TreeModel(
        name=name,
        side=side,
        frame=frame_label,
        root_node=root_node,
        root_xyz=root_xyz,
        mask_path=mask_path,
        debug_mask_path=debug_mask_path,
        preview_path=preview_path,
        obj_path=obj_path,
        centerline_obj_path=centerline_obj_path,
        json_path=json_path,
        edges=edges,
    )
    write_json(json_path, tree)
    return tree


def process_series(name: str, side: str, mask_dir: Path, output_dir: Path) -> TreeModel:
    mask_path = choose_best_mask(mask_dir)
    mask = load_binary_mask(mask_path)
    return process_mask(name, side, mask, mask_path, mask_path.stem, output_dir)


def process_fused_series(name: str, side: str, mask_dir: Path, output_dir: Path) -> tuple[TreeModel, Path, Path, Path]:
    mask_paths = sorted(mask_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"no masks in {mask_dir}")

    masks = {path: load_binary_mask(path) for path in mask_paths}
    areas = {path: int(mask.sum()) for path, mask in masks.items()}
    ref_path = max(mask_paths, key=lambda path: areas[path])
    reference = masks[ref_path]
    max_area = areas[ref_path]

    counts = np.zeros(reference.shape, dtype=np.uint16)
    alignment: list[dict[str, float | int | str | bool]] = []
    selected = [path for path in mask_paths if areas[path] >= 0.55 * max_area]
    for path in selected:
        score, dr, dc, shifted = register_mask_translation(masks[path], reference, search_radius=20)
        shift_norm = float(math.hypot(dr, dc))
        keep = path == ref_path or (score >= 0.22 and shift_norm <= 90.0)
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
    save_binary(fused_mask, fused_mask_path)
    confidence_map_path = output_dir / f"{name}_confidence.png"
    save_confidence_map(counts, confidence_map_path)
    alignment_path = output_dir / f"{name}_alignment.json"
    alignment_path.write_text(json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8")

    tree = process_mask(name, side, fused_mask, fused_mask_path, f"fused_from_{ref_path.stem}", output_dir)
    return tree, fused_mask_path, confidence_map_path, alignment_path


def transform_edges(
    edges: list[EdgeModel],
    translate: np.ndarray,
    root_xyz: np.ndarray,
    rotation_deg: float = 0.0,
) -> list[EdgeModel]:
    theta = math.radians(rotation_deg)
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    transformed: list[EdgeModel] = []
    for edge in edges:
        pts = (edge.points3d - root_xyz) @ rot.T + translate
        transformed.append(
            EdgeModel(
                source=edge.source,
                target=edge.target,
                points2d=edge.points2d.copy(),
                points3d=pts,
                radii=edge.radii.copy(),
            )
        )
    return transformed


def build_combined_scene(left_tree: TreeModel, right_tree: TreeModel, output_dir: Path) -> None:
    build_combined_scene_named("combined", left_tree, right_tree, output_dir)


def build_combined_scene_named(name: str, left_tree: TreeModel, right_tree: TreeModel, output_dir: Path) -> None:
    left_edges = transform_edges(
        left_tree.edges,
        translate=np.array([-90.0, 0.0, 8.0], dtype=float),
        root_xyz=left_tree.root_xyz,
        rotation_deg=-8.0,
    )
    right_edges = transform_edges(
        right_tree.edges,
        translate=np.array([90.0, 0.0, -8.0], dtype=float),
        root_xyz=right_tree.root_xyz,
        rotation_deg=10.0,
    )
    combined_edges = left_edges + right_edges
    render_preview(combined_edges, output_dir / f"{name}_preview.png", f"{name} pseudo-3D coronary tree")
    write_obj_mesh(output_dir / f"{name}.obj", combined_edges)
    write_centerlines_obj(output_dir / f"{name}_centerlines.obj", combined_edges)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    left_tree = process_series("left_tree", "left", base_dir / "extracted" / "left_mask", output_dir)
    right_tree = process_series("right_tree", "right", base_dir / "extracted" / "right_mask", output_dir)
    left_tree_fused, left_fused_mask, left_confidence, left_alignment = process_fused_series(
        "left_tree_fused", "left", base_dir / "extracted" / "left_mask", output_dir
    )
    right_tree_fused, right_fused_mask, right_confidence, right_alignment = process_fused_series(
        "right_tree_fused", "right", base_dir / "extracted" / "right_mask", output_dir
    )
    build_combined_scene(left_tree, right_tree, output_dir)
    build_combined_scene_named("combined_fused", left_tree_fused, right_tree_fused, output_dir)

    summary = {
        "left_tree": {
            "frame": left_tree.frame,
            "mask": str(left_tree.mask_path),
            "preview": str(left_tree.preview_path),
            "mesh": str(left_tree.obj_path),
        },
        "right_tree": {
            "frame": right_tree.frame,
            "mask": str(right_tree.mask_path),
            "preview": str(right_tree.preview_path),
            "mesh": str(right_tree.obj_path),
        },
        "left_tree_fused": {
            "frame": left_tree_fused.frame,
            "mask": str(left_tree_fused.mask_path),
            "preview": str(left_tree_fused.preview_path),
            "mesh": str(left_tree_fused.obj_path),
            "confidence_map": str(left_confidence),
            "alignment": str(left_alignment),
        },
        "right_tree_fused": {
            "frame": right_tree_fused.frame,
            "mask": str(right_tree_fused.mask_path),
            "preview": str(right_tree_fused.preview_path),
            "mesh": str(right_tree_fused.obj_path),
            "confidence_map": str(right_confidence),
            "alignment": str(right_alignment),
        },
        "combined": {
            "preview": str(output_dir / "combined_preview.png"),
            "mesh": str(output_dir / "combined.obj"),
        },
        "combined_fused": {
            "preview": str(output_dir / "combined_fused_preview.png"),
            "mesh": str(output_dir / "combined_fused.obj"),
        },
        "note": (
            "This is a pseudo-3D reconstruction from single-view masks. "
            "True anatomical 3D requires at least two calibrated views for the same vessel tree."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
