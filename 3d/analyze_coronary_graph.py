#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from reconstruct_coronary_tree import (
    NEIGHBOR_OFFSETS,
    load_binary_mask,
    polyline_length,
    preprocess_mask,
    prune_short_branches,
    zhang_suen_thinning,
)


CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)
ANNOTATOR_NAMES = {
    "1": "Demkin",
    "2": "Bilan",
    "3": "Veremeenko",
    "4": "Vodolazko",
    "5": "Kabakov",
    "6": "Solovyova",
    "7": "Shtepa",
    "8": "Kazhanenko",
}


@dataclass
class BranchDescriptor:
    edge_id: int
    neighbor_node: int
    direction_xy: np.ndarray
    angle_deg: float
    length_px: float
    radius_px: float
    signal_mean: float
    signal_std: float
    contrast_mean: float
    gradient_mean: float


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def parse_case_metadata(path: Path) -> dict[str, object]:
    parts = path.stem.split("_")
    meta: dict[str, object] = {
        "basename": path.stem,
        "filename": path.name,
        "path": str(path),
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else None,
    }
    if len(parts) >= 4 and parts[0].startswith("p") and parts[2].startswith("f"):
        meta.update(
            {
                "patient_id": parts[0],
                "series_id": parts[1],
                "frame_id": parts[2],
                "annotator_id": parts[3] if len(parts) >= 4 else None,
                "annotator_name": ANNOTATOR_NAMES.get(parts[3], "unknown") if len(parts) >= 4 else None,
            }
        )
    return meta


def image_stats(gray: np.ndarray, mask: np.ndarray, signal: np.ndarray, gradient: np.ndarray) -> dict[str, object]:
    vessel_values = signal[mask]
    bg_mask = ndi.binary_dilation(mask, iterations=8) & ~ndi.binary_dilation(mask, iterations=2)
    bg_values = signal[bg_mask] if bg_mask.any() else signal[~mask]
    return {
        "shape_hw": [int(gray.shape[0]), int(gray.shape[1])],
        "gray_min": float(gray.min()),
        "gray_max": float(gray.max()),
        "vessel_signal_mean": float(np.mean(vessel_values)) if vessel_values.size else 0.0,
        "vessel_signal_std": float(np.std(vessel_values)) if vessel_values.size else 0.0,
        "background_signal_mean": float(np.mean(bg_values)) if bg_values.size else 0.0,
        "mask_gradient_mean": float(np.mean(gradient[mask])) if mask.any() else 0.0,
        "mask_gradient_p95": float(np.percentile(gradient[mask], 95)) if mask.any() else 0.0,
    }


def normalize_image(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, [1.0, 99.0])
    if hi <= lo:
        lo = float(gray.min())
        hi = float(gray.max())
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def vessel_signal_image(image01: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
    ring = ndi.binary_dilation(mask, iterations=8) & ~ndi.binary_dilation(mask, iterations=2)
    if not ring.any():
        ring = ~mask
    inside = float(np.median(image01[mask])) if mask.any() else 0.5
    outside = float(np.median(image01[ring])) if ring.any() else 0.5
    if inside < outside:
        return 1.0 - image01, "dark_vessels"
    return image01.copy(), "bright_vessels"


def skeleton_neighbor_map(skeleton: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    coords = {tuple(x) for x in np.argwhere(skeleton)}
    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        nbs = []
        for dr, dc in NEIGHBOR_OFFSETS:
            q = (r + dr, c + dc)
            if q in coords:
                nbs.append(q)
        out[(r, c)] = nbs
    return out


def build_contracted_skeleton_graph(skeleton: np.ndarray) -> tuple[nx.Graph, dict[int, dict[str, object]]]:
    mapping = skeleton_neighbor_map(skeleton)
    if not mapping:
        return nx.Graph(), {}

    degree = {pixel: len(neighbors) for pixel, neighbors in mapping.items()}
    key_mask = np.zeros_like(skeleton, dtype=bool)
    for pixel, deg in degree.items():
        if deg != 2:
            key_mask[pixel] = True

    if not key_mask.any():
        pixel = min(mapping)
        graph = nx.Graph()
        graph.add_node(0, pixel=pixel, pixels=[pixel], kind="cycle_anchor")
        return graph, {0: {"pixel": pixel, "pixels": [pixel], "kind": "cycle_anchor"}}

    labels, label_count = ndi.label(key_mask, structure=CONNECTIVITY_8)
    pixel_to_node: dict[tuple[int, int], int] = {}
    node_data: dict[int, dict[str, object]] = {}
    graph = nx.Graph()

    for label in range(1, label_count + 1):
        pixels = [tuple(x) for x in np.argwhere(labels == label)]
        node_id = label - 1
        centroid = np.mean(np.array(pixels, dtype=float), axis=0)
        pixel_arr = min(
            (np.array(pixel, dtype=float) for pixel in pixels),
            key=lambda candidate: float(np.linalg.norm(candidate - centroid)),
        )
        pixel = tuple(pixel_arr.astype(int))
        local_degrees = [degree[p] for p in pixels]
        kind = "junction" if max(local_degrees) >= 3 else "endpoint"
        graph.add_node(node_id, pixel=pixel, pixels=pixels, kind=kind)
        node_data[node_id] = {"pixel": pixel, "pixels": pixels, "kind": kind}
        for p in pixels:
            pixel_to_node[p] = node_id

    visited_starts: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edge_id = 0
    for source_pixel, source_node in list(pixel_to_node.items()):
        for neighbor in mapping[source_pixel]:
            if pixel_to_node.get(neighbor) == source_node:
                continue
            start_key = (source_pixel, neighbor)
            if start_key in visited_starts:
                continue

            path = [source_pixel, neighbor]
            prev = source_pixel
            cur = neighbor
            visited_starts.add((source_pixel, neighbor))
            target_node = pixel_to_node.get(cur)

            while target_node is None:
                options = [x for x in mapping[cur] if x != prev]
                if not options:
                    break
                nxt = options[0]
                visited_starts.add((cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
                target_node = pixel_to_node.get(cur)

            if target_node is None or target_node == source_node:
                continue

            reverse_path = list(reversed(path))
            reverse_key = (reverse_path[0], reverse_path[1]) if len(reverse_path) > 1 else None
            if reverse_key is not None:
                visited_starts.add(reverse_key)

            points_xy = np.array([(c, r) for r, c in path], dtype=float)
            length = polyline_length(points_xy)
            if length <= 0:
                continue

            if graph.has_edge(source_node, target_node):
                old_length = graph[source_node][target_node]["length"]
                if length <= old_length:
                    continue
            graph.add_edge(source_node, target_node, id=edge_id, path=path, length=float(length))
            edge_id += 1

    return graph, node_data


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def sample_bilinear(arr: np.ndarray, y: float, x: float) -> float:
    if y < 0 or x < 0 or y > arr.shape[0] - 1 or x > arr.shape[1] - 1:
        return float("nan")
    y0 = int(math.floor(y))
    x0 = int(math.floor(x))
    y1 = min(y0 + 1, arr.shape[0] - 1)
    x1 = min(x0 + 1, arr.shape[1] - 1)
    wy = y - y0
    wx = x - x0
    return float(
        arr[y0, x0] * (1 - wy) * (1 - wx)
        + arr[y1, x0] * wy * (1 - wx)
        + arr[y0, x1] * (1 - wy) * wx
        + arr[y1, x1] * wy * wx
    )


def oriented_edge_path(graph: nx.Graph, node: int, neighbor: int) -> list[tuple[int, int]]:
    path = graph[node][neighbor]["path"]
    node_pixels = set(graph.nodes[node]["pixels"])
    if path[0] in node_pixels:
        return path
    return list(reversed(path))


def branch_direction(path: list[tuple[int, int]], node_pixel: tuple[int, int], step_px: float = 14.0) -> np.ndarray:
    start = np.array([node_pixel[1], node_pixel[0]], dtype=float)
    points = np.array([(c, r) for r, c in path], dtype=float)
    if len(points) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    dist = np.linalg.norm(points - start[None, :], axis=1)
    idx = int(np.searchsorted(dist, step_px, side="left"))
    idx = min(max(idx, 1), len(points) - 1)
    return unit(points[idx] - start)


def describe_branch(
    graph: nx.Graph,
    node: int,
    neighbor: int,
    node_pixel: tuple[int, int],
    signal: np.ndarray,
    gradient: np.ndarray,
    dist: np.ndarray,
) -> BranchDescriptor:
    path = oriented_edge_path(graph, node, neighbor)
    direction = branch_direction(path, node_pixel)
    angle_deg = math.degrees(math.atan2(direction[1], direction[0]))

    sample_count = min(max(8, int(2.5 * max(float(dist[node_pixel]), 2.0))), len(path))
    sample_path = path[:sample_count]

    center_values = []
    gradient_values = []
    contrast_values = []
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    for r, c in sample_path:
        radius = max(float(dist[r, c]), 1.0)
        center = float(signal[r, c])
        sides = [
            sample_bilinear(signal, r + normal[1] * radius * 1.7, c + normal[0] * radius * 1.7),
            sample_bilinear(signal, r - normal[1] * radius * 1.7, c - normal[0] * radius * 1.7),
        ]
        sides = [x for x in sides if not math.isnan(x)]
        background = float(np.mean(sides)) if sides else center
        center_values.append(center)
        gradient_values.append(float(gradient[r, c]))
        contrast_values.append(center - background)

    radii = [float(dist[r, c]) for r, c in sample_path]
    return BranchDescriptor(
        edge_id=int(graph[node][neighbor]["id"]),
        neighbor_node=int(neighbor),
        direction_xy=direction,
        angle_deg=float(angle_deg),
        length_px=float(graph[node][neighbor]["length"]),
        radius_px=float(np.median(radii)) if radii else 0.0,
        signal_mean=float(np.mean(center_values)) if center_values else 0.0,
        signal_std=float(np.std(center_values)) if center_values else 0.0,
        contrast_mean=float(np.mean(contrast_values)) if contrast_values else 0.0,
        gradient_mean=float(np.mean(gradient_values)) if gradient_values else 0.0,
    )


def similarity_ratio(a: float, b: float, scale: float) -> float:
    if a <= 1e-6 or b <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - abs(math.log(a / b)) / scale, 0.0, 1.0))


def pair_angle_deg(a: BranchDescriptor, b: BranchDescriptor) -> float:
    dot = float(np.clip(np.dot(a.direction_xy, b.direction_xy), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def continuity_score(a: BranchDescriptor, b: BranchDescriptor) -> float:
    angle = pair_angle_deg(a, b)
    opposite = float(np.clip((angle - 90.0) / 90.0, 0.0, 1.0))
    radius_sim = similarity_ratio(max(a.radius_px, 0.1), max(b.radius_px, 0.1), math.log(3.0))
    signal_sim = 1.0 - float(np.clip(abs(a.signal_mean - b.signal_mean) / 0.35, 0.0, 1.0))
    contrast_sim = 1.0 - float(np.clip(abs(a.contrast_mean - b.contrast_mean) / 0.35, 0.0, 1.0))
    return float(0.50 * opposite + 0.25 * radius_sim + 0.15 * signal_sim + 0.10 * contrast_sim)


def best_crossing_pairing(branches: list[BranchDescriptor]) -> tuple[float, list[tuple[int, int]]]:
    if len(branches) != 4:
        return 0.0, []
    pairings = [
        [(0, 1), (2, 3)],
        [(0, 2), (1, 3)],
        [(0, 3), (1, 2)],
    ]
    scored = []
    for pairs in pairings:
        score = float(np.mean([continuity_score(branches[i], branches[j]) for i, j in pairs]))
        scored.append((score, pairs))
    return max(scored, key=lambda item: item[0])


def murray_score(branches: list[BranchDescriptor], gamma: float = 2.7) -> tuple[float, int]:
    if len(branches) < 3:
        return 0.0, -1
    radii = np.array([max(branch.radius_px, 0.1) for branch in branches], dtype=float)
    parent_idx = int(np.argmax(radii))
    parent = radii[parent_idx] ** gamma
    children = float(np.sum(np.delete(radii, parent_idx) ** gamma))
    if children <= 1e-6:
        return 0.0, parent_idx
    error = abs(math.log(parent / children))
    return float(np.exp(-error)), parent_idx


def classify_node(branches: list[BranchDescriptor], local_radius: float) -> dict[str, object]:
    degree = len(branches)
    if degree < 3:
        return {"type": "ordinary", "confidence": 1.0}

    pairwise = []
    for i in range(degree):
        for j in range(i + 1, degree):
            pairwise.append(
                {
                    "branches": [branches[i].edge_id, branches[j].edge_id],
                    "angle_deg": round(pair_angle_deg(branches[i], branches[j]), 3),
                    "continuity": round(continuity_score(branches[i], branches[j]), 4),
                }
            )

    signal_values = np.array([b.signal_mean for b in branches], dtype=float)
    contrast_values = np.array([b.contrast_mean for b in branches], dtype=float)
    lengths = np.array([b.length_px for b in branches], dtype=float)
    radii = np.array([max(b.radius_px, 0.1) for b in branches], dtype=float)
    signal_consistency = 1.0 - float(np.clip(np.std(signal_values) / 0.25, 0.0, 1.0))
    contrast_support = float(np.clip(np.mean(contrast_values > -0.08), 0.0, 1.0))
    median_radius = float(np.median(radii)) if len(radii) else max(local_radius, 1.0)
    min_required_len = max(6.0, 0.9 * median_radius)
    long_required_len = max(12.0, 2.0 * median_radius)
    short_branch = float(lengths.min()) < min_required_len
    lacks_two_stable_branches = int(np.sum(lengths >= long_required_len)) < 2
    length_support = float(
        np.mean(
            [
                np.clip(branch.length_px / max(2.5 * max(branch.radius_px, local_radius, 1.0), 1.0), 0.0, 1.0)
                for branch in branches
            ]
        )
    )
    local_radius_score = float(np.clip(local_radius / max(np.median([b.radius_px for b in branches]), 1.0), 0.0, 1.4) / 1.4)
    murray, parent_idx = murray_score(branches[:3] if degree == 3 else branches)

    if degree == 3:
        if short_branch or lacks_two_stable_branches:
            return {
                "type": "short_branch_artifact",
                "confidence": round(float(np.clip(0.45 + 0.20 * signal_consistency, 0.0, 0.65)), 4),
                "min_branch_length_px": round(float(lengths.min()), 3),
                "min_required_length_px": round(float(min_required_len), 3),
                "stable_branch_threshold_px": round(float(long_required_len), 3),
                "stable_branch_count": int(np.sum(lengths >= long_required_len)),
                "pairwise": pairwise,
            }
        angles = [x["angle_deg"] for x in pairwise]
        spread = float(np.clip((max(angles) - min(angles)) / 120.0, 0.0, 1.0)) if angles else 0.0
        confidence = float(
            np.clip(
                0.30
                + 0.25 * murray
                + 0.15 * signal_consistency
                + 0.13 * contrast_support
                + 0.12 * length_support
                + 0.08 * local_radius_score
                + 0.10 * spread,
                0.0,
                1.0,
            )
        )
        node_type = "bifurcation" if confidence >= 0.52 else "uncertain_bifurcation"
        return {
            "type": node_type,
            "confidence": round(confidence, 4),
            "parent_edge_candidate": branches[parent_idx].edge_id if parent_idx >= 0 else None,
            "murray_score": round(murray, 4),
            "pairwise": pairwise,
        }

    if degree == 4:
        crossing_score, pairs = best_crossing_pairing(branches)
        murray_all, parent_all = murray_score(branches)
        crossing_conf = float(
            np.clip(
                0.62 * crossing_score + 0.18 * signal_consistency + 0.12 * contrast_support + 0.08 * local_radius_score,
                0.0,
                1.0,
            )
        )
        if crossing_conf >= 0.62:
            node_type = "crossing_or_overlap"
            confidence = crossing_conf
        else:
            node_type = "uncertain_high_order_branch"
            confidence = float(np.clip(0.40 + 0.25 * murray_all + 0.20 * signal_consistency, 0.0, 0.82))
        return {
            "type": node_type,
            "confidence": round(confidence, 4),
            "crossing_score": round(crossing_score, 4),
            "crossing_edge_pairs": [[branches[i].edge_id, branches[j].edge_id] for i, j in pairs],
            "parent_edge_candidate": branches[parent_all].edge_id if parent_all >= 0 else None,
            "murray_score": round(murray_all, 4),
            "pairwise": pairwise,
        }

    confidence = float(np.clip(0.35 + 0.25 * signal_consistency + 0.20 * contrast_support, 0.0, 0.75))
    return {
        "type": "uncertain_complex_junction",
        "confidence": round(confidence, 4),
        "pairwise": pairwise,
    }


def analyze_graph(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    name: str,
    include_edge_paths: bool = False,
    write_json_file: bool = True,
    render_overlay_file: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    gray = load_gray(image_path)
    raw_mask = load_binary_mask(mask_path)
    if gray.shape != raw_mask.shape:
        raise ValueError(f"image/mask shape mismatch: {gray.shape} vs {raw_mask.shape}")

    mask = preprocess_mask(raw_mask) & raw_mask
    if not mask.any():
        mask = raw_mask.copy()
    skeleton = zhang_suen_thinning(mask)
    skeleton = prune_short_branches(skeleton, min_length=10, rounds=24)
    graph, node_data = build_contracted_skeleton_graph(skeleton)

    image01 = normalize_image(gray)
    signal, polarity = vessel_signal_image(image01, mask)
    grad_y = ndi.sobel(signal, axis=0)
    grad_x = ndi.sobel(signal, axis=1)
    gradient = np.hypot(grad_x, grad_y).astype(np.float32)
    dist = ndi.distance_transform_edt(mask)
    image_meta = parse_case_metadata(image_path)
    mask_meta = parse_case_metadata(mask_path)

    nodes_payload = []
    for node in sorted(graph.nodes):
        pixel = tuple(graph.nodes[node]["pixel"])
        graph_degree = int(graph.degree[node])
        if graph_degree >= 3:
            kind = "junction"
        elif graph_degree <= 1:
            kind = "endpoint"
        else:
            kind = "connector"
        branches = [
            describe_branch(graph, node, nbr, pixel, signal, gradient, dist)
            for nbr in sorted(graph.neighbors(node))
        ]
        classification = classify_node(branches, float(dist[pixel])) if len(branches) >= 3 else None
        nodes_payload.append(
            {
                "id": int(node),
                "pixel_rc": [int(pixel[0]), int(pixel[1])],
                "degree": graph_degree,
                "kind": kind,
                "classification": classification,
                "branches": [
                    {
                        "edge_id": branch.edge_id,
                        "neighbor_node": branch.neighbor_node,
                        "direction_xy": np.round(branch.direction_xy, 4).tolist(),
                        "angle_deg": round(branch.angle_deg, 3),
                        "length_px": round(branch.length_px, 3),
                        "radius_px": round(branch.radius_px, 3),
                        "signal_mean": round(branch.signal_mean, 5),
                        "signal_std": round(branch.signal_std, 5),
                        "contrast_mean": round(branch.contrast_mean, 5),
                        "gradient_mean": round(branch.gradient_mean, 5),
                    }
                    for branch in branches
                ],
            }
        )

    edges_payload = []
    for u, v, data in sorted(graph.edges(data=True), key=lambda item: int(item[2]["id"])):
        path = data["path"]
        radii = [float(dist[r, c]) for r, c in path]
        values = [float(signal[r, c]) for r, c in path]
        edge_payload = {
            "id": int(data["id"]),
            "source": int(u),
            "target": int(v),
            "length_px": round(float(data["length"]), 3),
            "radius_median_px": round(float(np.median(radii)), 3) if radii else 0.0,
            "signal_mean": round(float(np.mean(values)), 5) if values else 0.0,
        }
        if include_edge_paths:
            edge_payload["path_rc"] = [[int(r), int(c)] for r, c in path]
        edges_payload.append(edge_payload)

    summary = {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "image_meta": image_meta,
        "mask_meta": mask_meta,
        "name": name,
        "polarity": polarity,
        "mask_pixels": int(raw_mask.sum()),
        "graph_mask_pixels": int(mask.sum()),
        "skeleton_pixels": int(skeleton.sum()),
        "nodes": len(nodes_payload),
        "edges": len(edges_payload),
        "junction_counts": {},
        "statistics": image_stats(gray, mask, signal, gradient),
    }
    for node in nodes_payload:
        cls = node["classification"]
        if cls is None:
            continue
        node_type = str(cls["type"])
        summary["junction_counts"][node_type] = int(summary["junction_counts"].get(node_type, 0)) + 1

    payload = {
        "summary": summary,
        "nodes": nodes_payload,
        "edges": edges_payload,
    }

    json_path = output_dir / f"{name}_graph_analysis.json"
    overlay_path = output_dir / f"{name}_graph_overlay.png"
    payload["summary"]["json_path"] = str(json_path) if write_json_file else None
    payload["summary"]["overlay_path"] = str(overlay_path) if render_overlay_file else None
    if write_json_file:
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if render_overlay_file:
        render_overlay(image01, raw_mask, mask, skeleton, graph, nodes_payload, summary, overlay_path)
    return payload


def render_overlay(
    image01: np.ndarray,
    raw_mask: np.ndarray,
    graph_mask: np.ndarray,
    skeleton: np.ndarray,
    graph: nx.Graph,
    nodes_payload: list[dict[str, object]],
    summary: dict[str, object],
    path: Path,
) -> None:
    base = np.repeat((image01 * 255).clip(0, 255).astype(np.uint8)[..., None], 3, axis=2)
    outline = raw_mask ^ ndi.binary_erosion(raw_mask, iterations=1)
    base[outline] = (255, 190, 50)
    removed = raw_mask & ~graph_mask
    base[removed] = (110, 85, 30)
    base[skeleton] = (230, 40, 40)

    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image)

    edge_color = (255, 80, 80)
    for _, _, data in graph.edges(data=True):
        path_pixels = data["path"]
        if len(path_pixels) >= 2:
            draw.line([(c, r) for r, c in path_pixels], fill=edge_color, width=1)

    colors = {
        "bifurcation": (40, 230, 90),
        "uncertain_bifurcation": (230, 210, 40),
        "short_branch_artifact": (160, 160, 160),
        "crossing_or_overlap": (80, 160, 255),
        "uncertain_high_order_branch": (255, 140, 40),
        "uncertain_complex_junction": (255, 80, 210),
    }
    for node in nodes_payload:
        r, c = node["pixel_rc"]
        cls = node["classification"]
        if cls is None:
            if int(node["degree"]) <= 1:
                color = (60, 220, 220)
                radius = 2
            else:
                continue
        else:
            color = colors.get(str(cls["type"]), (255, 255, 255))
            radius = 5
        draw.ellipse((c - radius, r - radius, c + radius, r + radius), outline=(0, 0, 0), width=2)
        draw.ellipse((c - radius + 1, r - radius + 1, c + radius - 1, r + radius - 1), fill=color)

    panel_w = 420
    canvas = Image.new("RGB", (image.width + panel_w, image.height), (22, 24, 28))
    canvas.paste(image, (0, 0))
    panel = ImageDraw.Draw(canvas)
    x0 = image.width + 18
    y = 18

    def line(text: str, fill: tuple[int, int, int] = (235, 238, 242), step: int = 18) -> None:
        nonlocal y
        panel.text((x0, y), text[:58], fill=fill)
        y += step

    image_meta = summary.get("image_meta", {})
    mask_meta = summary.get("mask_meta", {})
    junction_counts = summary.get("junction_counts", {})
    stats = summary.get("statistics", {})
    line("Coronary graph analysis", (255, 255, 255), 24)
    line(f"name: {summary.get('name', '')}")
    line(f"patient: {image_meta.get('patient_id', 'n/a')}")
    line(f"series/frame: {image_meta.get('series_id', 'n/a')} / {image_meta.get('frame_id', 'n/a')}")
    line(f"annotator: {mask_meta.get('annotator_id', 'n/a')} {mask_meta.get('annotator_name', '')}")
    line(f"polarity: {summary.get('polarity', '')}")
    line(f"image: {image_meta.get('filename', '')}")
    line(f"mask: {mask_meta.get('filename', '')}")
    line(f"shape: {stats.get('shape_hw', ['?', '?'])[1]}x{stats.get('shape_hw', ['?', '?'])[0]}")
    line("", step=10)
    line("Counts", (255, 255, 255), 22)
    line(f"source mask px: {summary.get('mask_pixels', 0)}")
    line(f"graph mask px: {summary.get('graph_mask_pixels', 0)}")
    line(f"skeleton px: {summary.get('skeleton_pixels', 0)}")
    line(f"nodes / edges: {summary.get('nodes', 0)} / {summary.get('edges', 0)}")
    if junction_counts:
        for key, value in sorted(junction_counts.items()):
            line(f"{key}: {value}")
    else:
        line("junctions: 0")
    line("", step=10)
    line("Signal statistics", (255, 255, 255), 22)
    line(f"vessel mean/std: {stats.get('vessel_signal_mean', 0):.3f} / {stats.get('vessel_signal_std', 0):.3f}")
    line(f"background mean: {stats.get('background_signal_mean', 0):.3f}")
    line(f"gradient mean/p95: {stats.get('mask_gradient_mean', 0):.3f} / {stats.get('mask_gradient_p95', 0):.3f}")
    line("", step=10)
    line("Legend", (255, 255, 255), 22)
    legend_items = [
        ("source mask outline", (255, 190, 50)),
        ("masked out by cleanup", (110, 85, 30)),
        ("skeleton/edges", edge_color),
        ("endpoint", (60, 220, 220)),
        ("bifurcation", colors["bifurcation"]),
        ("short branch artifact", colors["short_branch_artifact"]),
        ("uncertain bifurcation", colors["uncertain_bifurcation"]),
        ("crossing/overlap", colors["crossing_or_overlap"]),
        ("complex/uncertain", colors["uncertain_complex_junction"]),
    ]
    for label, color in legend_items:
        panel.rectangle((x0, y + 3, x0 + 13, y + 16), fill=color, outline=(0, 0, 0))
        panel.text((x0 + 22, y), label, fill=(235, 238, 242))
        y += 19
    canvas.save(path)


def flatten_summary(result: dict[str, object]) -> dict[str, object]:
    summary = result["summary"]
    counts = summary.get("junction_counts", {})
    stats = summary.get("statistics", {})
    image_meta = summary.get("image_meta", {})
    mask_meta = summary.get("mask_meta", {})
    nodes = result.get("nodes", [])
    junction_conf = [
        float(node["classification"]["confidence"])
        for node in nodes
        if node.get("classification") is not None
    ]
    branch_radii = [
        float(branch["radius_px"])
        for node in nodes
        for branch in node.get("branches", [])
    ]
    branch_gradients = [
        float(branch["gradient_mean"])
        for node in nodes
        for branch in node.get("branches", [])
    ]
    return {
        "name": summary.get("name"),
        "patient_id": image_meta.get("patient_id"),
        "series_id": image_meta.get("series_id"),
        "frame_id": image_meta.get("frame_id"),
        "annotator_id": mask_meta.get("annotator_id"),
        "annotator_name": mask_meta.get("annotator_name"),
        "mask_pixels": summary.get("mask_pixels"),
        "skeleton_pixels": summary.get("skeleton_pixels"),
        "nodes": summary.get("nodes"),
        "edges": summary.get("edges"),
        "bifurcation": counts.get("bifurcation", 0),
        "uncertain_bifurcation": counts.get("uncertain_bifurcation", 0),
        "short_branch_artifact": counts.get("short_branch_artifact", 0),
        "crossing_or_overlap": counts.get("crossing_or_overlap", 0),
        "uncertain_high_order_branch": counts.get("uncertain_high_order_branch", 0),
        "uncertain_complex_junction": counts.get("uncertain_complex_junction", 0),
        "junction_confidence_mean": float(np.mean(junction_conf)) if junction_conf else 0.0,
        "branch_radius_mean": float(np.mean(branch_radii)) if branch_radii else 0.0,
        "branch_gradient_mean": float(np.mean(branch_gradients)) if branch_gradients else 0.0,
        "vessel_signal_mean": stats.get("vessel_signal_mean", 0.0),
        "vessel_signal_std": stats.get("vessel_signal_std", 0.0),
        "background_signal_mean": stats.get("background_signal_mean", 0.0),
        "mask_gradient_mean": stats.get("mask_gradient_mean", 0.0),
        "json_path": summary.get("json_path"),
        "overlay_path": summary.get("overlay_path"),
    }


def write_batch_statistics(results: list[dict[str, object]], output_dir: Path) -> dict[str, object]:
    rows = [flatten_summary(result) for result in results]
    csv_path = output_dir / "batch_graph_analysis_table.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    all_edges = []
    all_junctions = []
    for result in results:
        name = result["summary"]["name"]
        image_meta = result["summary"].get("image_meta", {})
        mask_meta = result["summary"].get("mask_meta", {})
        for edge in result.get("edges", []):
            all_edges.append(
                {
                    "name": name,
                    "series_id": image_meta.get("series_id"),
                    "frame_id": image_meta.get("frame_id"),
                    "annotator_id": mask_meta.get("annotator_id"),
                    "length_px": float(edge.get("length_px", 0.0)),
                    "radius_median_px": float(edge.get("radius_median_px", 0.0)),
                    "signal_mean": float(edge.get("signal_mean", 0.0)),
                }
            )
        for node in result.get("nodes", []):
            cls = node.get("classification")
            if cls is None:
                continue
            all_junctions.append(
                {
                    "name": name,
                    "series_id": image_meta.get("series_id"),
                    "frame_id": image_meta.get("frame_id"),
                    "annotator_id": mask_meta.get("annotator_id"),
                    "node_id": int(node["id"]),
                    "type": cls.get("type"),
                    "confidence": float(cls.get("confidence", 0.0)),
                    "degree": int(node.get("degree", 0)),
                }
            )

    edges_csv = output_dir / "batch_edges_table.csv"
    if all_edges:
        with edges_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_edges[0].keys()))
            writer.writeheader()
            writer.writerows(all_edges)

    junctions_csv = output_dir / "batch_junctions_table.csv"
    if all_junctions:
        with junctions_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_junctions[0].keys()))
            writer.writeheader()
            writer.writerows(all_junctions)

    plots = render_batch_plots(rows, all_edges, all_junctions, output_dir)
    report_path = write_batch_report(rows, all_edges, all_junctions, plots, output_dir)
    return {
        "rows": rows,
        "edges": len(all_edges),
        "junctions": len(all_junctions),
        "csv_path": str(csv_path),
        "edges_csv": str(edges_csv) if all_edges else None,
        "junctions_csv": str(junctions_csv) if all_junctions else None,
        "plots": plots,
        "report_path": str(report_path),
    }


def render_batch_plots(
    rows: list[dict[str, object]],
    edges: list[dict[str, object]],
    junctions: list[dict[str, object]],
    output_dir: Path,
) -> list[str]:
    plots: list[str] = []
    if not rows:
        return plots

    frame_labels = [str(row["name"]) for row in rows]
    x = np.arange(len(rows))
    bif = np.array([float(row["bifurcation"]) for row in rows])
    cross = np.array([float(row["crossing_or_overlap"]) for row in rows])
    uncertain = np.array(
        [
            float(row["uncertain_bifurcation"])
            + float(row["short_branch_artifact"])
            + float(row["uncertain_high_order_branch"])
            + float(row["uncertain_complex_junction"])
            for row in rows
        ]
    )

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar(x, bif, label="bifurcation", color="#28d85c")
    ax.bar(x, cross, bottom=bif, label="crossing/overlap", color="#50a0ff")
    ax.bar(x, uncertain, bottom=bif + cross, label="uncertain", color="#f0c828")
    ax.set_title("Junction classes per image")
    ax.set_xlabel("image index, sorted by filename")
    ax.set_ylabel("count")
    tick_step = max(1, len(rows) // 18)
    tick_positions = x[::tick_step]
    tick_labels = [
        f"{rows[i].get('series_id', '')}/{rows[i].get('frame_id', '')}/{rows[i].get('annotator_id', '')}"
        for i in tick_positions
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=55, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "plot_junction_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plots.append(str(path))

    if edges:
        radii = np.array([edge["radius_median_px"] for edge in edges], dtype=float)
        lengths = np.array([edge["length_px"] for edge in edges], dtype=float)
        signals = np.array([edge["signal_mean"] for edge in edges], dtype=float)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].hist(radii, bins=35, color="#5876d6")
        axes[0].set_title("Edge radius distribution")
        axes[0].set_xlabel("median radius, px")
        axes[1].hist(lengths, bins=35, color="#d65f58")
        axes[1].set_title("Edge length distribution")
        axes[1].set_xlabel("length, px")
        axes[2].hist(signals, bins=35, color="#50a070")
        axes[2].set_title("Centerline signal distribution")
        axes[2].set_xlabel("normalized vessel signal")
        fig.tight_layout()
        path = output_dir / "plot_edge_distributions.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(str(path))

        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        ax.scatter(radii, lengths, s=12, alpha=0.45, color="#5876d6")
        ax.set_title("Edge length vs radius")
        ax.set_xlabel("median radius, px")
        ax.set_ylabel("length, px")
        fig.tight_layout()
        path = output_dir / "plot_length_vs_radius.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(str(path))

    if junctions:
        conf = np.array([junction["confidence"] for junction in junctions], dtype=float)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(conf, bins=np.linspace(0, 1, 21), color="#28d85c", edgecolor="white")
        ax.set_title("Junction confidence distribution")
        ax.set_xlabel("confidence")
        ax.set_ylabel("count")
        fig.tight_layout()
        path = output_dir / "plot_junction_confidence.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(str(path))

    annotators = sorted({str(row.get("annotator_name") or row.get("annotator_id") or "unknown") for row in rows})
    if annotators:
        bif_by_annotator = []
        cross_by_annotator = []
        uncertain_by_annotator = []
        for annotator in annotators:
            subset = [
                row
                for row in rows
                if str(row.get("annotator_name") or row.get("annotator_id") or "unknown") == annotator
            ]
            bif_by_annotator.append(sum(float(row["bifurcation"]) for row in subset))
            cross_by_annotator.append(sum(float(row["crossing_or_overlap"]) for row in subset))
            uncertain_by_annotator.append(
                sum(
                    float(row["uncertain_bifurcation"])
                    + float(row["short_branch_artifact"])
                    + float(row["uncertain_high_order_branch"])
                    + float(row["uncertain_complex_junction"])
                    for row in subset
                )
            )
        fig, ax = plt.subplots(figsize=(9, 4.8))
        xi = np.arange(len(annotators))
        ax.bar(xi, bif_by_annotator, label="bifurcation", color="#28d85c")
        ax.bar(xi, cross_by_annotator, bottom=bif_by_annotator, label="crossing/overlap", color="#50a0ff")
        ax.bar(
            xi,
            uncertain_by_annotator,
            bottom=np.array(bif_by_annotator) + np.array(cross_by_annotator),
            label="uncertain",
            color="#f0c828",
        )
        ax.set_title("Junction classes by annotator")
        ax.set_ylabel("count")
        ax.set_xticks(xi)
        ax.set_xticklabels(annotators, rotation=35, ha="right")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "plot_junctions_by_annotator.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(str(path))

    return plots


def write_batch_report(
    rows: list[dict[str, object]],
    edges: list[dict[str, object]],
    junctions: list[dict[str, object]],
    plots: list[str],
    output_dir: Path,
) -> Path:
    report_path = output_dir / "batch_graph_analysis_report.md"
    if not rows:
        report_path.write_text("# Batch graph analysis\n\nNo rows.\n", encoding="utf-8")
        return report_path

    total_bif = sum(int(row["bifurcation"]) for row in rows)
    total_cross = sum(int(row["crossing_or_overlap"]) for row in rows)
    total_uncertain = sum(
        int(row["uncertain_bifurcation"])
        + int(row["short_branch_artifact"])
        + int(row["uncertain_high_order_branch"])
        + int(row["uncertain_complex_junction"])
        for row in rows
    )
    radius_values = [edge["radius_median_px"] for edge in edges]
    length_values = [edge["length_px"] for edge in edges]
    conf_values = [junction["confidence"] for junction in junctions]

    lines = [
        "# Batch graph analysis",
        "",
        "Сводка по статистическому анализу пар `image + mask` через skeleton-граф.",
        "",
        "## Пояснение",
        "",
        "- `bifurcation` ставится в основном по degree-3 узлу skeleton-графа, углам ветвей, локальным радиусам, Murray-like балансу радиусов, согласованности серого сигнала и локальному контрасту.",
        "- `crossing_or_overlap` ставится в основном по degree-4 узлу, где есть две пары почти противоположных продолжений с похожими радиусами и сигналом.",
        "- `uncertain_*` нужно рассматривать вручную: одна 2D-проекция не всегда отделяет реальную анатомическую бифуркацию от проекционного наложения.",
        "",
        "## Dataset summary",
        "",
        f"- Images analyzed: {len(rows)}",
        f"- Edges analyzed: {len(edges)}",
        f"- Junctions analyzed: {len(junctions)}",
        f"- Bifurcations: {total_bif}",
        f"- Crossings/overlaps: {total_cross}",
        f"- Uncertain junctions: {total_uncertain}",
        f"- Mean edge radius, px: {np.mean(radius_values):.3f}" if radius_values else "- Mean edge radius, px: n/a",
        f"- Median edge length, px: {np.median(length_values):.3f}" if length_values else "- Median edge length, px: n/a",
        f"- Mean junction confidence: {np.mean(conf_values):.3f}" if conf_values else "- Mean junction confidence: n/a",
        "",
        "## Generated plots",
        "",
    ]
    for plot in plots:
        lines.append(f"- `{Path(plot).name}`")
    lines.extend(
        [
            "",
            "## Output tables",
            "",
            "- `batch_graph_analysis_table.csv`: per-image graph and signal statistics.",
            "- `batch_edges_table.csv`: per-edge length/radius/signal statistics.",
            "- `batch_junctions_table.csv`: per-junction class and confidence.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def iter_batch(image_dir: Path, mask_dir: Path, pattern: str) -> list[tuple[Path, Path, str]]:
    out = []
    for image_path in sorted(image_dir.glob(pattern)):
        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            out.append((image_path, mask_path, image_path.stem))
    if not out:
        raise FileNotFoundError(f"no image/mask pairs for {image_dir}/{pattern} and {mask_dir}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 2D coronary skeleton graph and classify bifurcation/crossing junctions from image+mask."
    )
    parser.add_argument("--image", type=Path, help="Single grayscale angiography image.")
    parser.add_argument("--mask", type=Path, help="Single binary vessel mask.")
    parser.add_argument("--image-dir", type=Path, help="Batch image directory.")
    parser.add_argument("--mask-dir", type=Path, help="Batch mask directory with matching filenames.")
    parser.add_argument("--glob", default="*.png", help="Batch filename glob. Default: *.png")
    parser.add_argument("--output-dir", type=Path, default=Path("3d/outputs_graph_analysis"))
    parser.add_argument("--name", help="Output basename for a single image/mask pair.")
    parser.add_argument("--include-edge-paths", action="store_true", help="Store full edge pixel paths in per-image JSON.")
    parser.add_argument("--no-json", action="store_true", help="Do not write per-image JSON files; still writes batch tables.")
    parser.add_argument("--no-overlays", action="store_true", help="Do not write overlay PNG files.")
    parser.add_argument(
        "--max-overlays",
        type=int,
        default=None,
        help="Maximum overlay PNG files to write during a batch run. Default: all for single run, 12 for batch.",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only final batch paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image and args.mask:
        pairs = [(args.image, args.mask, args.name or args.image.stem)]
    elif args.image_dir and args.mask_dir:
        pairs = iter_batch(args.image_dir, args.mask_dir, args.glob)
    else:
        raise SystemExit("Provide either --image/--mask or --image-dir/--mask-dir.")

    results = []
    default_max_overlays = len(pairs) if len(pairs) == 1 else 12
    max_overlays = default_max_overlays if args.max_overlays is None else max(0, args.max_overlays)
    for index, (image_path, mask_path, name) in enumerate(pairs):
        render_overlay_file = (not args.no_overlays) and index < max_overlays
        result = analyze_graph(
            image_path,
            mask_path,
            args.output_dir,
            name,
            include_edge_paths=args.include_edge_paths,
            write_json_file=not args.no_json,
            render_overlay_file=render_overlay_file,
        )
        results.append(result)
        if not args.quiet:
            print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    if len(results) > 1:
        summaries = [result["summary"] for result in results]
        summary_path = args.output_dir / "batch_graph_analysis_summary.json"
        summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        stats = write_batch_statistics(results, args.output_dir)
        stats_path = args.output_dir / "batch_graph_analysis_statistics.json"
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"batch_summary={summary_path}")
        print(f"batch_statistics={stats_path}")


if __name__ == "__main__":
    main()
