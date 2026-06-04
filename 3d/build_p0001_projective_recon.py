#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from reconstruct_coronary_tree import (
    EdgeModel,
    build_graph_from_skeleton,
    cumulative_lengths,
    endpoint_root,
    load_binary_mask,
    polyline_length,
    preprocess_mask,
    prune_short_branches,
    write_centerlines_obj,
    write_obj_mesh,
    zhang_suen_thinning,
)


LEFT_SERIES = ("00000001", "00000002", "00000003")


@dataclass
class LeafPath:
    leaf: int
    length: float
    points: np.ndarray
    endpoint: np.ndarray
    branch_id: int
    enh_profile: np.ndarray
    time_profile: np.ndarray
    mask_support_profile: np.ndarray
    mask_time_profile: np.ndarray
    anatomy_signature: np.ndarray
    upstream_bifurcation: int | None
    upstream_bifurcation_signature: np.ndarray
    upstream_prior_score: float
    upstream_role_label: str
    distal_length_from_bifurcation: float


@dataclass
class BifurcationNode:
    node: int
    point: np.ndarray
    path_length: float
    bifurcation_level: int
    child_count: int
    subtree_leaves: int
    root_branch: int
    signature: np.ndarray
    prior_level_score: float
    prior_rank_score: float
    role_label: str


@dataclass
class ProximalSegmentMatch:
    source_node: int
    target_node: int
    parent_source: int
    parent_target: int
    bifurcation_level: int
    points_a: np.ndarray
    points_b: np.ndarray
    length_a: float
    length_b: float


@dataclass
class SeriesView:
    series: str
    mask_path: Path
    mask: np.ndarray
    graph: nx.Graph
    node_pixels: dict[int, tuple[int, int]]
    root: int
    root_xy: np.ndarray
    leaves: list[LeafPath]
    centroid: np.ndarray
    basis: np.ndarray
    scales: np.ndarray
    enhancement_map: np.ndarray
    peak_time_map: np.ndarray
    mask_support_map: np.ndarray
    mask_time_map: np.ndarray
    bifurcations: list[BifurcationNode]
    parent_map: dict[int, int | None]
    children: dict[int, list[int]]
    path_lengths: dict[int, float]


def pixel_to_xy(pixel: tuple[int, int]) -> np.ndarray:
    r, c = pixel
    return np.array([float(c), float(r)], dtype=float)


def resample_curve(points: np.ndarray, n_samples: int) -> np.ndarray:
    if len(points) == 1:
        return np.repeat(points, n_samples, axis=0)
    s = cumulative_lengths(points)
    targets = np.linspace(0.0, s[-1], n_samples)
    return np.column_stack([np.interp(targets, s, points[:, i]) for i in range(points.shape[1])])


def direction_unit(points: np.ndarray, at_start: bool, step: int = 6) -> np.ndarray:
    if len(points) < 2:
        return np.array([1.0, 0.0], dtype=float)
    if at_start:
        delta = points[min(step, len(points) - 1)] - points[0]
    else:
        delta = points[-1] - points[max(0, len(points) - 1 - step)]
    norm = np.linalg.norm(delta)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=float)
    return delta / norm


def common_prefix_ratio(points_a: np.ndarray, points_b: np.ndarray, tol: float = 12.0) -> float:
    samples_a = resample_curve(points_a, 48)
    samples_b = resample_curve(points_b, 48)
    distances = np.linalg.norm(samples_a - samples_b, axis=1)
    keep = distances < tol
    prefix = np.cumprod(keep.astype(np.int8)).astype(bool)
    if not np.any(prefix):
        return 0.0
    last = int(np.max(np.where(prefix)[0]))
    return float(last / (len(distances) - 1))


def sample_map_mean(image: np.ndarray, point_xy: np.ndarray, radius: int = 4) -> float:
    cx = int(round(float(point_xy[0])))
    cy = int(round(float(point_xy[1])))
    x0 = max(0, cx - radius)
    x1 = min(image.shape[1], cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(image.shape[0], cy + radius + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch))


def prior_residual(signature: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    dim = min(len(signature), len(mean), len(std))
    if dim == 0:
        return 0.0
    signature = signature[:dim]
    mean = mean[:dim]
    std = std[:dim]
    scale = np.maximum(std, 0.05)
    return float(np.mean(np.abs((signature - mean) / scale)))


def build_left_cohort_prior(
    base_dir: Path,
    exclude_patient: str = "p0001",
) -> dict[str, object] | None:
    descriptors_path = base_dir / "outputs_dataset_temporal_prior" / "series_descriptors.json"
    if not descriptors_path.exists():
        return None
    descriptors = json.loads(descriptors_path.read_text(encoding="utf-8"))
    selected = [
        item
        for item in descriptors
        if item.get("status") == "ok"
        and item.get("patient") != exclude_patient
        and item.get("series") in LEFT_SERIES
    ]
    if not selected:
        return None

    by_level: dict[int, list[np.ndarray]] = {}
    by_rank: dict[int, list[np.ndarray]] = {}
    for item in selected:
        proximal = item.get("proximal_bifurcations", [])
        for rank, bif in enumerate(proximal[:4]):
            signature = np.array(bif["signature"], dtype=float)
            level = int(bif["bifurcation_level"])
            by_level.setdefault(level, []).append(signature)
            by_rank.setdefault(rank, []).append(signature)

    if not by_level:
        return None

    return {
        "num_series": len(selected),
        "by_level": {
            level: {
                "mean": np.stack(vectors, axis=0).mean(axis=0),
                "std": np.stack(vectors, axis=0).std(axis=0),
            }
            for level, vectors in by_level.items()
        },
        "by_rank": {
            rank: {
                "mean": np.stack(vectors, axis=0).mean(axis=0),
                "std": np.stack(vectors, axis=0).std(axis=0),
            }
            for rank, vectors in by_rank.items()
        },
    }


def role_penalty(role_a: str, role_b: str) -> float:
    if role_a == role_b:
        return 0.0
    if role_a == "unassigned" or role_b == "unassigned":
        return 0.18
    compatible = {
        ("lad_axis", "diagonal_like"),
        ("diagonal_like", "lad_axis"),
        ("lcx_axis", "om_like"),
        ("om_like", "lcx_axis"),
    }
    if (role_a, role_b) in compatible:
        return 0.28
    return 0.9


def pca_frame(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    xy = coords[:, [1, 0]].astype(float)
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    scales = np.sqrt(np.maximum(values, 1e-6))
    return centroid, vectors, scales


def orient_edge_points(
    graph: nx.Graph,
    node_pixels: dict[int, tuple[int, int]],
    source: int,
    target: int,
) -> np.ndarray:
    path = graph[source][target]["path"]
    if path[0] != node_pixels[source]:
        path = list(reversed(path))
    return np.array([[float(c), float(r)] for r, c in path], dtype=float)


def build_leaf_path(
    graph: nx.Graph,
    node_pixels: dict[int, tuple[int, int]],
    root: int,
    leaf: int,
    parents: dict[int, int | None],
    branch_ids: dict[int, int],
    path_lengths: dict[int, float],
    enhancement_map: np.ndarray,
    peak_time_map: np.ndarray,
    mask_support_map: np.ndarray,
    mask_time_map: np.ndarray,
) -> LeafPath:
    nodes = []
    cur = leaf
    while cur is not None:
        nodes.append(cur)
        cur = parents[cur]
    nodes = list(reversed(nodes))

    points: list[np.ndarray] = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        segment = orient_edge_points(graph, node_pixels, a, b)
        if points:
            segment = segment[1:]
        points.append(segment)
    polyline = np.concatenate(points, axis=0) if points else np.array([pixel_to_xy(node_pixels[root])], dtype=float)
    samples = resample_curve(polyline, 16)
    rr = np.clip(np.round(samples[:, 1]).astype(int), 0, enhancement_map.shape[0] - 1)
    cc = np.clip(np.round(samples[:, 0]).astype(int), 0, enhancement_map.shape[1] - 1)
    enh_profile = enhancement_map[rr, cc].astype(float)
    if enh_profile.max() > 0:
        enh_profile = enh_profile / enh_profile.max()
    time_profile = peak_time_map[rr, cc].astype(float)
    mask_support_profile = mask_support_map[rr, cc].astype(float)
    mask_time_profile = mask_time_map[rr, cc].astype(float)
    return LeafPath(
        leaf=leaf,
        length=path_lengths[leaf],
        points=polyline,
        endpoint=polyline[-1],
        branch_id=branch_ids[leaf],
        enh_profile=enh_profile,
        time_profile=time_profile,
        mask_support_profile=mask_support_profile,
        mask_time_profile=mask_time_profile,
        anatomy_signature=np.zeros(18, dtype=float),
        upstream_bifurcation=None,
        upstream_bifurcation_signature=np.zeros(16, dtype=float),
        upstream_prior_score=0.0,
        upstream_role_label="unassigned",
        distal_length_from_bifurcation=path_lengths[leaf],
    )


def extract_view(
    series: str,
    mask_path: Path,
    enhancement_path: Path | None = None,
    peak_time_path: Path | None = None,
    mask_support_path: Path | None = None,
    mask_time_path: Path | None = None,
    cohort_prior: dict[str, object] | None = None,
) -> SeriesView:
    mask = preprocess_mask(load_binary_mask(mask_path))
    skeleton = zhang_suen_thinning(mask)
    skeleton = prune_short_branches(skeleton, min_length=18, rounds=30)
    graph, node_pixels = build_graph_from_skeleton(skeleton)
    root = endpoint_root(graph, node_pixels)
    root_xy = pixel_to_xy(node_pixels[root])
    enhancement_map = (
        np.array(Image.open(enhancement_path).convert("L"), dtype=np.float32) / 255.0
        if enhancement_path and enhancement_path.exists()
        else mask.astype(np.float32)
    )
    peak_time_map = (
        np.array(Image.open(peak_time_path).convert("RGB"), dtype=np.float32)[..., 2] / 255.0
        if peak_time_path and peak_time_path.exists()
        else np.zeros(mask.shape, dtype=np.float32)
    )
    mask_support_map = (
        np.array(Image.open(mask_support_path).convert("L"), dtype=np.float32) / 255.0
        if mask_support_path and mask_support_path.exists()
        else mask.astype(np.float32)
    )
    mask_time_map = (
        np.array(Image.open(mask_time_path).convert("L"), dtype=np.float32) / 255.0
        if mask_time_path and mask_time_path.exists()
        else np.zeros(mask.shape, dtype=np.float32)
    )

    parents = nx.single_source_shortest_path(graph, root)
    parent_map: dict[int, int | None] = {root: None}
    path_lengths = {root: 0.0}
    branch_ids = {root: -1}

    # Build rooted metadata from BFS.
    bfs_edges = list(nx.bfs_edges(graph, root))
    children: dict[int, list[int]] = {node: [] for node in graph.nodes}
    for parent, child in bfs_edges:
        parent_map[child] = parent
        children[parent].append(child)
        path_lengths[child] = path_lengths[parent] + graph[parent][child]["length"]
        branch_ids[child] = child if parent == root else branch_ids[parent]

    leaves = [node for node in graph.nodes if graph.degree[node] == 1 and node != root]
    leaf_paths = [
        build_leaf_path(
            graph,
            node_pixels,
            root,
            leaf,
            parent_map,
            branch_ids,
            path_lengths,
            enhancement_map,
            peak_time_map,
            mask_support_map,
            mask_time_map,
        )
        for leaf in leaves
        if path_lengths[leaf] > 0.22 * max(path_lengths.values())
    ]
    leaf_paths.sort(key=lambda item: item.length, reverse=True)

    # Endpoint non-maximum suppression to remove duplicated tiny terminal leaves.
    kept: list[LeafPath] = []
    for leaf in leaf_paths:
        if any(np.linalg.norm(leaf.endpoint - other.endpoint) < 28.0 for other in kept):
            continue
        kept.append(leaf)
        if len(kept) >= 12:
            break

    centroid, basis, scales = pca_frame(mask)
    max_path_length = max(path_lengths.values(), default=1.0)
    leaf_set = set(leaves)
    leaf_count_cache: dict[int, int] = {}
    bif_level_cache: dict[int, int] = {root: 0}

    def descendant_leaf_count(node: int) -> int:
        if node in leaf_count_cache:
            return leaf_count_cache[node]
        if node in leaf_set:
            leaf_count_cache[node] = 1
            return 1
        total = sum(descendant_leaf_count(child) for child in children.get(node, []))
        leaf_count_cache[node] = total
        return total

    def bifurcation_level(node: int) -> int:
        if node in bif_level_cache:
            return bif_level_cache[node]
        parent = parent_map.get(node)
        level = bifurcation_level(parent) if parent is not None else 0
        if node != root and len(children.get(node, [])) >= 2:
            level += 1
        bif_level_cache[node] = level
        return level

    bifurcations: list[BifurcationNode] = []
    max_subtree = 1
    for node in graph.nodes:
        if node == root or len(children.get(node, [])) < 2:
            continue
        max_subtree = max(max_subtree, descendant_leaf_count(node))

    for node in graph.nodes:
        node_children = children.get(node, [])
        if node == root or len(node_children) < 2:
            continue
        point_xy = pixel_to_xy(node_pixels[node])
        pca_xy = (basis.T @ (point_xy - centroid)) / np.maximum(scales, 1e-6)
        parent = parent_map.get(node)
        if parent is not None:
            parent_dir = direction_unit(orient_edge_points(graph, node_pixels, parent, node), at_start=False)
        else:
            parent_dir = np.array([1.0, 0.0], dtype=float)
        child_dirs = [
            direction_unit(orient_edge_points(graph, node_pixels, node, child), at_start=True)
            for child in node_children
        ]
        mean_child_dir = np.mean(child_dirs, axis=0)
        mean_child_norm = np.linalg.norm(mean_child_dir)
        if mean_child_norm > 1e-6:
            mean_child_dir = mean_child_dir / mean_child_norm
        else:
            mean_child_dir = np.array([1.0, 0.0], dtype=float)
        child_lengths = sorted([graph[node][child]["length"] for child in node_children], reverse=True)
        while len(child_lengths) < 2:
            child_lengths.append(0.0)
        parent_dir_pca = basis.T @ parent_dir
        child_dir_pca = basis.T @ mean_child_dir
        local_enh = sample_map_mean(enhancement_map, point_xy, radius=4)
        local_time = sample_map_mean(peak_time_map, point_xy, radius=4)
        local_mask_support = sample_map_mean(mask_support_map, point_xy, radius=4)
        local_mask_time = sample_map_mean(mask_time_map, point_xy, radius=4)
        bifurcations.append(
            BifurcationNode(
                node=node,
                point=point_xy,
                path_length=path_lengths[node],
                bifurcation_level=bifurcation_level(node),
                child_count=len(node_children),
                subtree_leaves=descendant_leaf_count(node),
                root_branch=branch_ids[node],
                signature=np.array(
                    [
                        path_lengths[node] / max(max_path_length, 1.0),
                        bifurcation_level(node) / 4.0,
                        descendant_leaf_count(node) / max(max_subtree, 1),
                        len(node_children) / 4.0,
                        pca_xy[0],
                        pca_xy[1],
                        parent_dir_pca[0],
                        parent_dir_pca[1],
                        child_dir_pca[0],
                        child_dir_pca[1],
                        child_lengths[0] / max(max_path_length, 1.0),
                        child_lengths[1] / max(max_path_length, 1.0),
                        local_enh,
                        local_time,
                        local_mask_support,
                        local_mask_time,
                    ],
                    dtype=float,
                ),
                prior_level_score=0.0,
                prior_rank_score=0.0,
                role_label="unassigned",
            )
        )
    bifurcations.sort(key=lambda item: (item.subtree_leaves, -item.path_length), reverse=True)
    bifurcations = bifurcations[:8]
    if cohort_prior is not None:
        for rank, bif in enumerate(bifurcations):
            level_payload = cohort_prior["by_level"].get(int(bif.bifurcation_level))
            rank_payload = cohort_prior["by_rank"].get(rank)
            if level_payload is not None:
                bif.prior_level_score = prior_residual(
                    bif.signature,
                    level_payload["mean"],
                    level_payload["std"],
                )
            if rank_payload is not None:
                bif.prior_rank_score = prior_residual(
                    bif.signature,
                    rank_payload["mean"],
                    rank_payload["std"],
                )
    bifurcation_by_node = {item.node: item for item in bifurcations}

    level1 = [item for item in bifurcations if item.bifurcation_level == 1]
    if level1:
        main_split = sorted(level1, key=lambda item: (-item.subtree_leaves, item.path_length))[0]
        main_split.role_label = "main_split"
        child_infos = []
        for child in children.get(main_split.node, []):
            descendant_nodes = []
            stack = [child]
            while stack:
                cur = stack.pop()
                descendant_nodes.append(cur)
                stack.extend(children.get(cur, []))
            child_bifs = [bifurcation_by_node[node] for node in descendant_nodes if node in bifurcation_by_node]
            max_path = max((path_lengths[node] for node in descendant_nodes), default=path_lengths.get(child, 0.0))
            child_time = sample_map_mean(mask_time_map, pixel_to_xy(node_pixels[child]), radius=5)
            child_infos.append(
                {
                    "child": child,
                    "leaf_count": descendant_leaf_count(child),
                    "max_path": max_path,
                    "time": child_time,
                    "bifs": child_bifs,
                }
            )
        child_infos.sort(key=lambda item: (-item["max_path"], item["time"], -item["leaf_count"]))
        if child_infos:
            lad_info = child_infos[0]
            lcx_info = child_infos[1] if len(child_infos) > 1 else None
            for bif in lad_info["bifs"]:
                if bif.node == main_split.node:
                    continue
                bif.role_label = "lad_axis" if bif.bifurcation_level <= main_split.bifurcation_level + 1 else "diagonal_like"
            if lcx_info is not None:
                for bif in lcx_info["bifs"]:
                    if bif.node == main_split.node:
                        continue
                    bif.role_label = "lcx_axis" if bif.bifurcation_level <= main_split.bifurcation_level + 1 else "om_like"

    max_length = max((leaf.length for leaf in kept), default=1.0)
    for leaf in kept:
        cur = leaf.leaf
        while cur is not None and cur not in bifurcation_by_node:
            cur = parent_map.get(cur)
        if cur is not None and cur in bifurcation_by_node:
            bif = bifurcation_by_node[cur]
            leaf.upstream_bifurcation = bif.node
            leaf.upstream_bifurcation_signature = bif.signature.copy()
            leaf.upstream_prior_score = 0.6 * bif.prior_level_score + 0.4 * bif.prior_rank_score
            leaf.upstream_role_label = bif.role_label
            leaf.distal_length_from_bifurcation = max(leaf.length - bif.path_length, 0.0)
        pca_xy = (basis.T @ (leaf.endpoint - centroid)) / np.maximum(scales, 1e-6)
        start_dir = direction_unit(leaf.points, at_start=True)
        end_dir = direction_unit(leaf.points, at_start=False)
        root_delta = leaf.endpoint - root_xy
        root_norm = np.linalg.norm(root_delta)
        root_dir = root_delta / max(root_norm, 1e-6)
        chord = np.linalg.norm(leaf.points[-1] - leaf.points[0])
        tortuosity = leaf.length / max(chord, 1.0)
        shared = sorted(
            [common_prefix_ratio(leaf.points, other.points) for other in kept if other.leaf != leaf.leaf],
            reverse=True,
        )[:4]
        while len(shared) < 4:
            shared.append(0.0)
        leaf.anatomy_signature = np.array(
            [
                leaf.length / max_length,
                pca_xy[0],
                pca_xy[1],
                start_dir[0],
                start_dir[1],
                end_dir[0],
                end_dir[1],
                root_dir[0],
                root_dir[1],
                tortuosity,
                *shared,
                float(np.mean(leaf.enh_profile)),
                float(np.mean(leaf.time_profile)),
                float(np.mean(leaf.mask_support_profile)),
                float(np.mean(leaf.mask_time_profile)),
            ],
            dtype=float,
        )
    return SeriesView(
        series=series,
        mask_path=mask_path,
        mask=mask,
        graph=graph,
        node_pixels=node_pixels,
        root=root,
        root_xy=root_xy,
        leaves=kept,
        centroid=centroid,
        basis=basis,
        scales=scales,
        enhancement_map=enhancement_map,
        peak_time_map=peak_time_map,
        mask_support_map=mask_support_map,
        mask_time_map=mask_time_map,
        bifurcations=bifurcations,
        parent_map=parent_map,
        children=children,
        path_lengths=path_lengths,
    )


def similarity_candidates(source: SeriesView, target: SeriesView) -> list[tuple[np.ndarray, np.ndarray]]:
    scale = float(np.mean(target.scales) / max(np.mean(source.scales), 1e-6))
    transforms = []
    for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        reflect = np.diag(np.array(signs, dtype=float))
        linear = target.basis @ reflect @ source.basis.T
        transforms.append((scale * linear, target.centroid - scale * linear @ source.centroid))
    return transforms


def apply_affine(points: np.ndarray, linear: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return points @ linear.T + offset


def ancestor_chain(node: int, parent_map: dict[int, int | None]) -> list[int]:
    chain = []
    cur: int | None = node
    while cur is not None:
        chain.append(cur)
        cur = parent_map.get(cur)
    return chain


def is_ancestor(ancestor: int, node: int, parent_map: dict[int, int | None]) -> bool:
    cur: int | None = node
    while cur is not None:
        if cur == ancestor:
            return True
        cur = parent_map.get(cur)
    return False


def rooted_polyline_between(
    graph: nx.Graph,
    node_pixels: dict[int, tuple[int, int]],
    parent_map: dict[int, int | None],
    start: int,
    end: int,
) -> np.ndarray:
    if start == end:
        return np.array([pixel_to_xy(node_pixels[start])], dtype=float)
    if not is_ancestor(start, end, parent_map):
        raise ValueError(f"node {start} is not an ancestor of {end}")

    nodes = []
    cur = end
    while cur != start:
        nodes.append(cur)
        cur = parent_map[cur]
        if cur is None:
            raise ValueError(f"failed to reach ancestor {start} from {end}")
    nodes.append(start)
    nodes.reverse()

    points: list[np.ndarray] = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        segment = orient_edge_points(graph, node_pixels, a, b)
        if points:
            segment = segment[1:]
        points.append(segment)
    return np.concatenate(points, axis=0) if points else np.array([pixel_to_xy(node_pixels[start])], dtype=float)


def match_bifurcations(
    source: SeriesView,
    target: SeriesView,
) -> tuple[list[tuple[BifurcationNode, BifurcationNode]], dict[str, object], np.ndarray]:
    if not source.bifurcations or not target.bifurcations:
        return [], {"candidate_index": -1, "count": 0, "score": float("inf"), "linear": None, "offset": None}, np.zeros(
            (len(source.bifurcations), len(target.bifurcations)),
            dtype=float,
        )

    best_pairs: list[tuple[BifurcationNode, BifurcationNode]] = []
    best_meta: dict[str, object] | None = None
    best_cost: np.ndarray | None = None
    target_scale = float(np.linalg.norm(target.scales))

    for idx, (linear, offset) in enumerate(similarity_candidates(source, target)):
        src_pts = np.array([item.point for item in source.bifurcations], dtype=float)
        tgt_pts = np.array([item.point for item in target.bifurcations], dtype=float)
        src_trans = apply_affine(src_pts, linear, offset)

        cost = np.zeros((len(source.bifurcations), len(target.bifurcations)), dtype=float)
        for i, bif_a in enumerate(source.bifurcations):
            for j, bif_b in enumerate(target.bifurcations):
                dist = np.linalg.norm(src_trans[i] - tgt_pts[j]) / max(target_scale, 1.0)
                path_diff = abs(bif_a.path_length - bif_b.path_length) / max(max(bif_a.path_length, bif_b.path_length), 1.0)
                subtree_diff = abs(bif_a.subtree_leaves - bif_b.subtree_leaves) / max(
                    max(bif_a.subtree_leaves, bif_b.subtree_leaves),
                    1.0,
                )
                child_diff = abs(bif_a.child_count - bif_b.child_count) / 4.0
                branch_diff = 0.0 if bif_a.root_branch == bif_b.root_branch else 0.18
                sig_diff = float(np.mean(np.abs(bif_a.signature - bif_b.signature)))
                role_diff = role_penalty(bif_a.role_label, bif_b.role_label)
                prior_mean = 0.5 * (
                    bif_a.prior_level_score
                    + bif_b.prior_level_score
                    + 0.7 * bif_a.prior_rank_score
                    + 0.7 * bif_b.prior_rank_score
                )
                prior_diff = abs(bif_a.prior_level_score - bif_b.prior_level_score) + 0.6 * abs(
                    bif_a.prior_rank_score - bif_b.prior_rank_score
                )
                cost[i, j] = (
                    dist
                    + 0.4 * path_diff
                    + 0.28 * subtree_diff
                    + 0.22 * child_diff
                    + 0.12 * branch_diff
                    + 0.42 * role_diff
                    + 0.62 * sig_diff
                    + 0.14 * prior_mean
                    + 0.1 * prior_diff
                )

        rows, cols = linear_sum_assignment(cost)
        pairs = []
        accepted_costs = []
        for r, c in zip(rows, cols):
            if cost[r, c] > 1.2:
                continue
            pairs.append((source.bifurcations[r], target.bifurcations[c]))
            accepted_costs.append(float(cost[r, c]))
        if not pairs:
            continue
        score = float(np.mean(accepted_costs))
        if best_meta is None or len(pairs) > best_meta["count"] or (
            len(pairs) == best_meta["count"] and score < best_meta["score"]
        ):
            best_pairs = pairs
            best_cost = cost.copy()
            best_meta = {
                "candidate_index": idx,
                "count": len(pairs),
                "score": score,
                "linear": linear.tolist(),
                "offset": offset.tolist(),
            }

    if best_meta is None:
        return [], {"candidate_index": -1, "count": 0, "score": float("inf"), "linear": None, "offset": None}, np.full(
            (len(source.bifurcations), len(target.bifurcations)),
            10.0,
            dtype=float,
        )
    if best_cost is None:
        raise RuntimeError(f"failed to build bifurcation cost matrix for {source.series} vs {target.series}")
    return best_pairs, best_meta, best_cost


def match_leaf_paths(
    source: SeriesView,
    target: SeriesView,
    bifurcation_map: dict[int, int] | None = None,
) -> tuple[list[tuple[LeafPath, LeafPath]], dict[str, object], np.ndarray]:
    best_pairs: list[tuple[LeafPath, LeafPath]] = []
    best_meta: dict[str, object] | None = None
    best_cost: np.ndarray | None = None
    target_scale = float(np.linalg.norm(target.scales))

    for idx, (linear, offset) in enumerate(similarity_candidates(source, target)):
        src_pts = np.array([leaf.endpoint for leaf in source.leaves], dtype=float)
        tgt_pts = np.array([leaf.endpoint for leaf in target.leaves], dtype=float)
        src_trans = apply_affine(src_pts, linear, offset)

        cost = np.zeros((len(source.leaves), len(target.leaves)), dtype=float)
        for i, leaf_a in enumerate(source.leaves):
            for j, leaf_b in enumerate(target.leaves):
                dist = np.linalg.norm(src_trans[i] - tgt_pts[j]) / max(target_scale, 1.0)
                length_diff = abs(leaf_a.length - leaf_b.length) / max(max(leaf_a.length, leaf_b.length), 1.0)
                branch_diff = 0.0 if leaf_a.branch_id == leaf_b.branch_id else 0.25
                enh_diff = float(np.mean(np.abs(leaf_a.enh_profile - leaf_b.enh_profile)))
                time_diff = float(np.mean(np.abs(leaf_a.time_profile - leaf_b.time_profile)))
                mask_support_diff = float(np.mean(np.abs(leaf_a.mask_support_profile - leaf_b.mask_support_profile)))
                mask_time_diff = float(np.mean(np.abs(leaf_a.mask_time_profile - leaf_b.mask_time_profile)))
                anatomy_diff = float(np.mean(np.abs(leaf_a.anatomy_signature - leaf_b.anatomy_signature)))
                bif_sig_diff = float(
                    np.mean(np.abs(leaf_a.upstream_bifurcation_signature - leaf_b.upstream_bifurcation_signature))
                )
                prior_upstream_mean = 0.5 * (leaf_a.upstream_prior_score + leaf_b.upstream_prior_score)
                prior_upstream_diff = abs(leaf_a.upstream_prior_score - leaf_b.upstream_prior_score)
                role_diff = role_penalty(leaf_a.upstream_role_label, leaf_b.upstream_role_label)
                distal_diff = abs(
                    leaf_a.distal_length_from_bifurcation - leaf_b.distal_length_from_bifurcation
                ) / max(
                    max(leaf_a.distal_length_from_bifurcation, leaf_b.distal_length_from_bifurcation),
                    1.0,
                )
                bif_consistency = 0.0
                if bifurcation_map and leaf_a.upstream_bifurcation is not None:
                    target_bif = bifurcation_map.get(int(leaf_a.upstream_bifurcation))
                    if target_bif is None:
                        bif_consistency += 0.12
                    elif leaf_b.upstream_bifurcation != target_bif:
                        bif_consistency += 0.55
                elif leaf_a.upstream_bifurcation is not None and leaf_b.upstream_bifurcation is None:
                    bif_consistency += 0.2
                cost[i, j] = (
                    dist
                    + 0.45 * length_diff
                    + branch_diff
                    + 0.18 * enh_diff
                    + 0.12 * time_diff
                    + 0.3 * anatomy_diff
                    + 0.3 * mask_support_diff
                    + 0.28 * mask_time_diff
                    + 0.3 * bif_sig_diff
                    + 0.12 * prior_upstream_mean
                    + 0.12 * prior_upstream_diff
                    + 0.35 * role_diff
                    + 0.2 * distal_diff
                    + bif_consistency
                )

        rows, cols = linear_sum_assignment(cost)
        pairs = []
        accepted_costs = []
        for r, c in zip(rows, cols):
            if cost[r, c] > 1.35:
                continue
            pairs.append((source.leaves[r], target.leaves[c]))
            accepted_costs.append(float(cost[r, c]))
        if not pairs:
            continue
        score = float(np.mean(accepted_costs))
        if best_meta is None or len(pairs) > best_meta["count"] or (
            len(pairs) == best_meta["count"] and score < best_meta["score"]
        ):
            best_pairs = pairs
            best_cost = cost.copy()
            best_meta = {
                "candidate_index": idx,
                "count": len(pairs),
                "score": score,
                "linear": linear.tolist(),
                "offset": offset.tolist(),
            }

    if best_meta is None:
        raise RuntimeError(f"failed to match leaves for {source.series} vs {target.series}")
    if best_cost is None:
        raise RuntimeError(f"failed to build leaf cost matrix for {source.series} vs {target.series}")
    return best_pairs, best_meta, best_cost


def build_proximal_segment_matches(
    source: SeriesView,
    target: SeriesView,
    bif_pairs: list[tuple[BifurcationNode, BifurcationNode]],
    max_bif_level: int = 3,
    max_segments: int = 6,
) -> list[ProximalSegmentMatch]:
    bif_map = {int(bif_a.node): bif_b for bif_a, bif_b in bif_pairs}
    selected = [
        bif_a
        for bif_a, _bif_b in bif_pairs
        if bif_a.bifurcation_level <= max_bif_level
    ]
    selected.sort(key=lambda item: (item.bifurcation_level, -item.subtree_leaves, item.path_length))
    selected_nodes = {int(item.node) for item in selected}

    matches: list[ProximalSegmentMatch] = []
    for bif_a in selected:
        bif_b = bif_map.get(int(bif_a.node))
        if bif_b is None:
            continue
        parent_source = source.root
        for ancestor in ancestor_chain(int(bif_a.node), source.parent_map)[1:]:
            if ancestor in selected_nodes:
                parent_source = int(ancestor)
                break
        if parent_source == source.root:
            parent_target = target.root
        else:
            mapped_parent = bif_map.get(parent_source)
            if mapped_parent is None:
                continue
            parent_target = int(mapped_parent.node)
        if not is_ancestor(parent_target, int(bif_b.node), target.parent_map):
            continue
        try:
            points_a = rooted_polyline_between(
                source.graph,
                source.node_pixels,
                source.parent_map,
                parent_source,
                int(bif_a.node),
            )
            points_b = rooted_polyline_between(
                target.graph,
                target.node_pixels,
                target.parent_map,
                parent_target,
                int(bif_b.node),
            )
        except ValueError:
            continue
        length_a = float(polyline_length(points_a))
        length_b = float(polyline_length(points_b))
        if max(length_a, length_b) < 24.0:
            continue
        matches.append(
            ProximalSegmentMatch(
                source_node=int(bif_a.node),
                target_node=int(bif_b.node),
                parent_source=parent_source,
                parent_target=parent_target,
                bifurcation_level=int(bif_a.bifurcation_level),
                points_a=points_a,
                points_b=points_b,
                length_a=length_a,
                length_b=length_b,
            )
        )

    matches.sort(key=lambda item: (item.bifurcation_level, -max(item.length_a, item.length_b)))
    return matches[:max_segments]


def pair_maps(pairs: list[tuple[LeafPath, LeafPath]]) -> tuple[dict[int, int], dict[int, int]]:
    forward: dict[int, int] = {}
    backward: dict[int, int] = {}
    for leaf_a, leaf_b in pairs:
        forward[int(leaf_a.leaf)] = int(leaf_b.leaf)
        backward[int(leaf_b.leaf)] = int(leaf_a.leaf)
    return forward, backward


def triplet_supported_pairs(
    series_a: str,
    series_b: str,
    pairs_ab: list[tuple[LeafPath, LeafPath]],
    matched_maps: dict[tuple[str, str], dict[int, int]],
) -> tuple[list[tuple[LeafPath, LeafPath]], dict[str, object]]:
    other_series = [series for series in LEFT_SERIES if series not in (series_a, series_b)]
    if not other_series:
        return pairs_ab, {
            "third_view": None,
            "triplet_support": len(pairs_ab),
            "support_ratio": 1.0,
            "supported_leaf_triplets": [],
        }

    series_c = other_series[0]
    map_ac = matched_maps.get((series_a, series_c), {})
    map_bc = matched_maps.get((series_b, series_c), {})
    supported: list[tuple[LeafPath, LeafPath]] = []
    supported_triplets: list[dict[str, int]] = []
    for leaf_a, leaf_b in pairs_ab:
        leaf_c_from_a = map_ac.get(int(leaf_a.leaf))
        leaf_c_from_b = map_bc.get(int(leaf_b.leaf))
        if leaf_c_from_a is None or leaf_c_from_b is None or leaf_c_from_a != leaf_c_from_b:
            continue
        supported.append((leaf_a, leaf_b))
        supported_triplets.append(
            {
                "leaf_a": int(leaf_a.leaf),
                "leaf_b": int(leaf_b.leaf),
                "leaf_c": int(leaf_c_from_a),
            }
        )
    return supported, {
        "third_view": series_c,
        "triplet_support": len(supported),
        "support_ratio": round(len(supported) / max(len(pairs_ab), 1), 4),
        "supported_leaf_triplets": supported_triplets,
    }


def greedy_multiview_triplets(
    view_a: SeriesView,
    view_b: SeriesView,
    view_c: SeriesView,
    cost_ab: np.ndarray,
    cost_ac: np.ndarray,
    cost_bc: np.ndarray,
    max_pair_cost: float = 1.5,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for i, leaf_a in enumerate(view_a.leaves):
        for j, leaf_b in enumerate(view_b.leaves):
            pair_ab = float(cost_ab[i, j])
            if pair_ab > max_pair_cost:
                continue
            for k, leaf_c in enumerate(view_c.leaves):
                pair_ac = float(cost_ac[i, k])
                pair_bc = float(cost_bc[j, k])
                if max(pair_ab, pair_ac, pair_bc) > max_pair_cost:
                    continue
                lengths = np.array([leaf_a.length, leaf_b.length, leaf_c.length], dtype=float)
                length_consistency = float(np.std(lengths) / max(np.mean(lengths), 1.0))
                score = (pair_ab + pair_ac + pair_bc) / 3.0 + 0.2 * length_consistency
                candidates.append(
                    {
                        "leaf_a": leaf_a,
                        "leaf_b": leaf_b,
                        "leaf_c": leaf_c,
                        "score": round(score, 4),
                        "pair_costs": {
                            "ab": round(pair_ab, 4),
                            "ac": round(pair_ac, 4),
                            "bc": round(pair_bc, 4),
                        },
                    }
                )

    candidates.sort(key=lambda item: item["score"])
    used_a: set[int] = set()
    used_b: set[int] = set()
    used_c: set[int] = set()
    selected: list[dict[str, object]] = []
    for item in candidates:
        leaf_a = item["leaf_a"]
        leaf_b = item["leaf_b"]
        leaf_c = item["leaf_c"]
        if leaf_a.leaf in used_a or leaf_b.leaf in used_b or leaf_c.leaf in used_c:
            continue
        used_a.add(int(leaf_a.leaf))
        used_b.add(int(leaf_b.leaf))
        used_c.add(int(leaf_c.leaf))
        selected.append(item)
    return selected


def evaluate_pair_geometry(
    pairs: list[tuple[LeafPath, LeafPath]],
    n_samples: int = 10,
) -> dict[str, object] | None:
    x1, x2, edge_meta = gather_correspondences(pairs, n_samples=n_samples)
    if len(x1) < 8:
        return None
    try:
        F, inliers = ransac_fundamental(x1, x2, iterations=3000, threshold=4.0)
    except RuntimeError:
        return None
    errors = sampson_distance(F, x1, x2)
    return {
        "F": F,
        "x1": x1,
        "x2": x2,
        "edge_meta": edge_meta,
        "correspondences": int(len(x1)),
        "inliers": int(inliers.sum()),
        "inlier_ratio": round(float(inliers.mean()), 4),
        "median_sampson": round(float(np.median(errors)), 4),
    }


def gather_proximal_correspondences(
    segments: list[ProximalSegmentMatch],
    n_samples: int = 12,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    src_points: list[np.ndarray] = []
    tgt_points: list[np.ndarray] = []
    edge_meta: list[dict[str, object]] = []
    for segment in segments:
        samples_a = resample_curve(segment.points_a, n_samples)
        samples_b = resample_curve(segment.points_b, n_samples)
        src_points.append(samples_a)
        tgt_points.append(samples_b)
        edge_meta.append(
            {
                "source_node": int(segment.source_node),
                "target_node": int(segment.target_node),
                "parent_source": int(segment.parent_source),
                "parent_target": int(segment.parent_target),
                "bifurcation_level": int(segment.bifurcation_level),
                "length_a": round(segment.length_a, 3),
                "length_b": round(segment.length_b, 3),
            }
        )
    x1 = np.concatenate(src_points, axis=0)
    x2 = np.concatenate(tgt_points, axis=0)

    seen: set[tuple[int, int, int, int]] = set()
    keep_idx: list[int] = []
    for idx, (p1, p2) in enumerate(zip(x1, x2)):
        key = (int(round(p1[0])), int(round(p1[1])), int(round(p2[0])), int(round(p2[1])))
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(idx)
    return x1[keep_idx], x2[keep_idx], edge_meta


def evaluate_proximal_geometry(
    segments: list[ProximalSegmentMatch],
    n_samples: int = 12,
) -> dict[str, object] | None:
    x1, x2, edge_meta = gather_proximal_correspondences(segments, n_samples=n_samples)
    if len(x1) < 8:
        return None
    try:
        F, inliers = ransac_fundamental(x1, x2, iterations=3000, threshold=4.0)
    except RuntimeError:
        return None
    errors = sampson_distance(F, x1, x2)
    return {
        "F": F,
        "x1": x1,
        "x2": x2,
        "edge_meta": edge_meta,
        "correspondences": int(len(x1)),
        "inliers": int(inliers.sum()),
        "inlier_ratio": round(float(inliers.mean()), 4),
        "median_sampson": round(float(np.median(errors)), 4),
    }


def gather_correspondences(
    pairs: list[tuple[LeafPath, LeafPath]],
    n_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    src_points: list[np.ndarray] = []
    tgt_points: list[np.ndarray] = []
    edge_meta: list[dict[str, object]] = []
    for leaf_a, leaf_b in pairs:
        samples_a = resample_curve(leaf_a.points, n_samples)
        samples_b = resample_curve(leaf_b.points, n_samples)
        src_points.append(samples_a[1:])  # drop root sample duplicated across paths
        tgt_points.append(samples_b[1:])
        edge_meta.append(
            {
                "leaf_a": leaf_a.leaf,
                "leaf_b": leaf_b.leaf,
                "length_a": round(leaf_a.length, 3),
                "length_b": round(leaf_b.length, 3),
            }
        )
    x1 = np.concatenate(src_points, axis=0)
    x2 = np.concatenate(tgt_points, axis=0)

    seen: set[tuple[int, int, int, int]] = set()
    keep_idx: list[int] = []
    for idx, (p1, p2) in enumerate(zip(x1, x2)):
        key = (int(round(p1[0])), int(round(p1[1])), int(round(p2[0])), int(round(p2[1])))
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(idx)
    return x1[keep_idx], x2[keep_idx], edge_meta


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    rms = np.sqrt(np.mean(np.sum(shifted**2, axis=1)))
    scale = math.sqrt(2.0) / max(rms, 1e-8)
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    homog = np.column_stack([points, np.ones(len(points))])
    norm = (T @ homog.T).T
    return norm[:, :2], T


def estimate_fundamental_8point(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    norm1, T1 = normalize_points(x1)
    norm2, T2 = normalize_points(x2)
    A = np.column_stack(
        [
            norm2[:, 0] * norm1[:, 0],
            norm2[:, 0] * norm1[:, 1],
            norm2[:, 0],
            norm2[:, 1] * norm1[:, 0],
            norm2[:, 1] * norm1[:, 1],
            norm2[:, 1],
            norm1[:, 0],
            norm1[:, 1],
            np.ones(len(norm1)),
        ]
    )
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    F = vt[-1].reshape(3, 3)
    u, s, vt = np.linalg.svd(F)
    s[-1] = 0.0
    F = u @ np.diag(s) @ vt
    F = T2.T @ F @ T1
    return F / max(np.linalg.norm(F), 1e-8)


def sampson_distance(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    p1 = np.column_stack([x1, np.ones(len(x1))])
    p2 = np.column_stack([x2, np.ones(len(x2))])
    Fx1 = (F @ p1.T).T
    Ftx2 = (F.T @ p2.T).T
    numer = np.sum(p2 * Fx1, axis=1) ** 2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return numer / np.maximum(denom, 1e-8)


def ransac_fundamental(
    x1: np.ndarray,
    x2: np.ndarray,
    iterations: int = 2500,
    threshold: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    best_inliers: np.ndarray | None = None
    best_F: np.ndarray | None = None
    for _ in range(iterations):
        idx = rng.choice(len(x1), size=8, replace=False)
        try:
            F = estimate_fundamental_8point(x1[idx], x2[idx])
        except np.linalg.LinAlgError:
            continue
        errors = sampson_distance(F, x1, x2)
        inliers = errors < threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_F = F
    if best_inliers is None or best_F is None or best_inliers.sum() < 8:
        raise RuntimeError("failed to estimate a stable fundamental matrix")
    F = estimate_fundamental_8point(x1[best_inliers], x2[best_inliers])
    return F, best_inliers


def camera_from_fundamental(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, _, vt = np.linalg.svd(F.T)
    e2 = vt[-1]
    e2 = e2 / max(e2[-1], 1e-8)
    ex = np.array(
        [
            [0.0, -e2[2], e2[1]],
            [e2[2], 0.0, -e2[0]],
            [-e2[1], e2[0], 0.0],
        ],
        dtype=float,
    )
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([ex @ F, e2.reshape(3, 1)])
    return P1, P2


def triangulate_points(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    points = []
    for p1, p2 in zip(x1, x2):
        A = np.vstack(
            [
                p1[0] * P1[2] - P1[0],
                p1[1] * P1[2] - P1[1],
                p2[0] * P2[2] - P2[0],
                p2[1] * P2[2] - P2[1],
            ]
        )
        _, _, vt = np.linalg.svd(A)
        X = vt[-1]
        X = X / max(X[-1], 1e-8)
        points.append(X[:3])
    return np.array(points, dtype=float)


def render_correspondence_preview(
    view_a: SeriesView,
    view_b: SeriesView,
    pairs: list[tuple[LeafPath, LeafPath]],
    out_path: Path,
) -> None:
    mask_a = (view_a.mask.astype(np.uint8) * 255)
    mask_b = (view_b.mask.astype(np.uint8) * 255)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mask_a, cmap="gray")
    axes[1].imshow(mask_b, cmap="gray")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pairs), 1)))
    for i, (leaf_a, leaf_b) in enumerate(pairs):
        color = colors[i % len(colors)]
        axes[0].plot(leaf_a.points[:, 0], leaf_a.points[:, 1], color=color, linewidth=1.5)
        axes[1].plot(leaf_b.points[:, 0], leaf_b.points[:, 1], color=color, linewidth=1.5)
        axes[0].scatter([leaf_a.endpoint[0]], [leaf_a.endpoint[1]], color=color, s=18)
        axes[1].scatter([leaf_b.endpoint[0]], [leaf_b.endpoint[1]], color=color, s=18)
    axes[0].set_title(view_a.series)
    axes[1].set_title(view_b.series)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_proximal_correspondence_preview(
    view_a: SeriesView,
    view_b: SeriesView,
    segments: list[ProximalSegmentMatch],
    out_path: Path,
) -> None:
    mask_a = (view_a.mask.astype(np.uint8) * 255)
    mask_b = (view_b.mask.astype(np.uint8) * 255)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mask_a, cmap="gray")
    axes[1].imshow(mask_b, cmap="gray")
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(segments), 1)))
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        axes[0].plot(segment.points_a[:, 0], segment.points_a[:, 1], color=color, linewidth=2.0)
        axes[1].plot(segment.points_b[:, 0], segment.points_b[:, 1], color=color, linewidth=2.0)
        axes[0].scatter([segment.points_a[-1, 0]], [segment.points_a[-1, 1]], color=color, s=24)
        axes[1].scatter([segment.points_b[-1, 0]], [segment.points_b[-1, 1]], color=color, s=24)
    axes[0].set_title(f"{view_a.series} proximal")
    axes[1].set_title(f"{view_b.series} proximal")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_3d_preview(edges: list[EdgeModel], out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    all_points = []
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(edges), 1)))
    for i, edge in enumerate(edges):
        pts = edge.points3d
        all_points.append(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[i % len(colors)], linewidth=1.4)
    pts = np.concatenate(all_points, axis=0)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    radius = max((pts.max(axis=0) - pts.min(axis=0))) * 0.55
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=24, azim=-61)
    ax.set_axis_off()
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_projective_edges(
    F: np.ndarray,
    pairs: list[tuple[LeafPath, LeafPath]],
) -> list[EdgeModel]:
    P1, P2 = camera_from_fundamental(F)
    raw_polylines: list[np.ndarray] = []
    for idx, (leaf_a, leaf_b) in enumerate(pairs):
        samples_a = resample_curve(leaf_a.points, 24)
        samples_b = resample_curve(leaf_b.points, 24)
        X = triangulate_points(P1, P2, samples_a, samples_b)
        raw_polylines.append(X)

    all_points = np.concatenate(raw_polylines, axis=0)
    center = np.median(all_points, axis=0, keepdims=True)
    centered = all_points - center
    scale = max(np.percentile(np.linalg.norm(centered[:, :2], axis=1), 90), 1.0)

    edges: list[EdgeModel] = []
    for idx, ((leaf_a, _leaf_b), X) in enumerate(zip(pairs, raw_polylines)):
        samples_a = resample_curve(leaf_a.points, 24)
        X = (X - center) / scale * 180.0
        radii = np.full(len(X), 1.6, dtype=float)
        edges.append(
            EdgeModel(
                source=idx,
                target=idx + 1,
                points2d=samples_a.copy(),
                points3d=X,
                radii=radii,
            )
        )
    return edges


def build_projective_edges_from_segments(
    F: np.ndarray,
    segments: list[ProximalSegmentMatch],
) -> list[EdgeModel]:
    P1, P2 = camera_from_fundamental(F)
    raw_polylines: list[np.ndarray] = []
    samples_cache: list[np.ndarray] = []
    for segment in segments:
        samples_a = resample_curve(segment.points_a, 28)
        samples_b = resample_curve(segment.points_b, 28)
        X = triangulate_points(P1, P2, samples_a, samples_b)
        raw_polylines.append(X)
        samples_cache.append(samples_a)

    all_points = np.concatenate(raw_polylines, axis=0)
    center = np.median(all_points, axis=0, keepdims=True)
    centered = all_points - center
    scale = max(np.percentile(np.linalg.norm(centered[:, :2], axis=1), 90), 1.0)

    edges: list[EdgeModel] = []
    for idx, (segment, X, samples_a) in enumerate(zip(segments, raw_polylines, samples_cache)):
        X = (X - center) / scale * 180.0
        radii = np.full(len(X), 1.9 - 0.18 * min(segment.bifurcation_level, 3), dtype=float)
        edges.append(
            EdgeModel(
                source=segment.parent_source,
                target=segment.source_node,
                points2d=samples_a.copy(),
                points3d=X,
                radii=radii,
            )
        )
    return edges


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / "outputs_p0001_projective"
    outputs_dir.mkdir(exist_ok=True)

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

    pair_cache: dict[tuple[str, str], dict[str, object]] = {}
    matched_maps: dict[tuple[str, str], dict[int, int]] = {}
    pair_costs: dict[tuple[str, str], np.ndarray] = {}

    for series_a, series_b in itertools.combinations(LEFT_SERIES, 2):
        view_a = views[series_a]
        view_b = views[series_b]
        bif_pairs, bif_meta, _bif_cost_matrix = match_bifurcations(view_a, view_b)
        bif_map = {int(bif_a.node): int(bif_b.node) for bif_a, bif_b in bif_pairs}
        proximal_segments = build_proximal_segment_matches(view_a, view_b, bif_pairs)
        pairs, match_meta, cost_matrix = match_leaf_paths(view_a, view_b, bifurcation_map=bif_map)
        pair_cache[(series_a, series_b)] = {
            "pairs": pairs,
            "match_meta": match_meta,
            "bif_pairs": bif_pairs,
            "bif_meta": bif_meta,
            "proximal_segments": proximal_segments,
        }
        forward, backward = pair_maps(pairs)
        matched_maps[(series_a, series_b)] = forward
        matched_maps[(series_b, series_a)] = backward
        pair_costs[(series_a, series_b)] = cost_matrix
        pair_costs[(series_b, series_a)] = cost_matrix.T

    pair_reports = []
    best_pair: tuple[str, str] | None = None
    best_payload: dict[str, object] | None = None

    for series_a, series_b in itertools.combinations(LEFT_SERIES, 2):
        payload = pair_cache[(series_a, series_b)]
        pairs = payload["pairs"]
        match_meta = payload["match_meta"]
        bif_pairs = payload["bif_pairs"]
        bif_meta = payload["bif_meta"]
        proximal_segments = payload["proximal_segments"]
        cycle_pairs, cycle_meta = triplet_supported_pairs(series_a, series_b, pairs, matched_maps)
        series_c = cycle_meta["third_view"]
        triplets = greedy_multiview_triplets(
            views[series_a],
            views[series_b],
            views[series_c],
            pair_costs[(series_a, series_b)],
            pair_costs[(series_a, series_c)],
            pair_costs[(series_b, series_c)],
        )
        triplet_pairs = [(item["leaf_a"], item["leaf_b"]) for item in triplets]

        evaluations: list[dict[str, object]] = []
        proximal_eval = evaluate_proximal_geometry(proximal_segments, n_samples=12)
        if proximal_eval is not None:
            evaluations.append(
                {
                    "mode": "proximal_segments",
                    "pairs": [],
                    "segments": proximal_segments,
                    "unit_count": len(proximal_segments),
                    **proximal_eval,
                }
            )
        default_eval = evaluate_pair_geometry(pairs, n_samples=10)
        if default_eval is not None:
            evaluations.append(
                {
                    "mode": "pairwise_all",
                    "pairs": pairs,
                    "segments": [],
                    "unit_count": len(pairs),
                    **default_eval,
                }
            )
        if len(triplet_pairs) >= 2:
            multiview_eval = evaluate_pair_geometry(triplet_pairs, n_samples=10)
            if multiview_eval is not None:
                evaluations.append(
                    {
                        "mode": "multiview_triplets",
                        "pairs": triplet_pairs,
                        "segments": [],
                        "unit_count": len(triplet_pairs),
                        **multiview_eval,
                    }
                )
        if not evaluations:
            continue

        chosen = max(
            evaluations,
            key=lambda item: (
                1 if item["mode"] == "proximal_segments" else 0,
                item["inlier_ratio"],
                -item["median_sampson"],
                item["inliers"],
                item["unit_count"],
            ),
        )
        report = {
            "pair": [series_a, series_b],
            "matched_leaf_paths": len(pairs),
            "used_leaf_paths": len(chosen["pairs"]),
            "used_proximal_segments": len(chosen["segments"]),
            "correspondences": chosen["correspondences"],
            "inliers": chosen["inliers"],
            "inlier_ratio": chosen["inlier_ratio"],
            "median_sampson": chosen["median_sampson"],
            "reconstruction_mode": chosen["mode"],
            "matching_meta": match_meta,
            "matched_bifurcations": len(bif_pairs),
            "available_proximal_segments": len(proximal_segments),
            "bifurcation_meta": bif_meta,
            "bifurcation_pairs": [
                {
                    "node_a": int(bif_a.node),
                    "node_b": int(bif_b.node),
                    "role_a": bif_a.role_label,
                    "role_b": bif_b.role_label,
                    "subtree_leaves_a": int(bif_a.subtree_leaves),
                    "subtree_leaves_b": int(bif_b.subtree_leaves),
                }
                for bif_a, bif_b in bif_pairs
            ],
            "proximal_segments": [
                {
                    "source_node": int(segment.source_node),
                    "target_node": int(segment.target_node),
                    "parent_source": int(segment.parent_source),
                    "parent_target": int(segment.parent_target),
                    "bifurcation_level": int(segment.bifurcation_level),
                    "length_a": round(segment.length_a, 3),
                    "length_b": round(segment.length_b, 3),
                }
                for segment in proximal_segments
            ],
            "third_view": cycle_meta["third_view"],
            "cycle_triplet_support": cycle_meta["triplet_support"],
            "cycle_support_ratio": cycle_meta["support_ratio"],
            "cycle_supported_leaf_triplets": cycle_meta["supported_leaf_triplets"],
            "multiview_triplet_support": len(triplets),
            "multiview_triplets": [
                {
                    "leaf_a": int(item["leaf_a"].leaf),
                    "leaf_b": int(item["leaf_b"].leaf),
                    "leaf_c": int(item["leaf_c"].leaf),
                    "score": item["score"],
                    "pair_costs": item["pair_costs"],
                }
                for item in triplets
            ],
            "edge_pairs": chosen["edge_meta"],
        }
        pair_reports.append(report)
        score = (
            1 if report["reconstruction_mode"] == "proximal_segments" else 0,
            report["available_proximal_segments"],
            report["multiview_triplet_support"],
            report["used_proximal_segments"],
            report["used_leaf_paths"],
            -report["median_sampson"],
            report["inlier_ratio"],
            report["inliers"],
        )
        if best_payload is None or score > (
            1 if best_payload["reconstruction_mode"] == "proximal_segments" else 0,
            best_payload["available_proximal_segments"],
            best_payload["multiview_triplet_support"],
            best_payload["used_proximal_segments"],
            best_payload["used_leaf_paths"],
            -best_payload["median_sampson"],
            best_payload["inlier_ratio"],
            best_payload["inliers"],
        ):
            best_pair = (series_a, series_b)
            best_payload = {
                **report,
                "F": chosen["F"],
                "pairs": chosen["pairs"],
                "segments": chosen["segments"],
            }

    if best_pair is None or best_payload is None:
        raise RuntimeError("No viable pairwise projective reconstruction found")

    pair_reports_path = outputs_dir / "pairwise_report.json"
    pair_reports_path.write_text(json.dumps(pair_reports, ensure_ascii=False, indent=2), encoding="utf-8")

    series_a, series_b = best_pair
    view_a = views[series_a]
    view_b = views[series_b]
    match_preview_path = outputs_dir / f"best_pair_{series_a}_{series_b}_matches.png"
    if best_payload["reconstruction_mode"] == "proximal_segments":
        render_proximal_correspondence_preview(
            view_a,
            view_b,
            best_payload["segments"],
            match_preview_path,
        )
        projective_edges = build_projective_edges_from_segments(best_payload["F"], best_payload["segments"])
    else:
        render_correspondence_preview(
            view_a,
            view_b,
            best_payload["pairs"],
            match_preview_path,
        )
        projective_edges = build_projective_edges(best_payload["F"], best_payload["pairs"])
    preview_path = outputs_dir / f"best_pair_{series_a}_{series_b}_projective_preview.png"
    centerlines_path = outputs_dir / f"best_pair_{series_a}_{series_b}_projective_centerlines.obj"
    mesh_path = outputs_dir / f"best_pair_{series_a}_{series_b}_projective.obj"
    render_3d_preview(projective_edges, preview_path, f"p0001 {series_a}-{series_b} projective recon")
    write_centerlines_obj(centerlines_path, projective_edges)
    write_obj_mesh(mesh_path, projective_edges)

    summary = {
        "candidate_pairs": pair_reports,
        "best_pair": {
            "series": [series_a, series_b],
            "matched_leaf_paths": best_payload["matched_leaf_paths"],
            "used_leaf_paths": best_payload["used_leaf_paths"],
            "used_proximal_segments": best_payload["used_proximal_segments"],
            "matched_bifurcations": best_payload["matched_bifurcations"],
            "available_proximal_segments": best_payload["available_proximal_segments"],
            "correspondences": best_payload["correspondences"],
            "inliers": best_payload["inliers"],
            "inlier_ratio": best_payload["inlier_ratio"],
            "median_sampson": best_payload["median_sampson"],
            "reconstruction_mode": best_payload["reconstruction_mode"],
            "third_view": best_payload["third_view"],
            "cycle_triplet_support": best_payload["cycle_triplet_support"],
            "cycle_support_ratio": best_payload["cycle_support_ratio"],
            "multiview_triplet_support": best_payload["multiview_triplet_support"],
            "match_preview": str(match_preview_path),
            "projective_preview": str(preview_path),
            "projective_centerlines": str(centerlines_path),
            "projective_mesh": str(mesh_path),
        },
        "cohort_prior": {
            "used": cohort_prior is not None,
            "num_reference_series": int(cohort_prior["num_series"]) if cohort_prior is not None else 0,
            "levels": sorted(int(level) for level in cohort_prior["by_level"].keys()) if cohort_prior is not None else [],
        },
        "note": (
            "This is an uncalibrated projective reconstruction from two matched left-coronary views. "
            "It is closer to classical two-view geometry than the previous heuristic depth model, "
            "but still not metric 3D without C-arm calibration."
        ),
    }
    (outputs_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
