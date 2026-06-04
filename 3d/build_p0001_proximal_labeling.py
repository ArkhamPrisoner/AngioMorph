#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_p0001_projective_recon import (
    LEFT_SERIES,
    BifurcationNode,
    SeriesView,
    build_left_cohort_prior,
    direction_unit,
    extract_view,
    pixel_to_xy,
    rooted_polyline_between,
    sample_map_mean,
)
from reconstruct_coronary_tree import polyline_length


@dataclass
class AxisCandidate:
    child: int
    terminal_leaf: int
    polyline: np.ndarray
    length: float
    subtree_leaves: int
    subtree_bifurcations: int
    continuation: float
    local_time: float
    terminal_time: float
    support_mean: float
    monotonicity: float
    prior_alignment: float
    lad_score: float
    role: str


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.array(values, dtype=float)
    if np.allclose(arr.max(), arr.min()):
        return [0.5 for _ in values]
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr.tolist()


def descendants(root: int, children: dict[int, list[int]]) -> list[int]:
    stack = [root]
    out: list[int] = []
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(children.get(cur, []))
    return out


def subtree_leaf_count(node: int, view: SeriesView) -> int:
    nodes = descendants(node, view.children)
    return sum(1 for item in nodes if item != view.root and view.graph.degree[item] == 1)


def subtree_bifurcation_count(node: int, view: SeriesView) -> int:
    nodes = descendants(node, view.children)
    bif_nodes = {item.node for item in view.bifurcations}
    return sum(1 for item in nodes if item in bif_nodes)


def terminal_leaf_for_child(node: int, view: SeriesView) -> int:
    nodes = descendants(node, view.children)
    leaf_nodes = [item for item in nodes if item != view.root and view.graph.degree[item] == 1]
    if not leaf_nodes:
        return node
    return max(leaf_nodes, key=lambda item: view.path_lengths.get(item, 0.0))


def sample_curve_map(image: np.ndarray, points: np.ndarray, n_samples: int = 24) -> np.ndarray:
    if len(points) == 1:
        samples = np.repeat(points, n_samples, axis=0)
    else:
        d = np.linspace(0.0, 1.0, n_samples)
        base = np.linspace(0.0, 1.0, len(points))
        samples = np.column_stack([np.interp(d, base, points[:, i]) for i in range(points.shape[1])])
    rr = np.clip(np.round(samples[:, 1]).astype(int), 0, image.shape[0] - 1)
    cc = np.clip(np.round(samples[:, 0]).astype(int), 0, image.shape[1] - 1)
    return image[rr, cc].astype(float)


def monotonicity_score(profile: np.ndarray, eps: float = 0.03) -> float:
    if len(profile) < 2:
        return 0.0
    diffs = np.diff(profile)
    return float(np.mean(diffs >= -eps))


def choose_main_split(view: SeriesView) -> tuple[BifurcationNode | None, float]:
    candidates = [item for item in view.bifurcations if item.bifurcation_level == 1]
    if not candidates:
        candidates = view.bifurcations[:]
    if not candidates:
        return None, 0.0

    scores = []
    for bif in candidates:
        score = (
            1.2 * bif.subtree_leaves
            - 0.4 * bif.path_length
            - 4.0 * bif.prior_level_score
            - 2.5 * bif.prior_rank_score
            + (3.0 if bif.role_label == "main_split" else 0.0)
        )
        scores.append((score, bif))
    scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scores[0]
    margin = best_score - scores[1][0] if len(scores) > 1 else best_score
    confidence = 1.0 / (1.0 + np.exp(-0.08 * margin))
    return best, float(confidence)


def parent_direction(view: SeriesView, node: int) -> np.ndarray:
    parent = view.parent_map.get(node)
    if parent is None:
        return np.array([1.0, 0.0], dtype=float)
    segment = rooted_polyline_between(view.graph, view.node_pixels, view.parent_map, parent, node)
    return direction_unit(segment, at_start=False)


def axis_candidates(view: SeriesView, main_split: BifurcationNode) -> list[AxisCandidate]:
    child_nodes = view.children.get(main_split.node, [])
    if not child_nodes:
        return []
    parent_dir = parent_direction(view, main_split.node)
    candidates: list[AxisCandidate] = []
    for child in child_nodes:
        terminal = terminal_leaf_for_child(child, view)
        polyline = rooted_polyline_between(view.graph, view.node_pixels, view.parent_map, main_split.node, terminal)
        if polyline_length(polyline) < 12.0:
            continue
        child_dir = direction_unit(polyline, at_start=True)
        continuation = float(np.dot(parent_dir, child_dir))
        time_profile = sample_curve_map(view.mask_time_map, polyline, n_samples=24)
        support_profile = sample_curve_map(view.mask_support_map, polyline, n_samples=24)
        length = float(polyline_length(polyline))
        local_time = float(np.mean(time_profile[:6]))
        terminal_time = float(np.mean(time_profile[-6:]))
        support_mean = float(np.mean(support_profile))
        candidates.append(
            AxisCandidate(
                child=child,
                terminal_leaf=terminal,
                polyline=polyline,
                length=length,
                subtree_leaves=subtree_leaf_count(child, view),
                subtree_bifurcations=subtree_bifurcation_count(child, view),
                continuation=continuation,
                local_time=local_time,
                terminal_time=terminal_time,
                support_mean=support_mean,
                monotonicity=monotonicity_score(time_profile),
                prior_alignment=0.0,
                lad_score=0.0,
                role="sidebranch_like",
            )
        )
    if not candidates:
        return []

    lengths = normalize([item.length for item in candidates])
    leaves = normalize([item.subtree_leaves for item in candidates])
    continuations = normalize([item.continuation for item in candidates])
    times = normalize([-item.local_time for item in candidates])
    supports = normalize([item.support_mean for item in candidates])
    for idx, item in enumerate(candidates):
        item.lad_score = (
            0.34 * lengths[idx]
            + 0.22 * leaves[idx]
            + 0.22 * continuations[idx]
            + 0.12 * times[idx]
            + 0.10 * supports[idx]
        )
    candidates.sort(key=lambda item: item.lad_score, reverse=True)
    if candidates:
        candidates[0].role = "lad_axis"
    if len(candidates) > 1:
        candidates[1].role = "lcx_axis"
    for item in candidates[2:]:
        item.role = "sidebranch_like"
    return candidates


def temporal_axis_consistency(axis: AxisCandidate) -> float:
    direction_score = max(axis.terminal_time - axis.local_time, 0.0)
    return float(0.65 * axis.monotonicity + 0.35 * min(direction_score * 2.0, 1.0))


def label_view(view: SeriesView) -> dict[str, object]:
    main_split, main_conf = choose_main_split(view)
    if main_split is None:
        return {
            "series": view.series,
            "status": "failed",
            "reason": "no_bifurcations",
        }

    main_path = rooted_polyline_between(view.graph, view.node_pixels, view.parent_map, view.root, main_split.node)
    axes = axis_candidates(view, main_split)
    lad = next((item for item in axes if item.role == "lad_axis"), None)
    lcx = next((item for item in axes if item.role == "lcx_axis"), None)
    if lad is None:
        return {
            "series": view.series,
            "status": "failed",
            "reason": "no_lad_axis_candidate",
        }

    lad_cons = temporal_axis_consistency(lad)
    lcx_cons = temporal_axis_consistency(lcx) if lcx is not None else 0.0
    score_gap = lad.lad_score - (lcx.lad_score if lcx is not None else 0.0)
    axis_conf = float(1.0 / (1.0 + np.exp(-6.0 * score_gap)))
    overall_conf = float(np.clip(0.45 * main_conf + 0.35 * lad_cons + 0.20 * axis_conf, 0.0, 1.0))

    return {
        "series": view.series,
        "status": "ok",
        "root_node": int(view.root),
        "root_xy": [round(float(x), 3) for x in view.root_xy],
        "main_split": {
            "node": int(main_split.node),
            "xy": [round(float(x), 3) for x in main_split.point],
            "confidence": round(main_conf, 4),
            "subtree_leaves": int(main_split.subtree_leaves),
            "prior_level_score": round(float(main_split.prior_level_score), 4),
            "prior_rank_score": round(float(main_split.prior_rank_score), 4),
        },
        "main_trunk": {
            "length": round(float(polyline_length(main_path)), 3),
            "temporal_support_mean": round(float(np.mean(sample_curve_map(view.mask_support_map, main_path, 18))), 4),
            "temporal_time_mean": round(float(np.mean(sample_curve_map(view.mask_time_map, main_path, 18))), 4),
        },
        "lad_axis": {
            "child": int(lad.child),
            "terminal_leaf": int(lad.terminal_leaf),
            "length": round(lad.length, 3),
            "subtree_leaves": int(lad.subtree_leaves),
            "subtree_bifurcations": int(lad.subtree_bifurcations),
            "local_time": round(lad.local_time, 4),
            "terminal_time": round(lad.terminal_time, 4),
            "support_mean": round(lad.support_mean, 4),
            "monotonicity": round(lad.monotonicity, 4),
            "temporal_consistency": round(lad_cons, 4),
            "lad_score": round(lad.lad_score, 4),
        },
        "lcx_axis": (
            {
                "child": int(lcx.child),
                "terminal_leaf": int(lcx.terminal_leaf),
                "length": round(lcx.length, 3),
                "subtree_leaves": int(lcx.subtree_leaves),
                "subtree_bifurcations": int(lcx.subtree_bifurcations),
                "local_time": round(lcx.local_time, 4),
                "terminal_time": round(lcx.terminal_time, 4),
                "support_mean": round(lcx.support_mean, 4),
                "monotonicity": round(lcx.monotonicity, 4),
                "temporal_consistency": round(lcx_cons, 4),
                "lad_score": round(lcx.lad_score, 4),
            }
            if lcx is not None
            else None
        ),
        "sidebranches": [
            {
                "child": int(item.child),
                "terminal_leaf": int(item.terminal_leaf),
                "length": round(item.length, 3),
                "subtree_leaves": int(item.subtree_leaves),
                "monotonicity": round(item.monotonicity, 4),
            }
            for item in axes
            if item.role == "sidebranch_like"
        ],
        "axis_confidence": round(axis_conf, 4),
        "overall_confidence": round(overall_conf, 4),
    }


def render_labeling(view: SeriesView, payload: dict[str, object], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(view.mask.astype(np.uint8) * 255, cmap="gray")
    root = np.array(payload["root_xy"], dtype=float)
    main_split = np.array(payload["main_split"]["xy"], dtype=float)
    trunk = rooted_polyline_between(view.graph, view.node_pixels, view.parent_map, view.root, int(payload["main_split"]["node"]))
    ax.plot(trunk[:, 0], trunk[:, 1], color="#f5f1e8", linewidth=3.0)
    ax.scatter([root[0]], [root[1]], color="#ffffff", s=36)
    ax.scatter([main_split[0]], [main_split[1]], color="#ffcc33", s=42)

    colors = {
        "lad_axis": "#e63946",
        "lcx_axis": "#00b4d8",
        "sidebranch_like": "#8d99ae",
    }
    for role_key in ("lad_axis", "lcx_axis"):
        axis_payload = payload.get(role_key)
        if not axis_payload:
            continue
        polyline = rooted_polyline_between(
            view.graph,
            view.node_pixels,
            view.parent_map,
            int(payload["main_split"]["node"]),
            int(axis_payload["terminal_leaf"]),
        )
        ax.plot(polyline[:, 0], polyline[:, 1], color=colors[role_key], linewidth=2.6)
        ax.scatter([polyline[-1, 0]], [polyline[-1, 1]], color=colors[role_key], s=24)

    for branch in payload.get("sidebranches", []):
        polyline = rooted_polyline_between(
            view.graph,
            view.node_pixels,
            view.parent_map,
            int(payload["main_split"]["node"]),
            int(branch["terminal_leaf"]),
        )
        ax.plot(polyline[:, 0], polyline[:, 1], color=colors["sidebranch_like"], linewidth=1.4, alpha=0.65)

    ax.set_title(f"{view.series} proximal labeling")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    temporal_dir = base_dir / "outputs_p0001_temporal"
    output_dir = base_dir / "outputs_p0001_proximal_labeling"
    output_dir.mkdir(exist_ok=True)
    cohort_prior = build_left_cohort_prior(base_dir, exclude_patient="p0001")

    summaries = []
    for series in LEFT_SERIES:
        view = extract_view(
            series,
            temporal_dir / f"p0001_{series}_temporal_refined_mask.png",
            enhancement_path=temporal_dir / f"p0001_{series}_temporal_max_enhancement.png",
            peak_time_path=temporal_dir / f"p0001_{series}_temporal_peak_time.png",
            mask_support_path=temporal_dir / f"p0001_{series}_mask_temporal_support.png",
            mask_time_path=temporal_dir / f"p0001_{series}_mask_time_centroid_gray.png",
            cohort_prior=cohort_prior,
        )
        payload = label_view(view)
        json_path = output_dir / f"p0001_{series}_proximal_labeling.json"
        preview_path = output_dir / f"p0001_{series}_proximal_labeling.png"
        payload["preview"] = str(preview_path)
        payload["json"] = str(json_path)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if payload["status"] == "ok":
            render_labeling(view, payload, preview_path)
        summaries.append(payload)

    summary = {
        "series": summaries,
        "cohort_prior": {
            "used": cohort_prior is not None,
            "num_reference_series": int(cohort_prior["num_series"]) if cohort_prior is not None else 0,
        },
        "note": (
            "This step does not reconstruct 3D. It stabilizes per-series 2D proximal labeling "
            "for main trunk, main split, LAD-like axis and LCx-like axis using temporal and cohort priors."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
