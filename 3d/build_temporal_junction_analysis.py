#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_coronary_graph import analyze_graph  # noqa: E402


TRACKED_TYPES = {
    "bifurcation": "bifurcation",
    "crossing_or_overlap": "crossing_or_overlap",
    "short_branch_artifact": "artifact",
    "uncertain_bifurcation": "uncertain",
    "uncertain_high_order_branch": "uncertain",
    "uncertain_complex_junction": "uncertain",
}


@dataclass
class Observation:
    name: str
    frame_id: str
    frame_num: int
    node_id: int
    pixel_r: float
    pixel_c: float
    node_type: str
    broad_type: str
    confidence: float


@dataclass
class Track:
    track_id: int
    broad_type: str
    observations: list[Observation] = field(default_factory=list)

    @property
    def last(self) -> Observation:
        return self.observations[-1]

    @property
    def center(self) -> tuple[float, float]:
        rows = np.array([obs.pixel_r for obs in self.observations], dtype=float)
        cols = np.array([obs.pixel_c for obs in self.observations], dtype=float)
        return float(rows.mean()), float(cols.mean())


def frame_number(frame_id: str) -> int:
    if frame_id.startswith("f"):
        return int(frame_id[1:])
    return int(frame_id)


def iter_pairs(image_dir: Path, mask_dir: Path, pattern: str) -> list[tuple[Path, Path, str]]:
    pairs = []
    for image_path in sorted(image_dir.glob(pattern)):
        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            pairs.append((image_path, mask_path, image_path.stem))
    return pairs


def group_key(name: str) -> tuple[str, str, str]:
    parts = name.split("_")
    if len(parts) >= 4:
        return parts[0], parts[1], parts[3]
    return "unknown", "unknown", "unknown"


def extract_observations(result: dict[str, object]) -> list[Observation]:
    summary = result["summary"]
    image_meta = summary.get("image_meta", {})
    name = str(summary["name"])
    frame_id = str(image_meta.get("frame_id") or name.split("_")[2])
    out = []
    for node in result["nodes"]:
        cls = node.get("classification")
        if not cls:
            continue
        node_type = str(cls.get("type", "unknown"))
        broad_type = TRACKED_TYPES.get(node_type)
        if broad_type is None:
            continue
        r, c = node["pixel_rc"]
        out.append(
            Observation(
                name=name,
                frame_id=frame_id,
                frame_num=frame_number(frame_id),
                node_id=int(node["id"]),
                pixel_r=float(r),
                pixel_c=float(c),
                node_type=node_type,
                broad_type=broad_type,
                confidence=float(cls.get("confidence", 0.0)),
            )
        )
    return out


def track_observations(observations: list[Observation], max_dist_px: float, max_frame_gap: int) -> list[Track]:
    observations = sorted(observations, key=lambda obs: (obs.frame_num, obs.broad_type, obs.pixel_r, obs.pixel_c))
    tracks: list[Track] = []
    next_id = 1

    for obs in observations:
        best_track = None
        best_dist = float("inf")
        for track in tracks:
            if track.broad_type != obs.broad_type:
                continue
            frame_gap = obs.frame_num - track.last.frame_num
            if frame_gap <= 0 or frame_gap > max_frame_gap:
                continue
            dist = math.hypot(obs.pixel_r - track.last.pixel_r, obs.pixel_c - track.last.pixel_c)
            if dist < best_dist and dist <= max_dist_px:
                best_track = track
                best_dist = dist
        if best_track is None:
            best_track = Track(track_id=next_id, broad_type=obs.broad_type)
            tracks.append(best_track)
            next_id += 1
        best_track.observations.append(obs)
    return tracks


def summarize_tracks(tracks: list[Track], min_support: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    track_rows = []
    obs_rows = []
    for track in tracks:
        frames = [obs.frame_num for obs in track.observations]
        center_r, center_c = track.center
        is_stable = len(set(frames)) >= min_support
        row = {
            "track_id": track.track_id,
            "broad_type": track.broad_type,
            "node_types": "|".join(sorted({obs.node_type for obs in track.observations})),
            "support_frames": len(set(frames)),
            "observations": len(track.observations),
            "first_frame": min(frames),
            "last_frame": max(frames),
            "center_r": round(center_r, 2),
            "center_c": round(center_c, 2),
            "mean_confidence": round(float(np.mean([obs.confidence for obs in track.observations])), 4),
            "stable": is_stable,
        }
        track_rows.append(row)
        for obs in track.observations:
            obs_rows.append(
                {
                    "track_id": track.track_id,
                    "stable": is_stable,
                    "broad_type": track.broad_type,
                    "name": obs.name,
                    "frame_id": obs.frame_id,
                    "frame_num": obs.frame_num,
                    "node_id": obs.node_id,
                    "pixel_r": obs.pixel_r,
                    "pixel_c": obs.pixel_c,
                    "node_type": obs.node_type,
                    "confidence": obs.confidence,
                }
            )
    return track_rows, obs_rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sequence(group_name: str, frame_df: pd.DataFrame, output_dir: Path) -> str:
    frame_df = frame_df.sort_values("frame_num")
    fig, ax = plt.subplots(figsize=(11, 5))
    for col, color, label in [
        ("raw_bifurcation", "#28d85c", "сырые бифуркации"),
        ("stable_bifurcation", "#0b7a2a", "устойчивые бифуркации"),
        ("artifact", "#9a9a9a", "короткие артефакты"),
        ("crossing_or_overlap", "#5aa0ff", "наложения"),
    ]:
        ax.plot(frame_df["frame_num"], frame_df[col], marker="o", linewidth=1.6, color=color, label=label)
    ax.set_title(f"{group_name}: временная устойчивость узлов")
    ax.set_xlabel("номер кадра")
    ax.set_ylabel("количество")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{group_name}_temporal_junctions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def draw_track_map(
    group_name: str,
    representative_image: Path,
    representative_mask: Path,
    tracks: list[Track],
    output_dir: Path,
    min_support: int,
) -> str:
    gray = np.array(Image.open(representative_image).convert("L"), dtype=np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    mask = np.array(Image.open(representative_mask).convert("L")) > 0
    outline = mask ^ (mask & np.roll(mask, 1, axis=0) & np.roll(mask, -1, axis=0) & np.roll(mask, 1, axis=1) & np.roll(mask, -1, axis=1))
    rgb[outline] = (255, 190, 50)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    colors = {
        "bifurcation": (40, 230, 90),
        "crossing_or_overlap": (80, 160, 255),
        "artifact": (160, 160, 160),
        "uncertain": (255, 140, 40),
    }
    for track in tracks:
        support = len({obs.frame_num for obs in track.observations})
        if support < min_support:
            continue
        r, c = track.center
        color = colors.get(track.broad_type, (255, 255, 255))
        radius = 5 if track.broad_type == "bifurcation" else 4
        draw.ellipse((c - radius, r - radius, c + radius, r + radius), fill=color, outline=(0, 0, 0), width=2)
        draw.text((c + 7, r - 7), str(support), fill=(255, 255, 255))
    path = output_dir / f"{group_name}_stable_track_map.png"
    image.save(path)
    return str(path)


def analyze_group(
    group_name: str,
    pairs: list[tuple[Path, Path, str]],
    output_dir: Path,
    max_dist_px: float,
    max_frame_gap: int,
    min_support: int,
) -> dict[str, object]:
    observations: list[Observation] = []
    raw_frame_rows = []
    for image_path, mask_path, name in pairs:
        result = analyze_graph(
            image_path=image_path,
            mask_path=mask_path,
            output_dir=output_dir,
            name=name,
            include_edge_paths=False,
            write_json_file=False,
            render_overlay_file=False,
        )
        observations.extend(extract_observations(result))
        counts = result["summary"]["junction_counts"]
        raw_frame_rows.append(
            {
                "name": name,
                "frame_num": frame_number(str(result["summary"]["image_meta"]["frame_id"])),
                "raw_bifurcation": int(counts.get("bifurcation", 0)),
                "crossing_or_overlap": int(counts.get("crossing_or_overlap", 0)),
                "artifact": int(counts.get("short_branch_artifact", 0)),
                "uncertain": int(
                    counts.get("uncertain_bifurcation", 0)
                    + counts.get("uncertain_high_order_branch", 0)
                    + counts.get("uncertain_complex_junction", 0)
                ),
            }
        )
    tracks = track_observations(observations, max_dist_px=max_dist_px, max_frame_gap=max_frame_gap)
    track_rows, obs_rows = summarize_tracks(tracks, min_support=min_support)
    stable_by_frame: dict[int, int] = {}
    for row in obs_rows:
        if row["stable"] and row["broad_type"] == "bifurcation":
            stable_by_frame[row["frame_num"]] = stable_by_frame.get(row["frame_num"], 0) + 1

    frame_rows = []
    for row in raw_frame_rows:
        frame_rows.append({**row, "stable_bifurcation": stable_by_frame.get(row["frame_num"], 0)})

    write_csv(track_rows, output_dir / f"{group_name}_tracks.csv")
    write_csv(obs_rows, output_dir / f"{group_name}_observations.csv")
    write_csv(frame_rows, output_dir / f"{group_name}_frames.csv")
    frame_df = pd.DataFrame(frame_rows)
    plot_path = plot_sequence(group_name, frame_df, output_dir)

    rep = max(pairs, key=lambda pair: Image.open(pair[1]).convert("L").point(lambda x: 1 if x > 0 else 0).getbbox()[2] if Image.open(pair[1]).convert("L").getbbox() else 0)
    map_path = draw_track_map(group_name, rep[0], rep[1], tracks, output_dir, min_support=min_support)

    stable_tracks = [row for row in track_rows if row["stable"]]
    stable_bif = [row for row in stable_tracks if row["broad_type"] == "bifurcation"]
    return {
        "group": group_name,
        "frames": len(pairs),
        "observations": len(observations),
        "tracks": len(track_rows),
        "stable_tracks": len(stable_tracks),
        "stable_bifurcation_tracks": len(stable_bif),
        "raw_bifurcation_total": int(sum(row["raw_bifurcation"] for row in raw_frame_rows)),
        "stable_bifurcation_observations": int(sum(row["stable_bifurcation"] for row in frame_rows)),
        "plot": plot_path,
        "track_map": map_path,
    }


def write_report(rows: list[dict[str, object]], output_dir: Path) -> Path:
    path = output_dir / "temporal_junction_report.md"
    lines = [
        "# Темпоральный анализ узлов",
        "",
        "Здесь узлы отслеживаются между соседними кадрами одной серии и одного разметчика.",
        "Зелёная бифуркация считается более надёжной, если появляется в нескольких кадрах рядом с тем же местом.",
        "",
        "| sequence | frames | raw bif total | stable bif observations | stable bif tracks | plot | map |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['group']}` | {row['frames']} | {row['raw_bifurcation_total']} | "
            f"{row['stable_bifurcation_observations']} | {row['stable_bifurcation_tracks']} | "
            f"[plot]({Path(row['plot']).name}) | [map]({Path(row['track_map']).name}) |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track coronary graph junctions through time.")
    parser.add_argument("--image-dir", type=Path, default=Path("3d/все/our_data_with_dublicates_297img/images"))
    parser.add_argument("--mask-dir", type=Path, default=Path("3d/все/our_data_with_dublicates_297img/masks"))
    parser.add_argument("--glob", default="*.png")
    parser.add_argument("--output-dir", type=Path, default=Path("3d/outputs_graph_analysis/temporal_junctions"))
    parser.add_argument("--max-dist-px", type=float, default=18.0)
    parser.add_argument("--max-frame-gap", type=int, default=4)
    parser.add_argument("--min-support", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = iter_pairs(args.image_dir, args.mask_dir, args.glob)
    groups: dict[tuple[str, str, str], list[tuple[Path, Path, str]]] = {}
    for pair in pairs:
        groups.setdefault(group_key(pair[2]), []).append(pair)

    summaries = []
    for key, group_pairs in sorted(groups.items()):
        patient, series, annotator = key
        group_name = f"{patient}_{series}_{annotator}"
        if len(group_pairs) < 2:
            continue
        summaries.append(
            analyze_group(
                group_name,
                sorted(group_pairs, key=lambda p: frame_number(p[2].split("_")[2])),
                args.output_dir,
                max_dist_px=args.max_dist_px,
                max_frame_gap=args.max_frame_gap,
                min_support=args.min_support,
            )
        )
    write_csv(summaries, args.output_dir / "temporal_junction_summary.csv")
    report = write_report(summaries, args.output_dir)
    payload = {
        "sequences": len(summaries),
        "summary_csv": str(args.output_dir / "temporal_junction_summary.csv"),
        "report": str(report),
        "max_dist_px": args.max_dist_px,
        "max_frame_gap": args.max_frame_gap,
        "min_support": args.min_support,
    }
    (args.output_dir / "temporal_junction_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
