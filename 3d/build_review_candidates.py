#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def frame_number(frame_id: str) -> int:
    if isinstance(frame_id, str) and frame_id.startswith("f"):
        return int(frame_id[1:])
    return int(frame_id)


def read_per_image(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "name": str,
            "patient_id": str,
            "series_id": str,
            "frame_id": str,
            "annotator_id": str,
            "annotator_name": str,
            "overlay_path": str,
        },
    )
    for col in [
        "mask_pixels",
        "skeleton_pixels",
        "nodes",
        "edges",
        "bifurcation",
        "uncertain_bifurcation",
        "short_branch_artifact",
        "crossing_or_overlap",
        "uncertain_high_order_branch",
        "uncertain_complex_junction",
        "junction_confidence_mean",
        "branch_radius_mean",
        "branch_gradient_mean",
        "vessel_signal_mean",
        "vessel_signal_std",
        "background_signal_mean",
        "mask_gradient_mean",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["frame_num"] = df["frame_id"].map(frame_number)
    df["uncertain_total"] = (
        df["uncertain_bifurcation"]
        + df["short_branch_artifact"]
        + df["uncertain_high_order_branch"]
        + df["uncertain_complex_junction"]
    )
    return df


def add_temporal_jumps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["patient_id", "series_id", "annotator_id", "frame_num"]).copy()
    out["mask_jump"] = 0.0
    out["bifurcation_jump"] = 0.0
    out["signal_jump"] = 0.0
    keys = ["patient_id", "series_id", "annotator_id"]
    for _, idx in out.groupby(keys).groups.items():
        group = out.loc[idx].sort_values("frame_num")
        out.loc[group.index, "mask_jump"] = group["mask_pixels"].diff().abs().fillna(0.0)
        out.loc[group.index, "bifurcation_jump"] = group["bifurcation"].diff().abs().fillna(0.0)
        out.loc[group.index, "signal_jump"] = group["vessel_signal_mean"].diff().abs().fillna(0.0)
    return out


def robust_z(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad < 1e-9:
        std = float(values.std())
        denom = std if std > 1e-9 else 1.0
    else:
        denom = 1.4826 * mad
    return (values - median) / denom


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = add_temporal_jumps(df)
    metrics = [
        "crossing_or_overlap",
        "uncertain_total",
        "bifurcation",
        "bifurcation_jump",
        "mask_jump",
        "signal_jump",
        "nodes",
    ]
    for metric in metrics:
        out[f"z_{metric}"] = robust_z(out[metric]).clip(lower=0.0)
    out["review_score"] = (
        2.5 * out["z_crossing_or_overlap"]
        + 2.0 * out["z_uncertain_total"]
        + 1.4 * out["z_bifurcation_jump"]
        + 1.2 * out["z_bifurcation"]
        + 0.9 * out["z_mask_jump"]
        + 0.6 * out["z_signal_jump"]
        + 0.5 * out["z_nodes"]
    )
    reasons = []
    for row in out.itertuples(index=False):
        row_reasons = []
        if row.crossing_or_overlap > 0:
            row_reasons.append(f"crossing/overlap={int(row.crossing_or_overlap)}")
        if row.uncertain_total > 0:
            row_reasons.append(f"uncertain={int(row.uncertain_total)}")
        if row.bifurcation_jump >= 8:
            row_reasons.append(f"скачок бифуркаций={row.bifurcation_jump:.0f}")
        if row.mask_jump >= out["mask_jump"].quantile(0.90):
            row_reasons.append(f"скачок площади={row.mask_jump:.0f}px")
        if row.bifurcation >= out["bifurcation"].quantile(0.95):
            row_reasons.append(f"много бифуркаций={int(row.bifurcation)}")
        if not row_reasons:
            row_reasons.append("высокий суммарный балл")
        reasons.append("; ".join(row_reasons))
    out["review_reason"] = reasons
    return out.sort_values("review_score", ascending=False)


def copy_overlays(candidates: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []
    for row in candidates.itertuples(index=False):
        source = Path(str(row.overlay_path))
        if source.exists():
            target = overlay_dir / source.name
            shutil.copy2(source, target)
            local_paths.append(str(target.relative_to(output_dir)))
        else:
            local_paths.append("")
    out = candidates.copy()
    out["local_overlay"] = local_paths
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "rank",
        "review_score",
        "review_reason",
        "name",
        "patient_id",
        "series_id",
        "frame_id",
        "annotator_id",
        "annotator_name",
        "mask_pixels",
        "nodes",
        "edges",
        "bifurcation",
        "crossing_or_overlap",
        "uncertain_total",
        "bifurcation_jump",
        "mask_jump",
        "vessel_signal_mean",
        "signal_jump",
        "junction_confidence_mean",
        "overlay_path",
        "local_overlay",
    ]
    df[cols].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_html(df: pd.DataFrame, output_dir: Path, title: str) -> Path:
    path = output_dir / "review_candidates.html"
    cards = []
    for row in df.itertuples(index=False):
        img = html.escape(str(row.local_overlay))
        cards.append(
            f"""
            <article class="card">
              <div class="meta">
                <h2>#{int(row.rank)} {html.escape(row.name)}</h2>
                <p><b>Балл:</b> {row.review_score:.2f}</p>
                <p><b>Причина:</b> {html.escape(row.review_reason)}</p>
                <p><b>Пациент/серия/кадр:</b> {html.escape(row.patient_id)} / {html.escape(row.series_id)} / {html.escape(row.frame_id)}</p>
                <p><b>Разметчик:</b> {html.escape(str(row.annotator_id))} {html.escape(str(row.annotator_name))}</p>
                <p><b>Бифуркации:</b> {int(row.bifurcation)}; <b>наложения:</b> {int(row.crossing_or_overlap)}; <b>сомнительные:</b> {int(row.uncertain_total)}</p>
                <p><b>Скачок бифуркаций:</b> {row.bifurcation_jump:.0f}; <b>скачок площади:</b> {row.mask_jump:.0f}px</p>
              </div>
              <a href="{img}"><img src="{img}" alt="{html.escape(row.name)}"></a>
            </article>
            """
        )
    html_text = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #111418; color: #eef2f6; }}
    header {{ padding: 22px 28px; position: sticky; top: 0; background: #111418; border-bottom: 1px solid #2b323b; z-index: 2; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    header p {{ margin: 0; color: #b9c2cc; }}
    main {{ padding: 22px; display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; }}
    .card {{ background: #1a1f26; border: 1px solid #303843; border-radius: 8px; overflow: hidden; }}
    .meta {{ padding: 14px 16px; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; }}
    p {{ margin: 4px 0; color: #d9dee5; font-size: 13px; }}
    img {{ width: 100%; display: block; background: #000; }}
    a {{ color: inherit; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Кадры отсортированы по суммарной подозрительности: наложения, сомнительные узлы, скачки во времени и необычно большое число бифуркаций.</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")
    return path


def write_markdown(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "review_candidates.md"
    lines = [
        "# Кандидаты для ручной проверки",
        "",
        "Выбраны кадры, где алгоритм видит больше всего риска: наложения, сомнительные узлы, скачки числа бифуркаций или площади маски.",
        "",
        "| rank | name | score | reason | bif | crossing | uncertain | overlay |",
        "|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {int(row.rank)} | `{row.name}` | {row.review_score:.2f} | {row.review_reason} | "
            f"{int(row.bifurcation)} | {int(row.crossing_or_overlap)} | {int(row.uncertain_total)} | "
            f"[png]({row.local_overlay}) |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build review queue for suspicious coronary graph overlays.")
    parser.add_argument("--analysis-dir", type=Path, default=Path("3d/outputs_graph_analysis/manual_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("3d/outputs_graph_analysis/review_candidates"))
    parser.add_argument("--top", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_image = read_per_image(args.analysis_dir / "batch_graph_analysis_table.csv")
    scored = score_candidates(per_image)
    top = scored.head(args.top).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top = copy_overlays(top, args.output_dir)
    write_csv(top, args.output_dir / "review_candidates.csv")
    md = write_markdown(top, args.output_dir)
    html_path = write_html(top, args.output_dir, f"Top {len(top)} coronary graph review candidates")
    print(f"html={html_path}")
    print(f"markdown={md}")
    print(f"csv={args.output_dir / 'review_candidates.csv'}")


if __name__ == "__main__":
    main()
