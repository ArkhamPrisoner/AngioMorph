#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


KNOWN_TERRITORY = {
    ("p0001", "00000001"): "left_coronary",
    ("p0001", "00000002"): "left_coronary",
    ("p0001", "00000003"): "left_coronary",
    ("p0001", "00000004"): "right_coronary",
}

ANATOMY_PRIOR = {
    "left_coronary": {
        "expected_root": "LM -> LAD + LCx",
        "expected_branches": "LAD diagonal/septal branches; LCx obtuse marginal branches; optional ramus intermedius",
        "min_peak_bifurcations": 3,
        "max_crossing_fraction": 0.20,
    },
    "right_coronary": {
        "expected_root": "RCA -> acute marginal branches -> PDA/PL near crux, depending on dominance",
        "expected_branches": "acute marginal, PDA, posterolateral branches; dominance can change distal topology",
        "min_peak_bifurcations": 2,
        "max_crossing_fraction": 0.20,
    },
    "unknown": {
        "expected_root": "unknown projection/territory",
        "expected_branches": "use generic coronary tree checks only",
        "min_peak_bifurcations": 1,
        "max_crossing_fraction": 0.25,
    },
}


def frame_number(frame_id: str) -> int:
    if isinstance(frame_id, str) and frame_id.startswith("f"):
        return int(frame_id[1:])
    return int(frame_id)


def read_tables(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dtype = {
        "name": str,
        "patient_id": str,
        "series_id": str,
        "frame_id": str,
        "annotator_id": str,
        "annotator_name": str,
    }
    per_image = pd.read_csv(analysis_dir / "batch_graph_analysis_table.csv", dtype=dtype)
    edges = pd.read_csv(
        analysis_dir / "batch_edges_table.csv",
        dtype={"name": str, "series_id": str, "frame_id": str, "annotator_id": str},
    )
    junctions = pd.read_csv(
        analysis_dir / "batch_junctions_table.csv",
        dtype={"name": str, "series_id": str, "frame_id": str, "annotator_id": str, "type": str},
    )
    per_image["frame_num"] = per_image["frame_id"].map(frame_number)
    edges["frame_num"] = edges["frame_id"].map(frame_number)
    junctions["frame_num"] = junctions["frame_id"].map(frame_number)
    numeric_cols = [
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
    ]
    for col in numeric_cols:
        if col in per_image.columns:
            per_image[col] = pd.to_numeric(per_image[col], errors="coerce").fillna(0.0)
    for col in ["length_px", "radius_median_px", "signal_mean"]:
        edges[col] = pd.to_numeric(edges[col], errors="coerce").fillna(0.0)
    for col in ["confidence", "degree"]:
        junctions[col] = pd.to_numeric(junctions[col], errors="coerce").fillna(0.0)
    return per_image, edges, junctions


def anatomy_score(row: pd.Series) -> tuple[float, list[str]]:
    territory = row["territory"]
    prior = ANATOMY_PRIOR.get(territory, ANATOMY_PRIOR["unknown"])
    notes: list[str] = []

    peak_bif = float(row["peak_bifurcation"])
    total_junctions = max(float(row["total_junctions"]), 1.0)
    crossing_fraction = float(row["total_crossing_or_overlap"]) / total_junctions
    uncertain_fraction = float(row["total_uncertain"]) / total_junctions
    temporal_coverage = float(row["frames_above_half_peak"]) / max(float(row["frames"]), 1.0)

    bif_score = min(1.0, peak_bif / max(float(prior["min_peak_bifurcations"]), 1.0))
    crossing_score = max(0.0, 1.0 - crossing_fraction / float(prior["max_crossing_fraction"]))
    uncertain_score = max(0.0, 1.0 - uncertain_fraction / 0.25)
    temporal_score = min(1.0, temporal_coverage / 0.35)
    confidence_score = float(row["mean_junction_confidence"])

    score = (
        0.30 * bif_score
        + 0.22 * crossing_score
        + 0.18 * uncertain_score
        + 0.15 * temporal_score
        + 0.15 * confidence_score
    )

    if peak_bif < prior["min_peak_bifurcations"]:
        notes.append("мало найденных бифуркаций для ожидаемой коронарной ветвистости")
    if crossing_fraction > prior["max_crossing_fraction"]:
        notes.append("много crossing/overlap: вероятны наложения или спорная маска")
    if uncertain_fraction > 0.15:
        notes.append("много неуверенных узлов: нужна ручная проверка")
    if temporal_coverage < 0.25:
        notes.append("короткое временное окно видимости дерева")
    if not notes:
        notes.append("грубая топология согласуется с выбранным анатомическим шаблоном")
    return float(np.clip(score, 0.0, 1.0)), notes


def build_series_summary(per_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["patient_id", "series_id", "annotator_id", "annotator_name"]
    for key, group in per_image.sort_values(keys + ["frame_num"]).groupby(keys, dropna=False):
        patient_id, series_id, annotator_id, annotator_name = key
        group = group.sort_values("frame_num")
        peak_idx = group["mask_pixels"].idxmax()
        peak = group.loc[peak_idx]
        total_uncertain = (
            group["uncertain_bifurcation"]
            + group["short_branch_artifact"]
            + group["uncertain_high_order_branch"]
            + group["uncertain_complex_junction"]
        )
        half_peak = float(group["mask_pixels"].max()) * 0.5
        territory = KNOWN_TERRITORY.get((str(patient_id), str(series_id)), "unknown")
        row = {
            "patient_id": patient_id,
            "series_id": series_id,
            "annotator_id": annotator_id,
            "annotator_name": annotator_name,
            "territory": territory,
            "frames": int(len(group)),
            "first_frame": int(group["frame_num"].min()),
            "last_frame": int(group["frame_num"].max()),
            "peak_frame": peak["frame_id"],
            "peak_mask_pixels": float(peak["mask_pixels"]),
            "peak_bifurcation": float(group["bifurcation"].max()),
            "peak_crossing_or_overlap": float(group["crossing_or_overlap"].max()),
            "total_junctions": float(
                group[
                    [
                        "bifurcation",
                        "crossing_or_overlap",
                        "uncertain_bifurcation",
                        "short_branch_artifact",
                        "uncertain_high_order_branch",
                        "uncertain_complex_junction",
                    ]
                ].sum(axis=1).sum()
            ),
            "total_crossing_or_overlap": float(group["crossing_or_overlap"].sum()),
            "total_uncertain": float(total_uncertain.sum()),
            "mean_junction_confidence": float(group["junction_confidence_mean"].replace(0, np.nan).mean(skipna=True) or 0.0),
            "mean_branch_radius": float(group["branch_radius_mean"].mean()),
            "mean_mask_gradient": float(group["mask_gradient_mean"].mean()),
            "mean_vessel_signal": float(group["vessel_signal_mean"].mean()),
            "frames_above_half_peak": int((group["mask_pixels"] >= half_peak).sum()),
        }
        score, notes = anatomy_score(pd.Series(row))
        row["anatomy_score"] = score
        row["notes"] = "; ".join(notes)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["patient_id", "series_id", "annotator_id"])


def build_projection_summary(series_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["patient_id", "series_id", "territory"]
    for key, group in series_summary.groupby(keys, dropna=False):
        patient_id, series_id, territory = key
        rows.append(
            {
                "patient_id": patient_id,
                "series_id": series_id,
                "territory": territory,
                "annotators": int(group["annotator_id"].nunique()),
                "frame_sequences": int(len(group)),
                "peak_mask_pixels_median": float(group["peak_mask_pixels"].median()),
                "peak_bifurcation_median": float(group["peak_bifurcation"].median()),
                "total_crossing_or_overlap": float(group["total_crossing_or_overlap"].sum()),
                "total_uncertain": float(group["total_uncertain"].sum()),
                "anatomy_score_mean": float(group["anatomy_score"].mean()),
                "mean_branch_radius": float(group["mean_branch_radius"].mean()),
                "mean_vessel_signal": float(group["mean_vessel_signal"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["patient_id", "series_id"])


def save_canonical_topology(path: Path) -> None:
    graph = nx.DiGraph()
    edges = [
        ("Aorta", "LM"),
        ("LM", "LAD"),
        ("LM", "LCx"),
        ("LM", "RI optional"),
        ("LAD", "D1"),
        ("LAD", "D2"),
        ("LAD", "Septal"),
        ("LAD", "Distal LAD"),
        ("LCx", "OM1"),
        ("LCx", "OM2"),
        ("LCx", "Distal LCx"),
        ("Aorta", "RCA"),
        ("RCA", "AM"),
        ("RCA", "Distal RCA"),
        ("Distal RCA", "PDA"),
        ("Distal RCA", "PL"),
    ]
    graph.add_edges_from(edges)
    pos = {
        "Aorta": (0, 0),
        "LM": (-1.3, -1),
        "LAD": (-2.2, -2),
        "LCx": (-0.6, -2),
        "RI optional": (-1.4, -2.3),
        "D1": (-3.2, -2.7),
        "D2": (-2.9, -3.4),
        "Septal": (-1.9, -3.0),
        "Distal LAD": (-2.2, -4),
        "OM1": (-0.2, -2.8),
        "OM2": (0.0, -3.4),
        "Distal LCx": (-0.7, -4),
        "RCA": (1.5, -1),
        "AM": (2.7, -1.9),
        "Distal RCA": (1.7, -2.6),
        "PDA": (1.0, -3.6),
        "PL": (2.4, -3.6),
    }
    colors = []
    for node in graph.nodes:
        if node in {"LM", "LAD", "LCx", "RCA"}:
            colors.append("#40c463")
        elif "optional" in node:
            colors.append("#f0c828")
        else:
            colors.append("#8ab4f8")
    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=14, width=1.8)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=1200, edgecolors="#222222")
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    ax.set_title("Упрощённый анатомический шаблон коронарного дерева")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_temporal(per_image: pd.DataFrame, output_dir: Path) -> list[str]:
    paths: list[str] = []
    for patient_id, patient_df in per_image.groupby("patient_id"):
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
        for (series_id, annotator_id), group in patient_df.groupby(["series_id", "annotator_id"]):
            group = group.sort_values("frame_num")
            label = f"{series_id}/{annotator_id}"
            axes[0].plot(group["frame_num"], group["mask_pixels"], marker="o", linewidth=1.4, label=label)
            axes[1].plot(group["frame_num"], group["bifurcation"], marker="o", linewidth=1.4, label=label)
            axes[2].plot(group["frame_num"], group["vessel_signal_mean"], marker="o", linewidth=1.4, label=label)
        axes[0].set_title(f"{patient_id}: площадь маски по времени")
        axes[0].set_ylabel("пиксели маски")
        axes[1].set_title("Бифуркации по времени")
        axes[1].set_ylabel("количество")
        axes[2].set_title("Средний сосудистый сигнал по времени")
        axes[2].set_ylabel("сигнал")
        axes[2].set_xlabel("номер кадра")
        for ax in axes:
            ax.grid(alpha=0.25)
        axes[0].legend(ncol=4, fontsize=7)
        fig.tight_layout()
        path = output_dir / f"{patient_id}_temporal_curves.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_projection_comparison(projection_summary: pd.DataFrame, output_dir: Path) -> list[str]:
    paths: list[str] = []
    labels = [
        f"{row.patient_id}/{row.series_id}\n{row.territory}"
        for row in projection_summary.itertuples(index=False)
    ]
    x = np.arange(len(projection_summary))

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    ax.bar(x, projection_summary["anatomy_score_mean"], color="#40c463")
    ax.set_ylim(0, 1)
    ax.set_title("Согласованность серий с анатомическим шаблоном")
    ax.set_ylabel("балл 0..1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    path = output_dir / "projection_anatomy_score.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax1 = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    ax1.bar(x - 0.18, projection_summary["peak_bifurcation_median"], width=0.36, color="#40c463", label="пик бифуркаций")
    ax1.bar(x + 0.18, projection_summary["total_crossing_or_overlap"], width=0.36, color="#5aa0ff", label="crossing/overlap всего")
    ax1.set_title("Бифуркации и наложения по проекциям")
    ax1.set_ylabel("количество")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend()
    fig.tight_layout()
    path = output_dir / "projection_bifurcation_crossing.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_report(
    output_dir: Path,
    series_summary: pd.DataFrame,
    projection_summary: pd.DataFrame,
    plot_paths: list[str],
) -> Path:
    report_path = output_dir / "patient_context_report.md"
    lines = [
        "# Пациентский анализ коронарного графа",
        "",
        "Этот отчёт смотрит не один кадр, а последовательности кадров одного пациента и разные серии как разные проекции или разные проходы контраста.",
        "",
        "## Что добавлено",
        "",
        "- Временная аналитика: площадь маски, число бифуркаций и средний сосудистый сигнал по номеру кадра.",
        "- Межпроекционное сравнение: серии одного пациента сравниваются по сложности графа, числу бифуркаций, наложений и радиусам.",
        "- Анатомическая проверка: результат сверяется с упрощённым шаблоном LM/LAD/LCx/RCA, но только как подсказка, а не как жёсткое правило.",
        "",
        "## Важное ограничение",
        "",
        "Без геометрии C-arm и синхронизации двух проекций это не настоящая 3D-реконструкция. Это согласование графов и временных признаков, которое помогает найти подозрительные места.",
        "",
        "## Сводка по сериям",
        "",
    ]
    for row in projection_summary.itertuples(index=False):
        prior = ANATOMY_PRIOR.get(row.territory, ANATOMY_PRIOR["unknown"])
        lines.extend(
            [
                f"### {row.patient_id} / серия {row.series_id}",
                "",
                f"- Анатомическая зона: `{row.territory}`",
                f"- Ожидаемая схема: {prior['expected_root']}",
                f"- Ожидаемые ветви: {prior['expected_branches']}",
                f"- Разметчиков/последовательностей: {row.annotators}/{row.frame_sequences}",
                f"- Медианный пик бифуркаций: {row.peak_bifurcation_median:.1f}",
                f"- Всего crossing/overlap: {row.total_crossing_or_overlap:.0f}",
                f"- Средний анатомический балл: {row.anatomy_score_mean:.3f}",
                "",
            ]
        )
    lines.extend(["## Графики", ""])
    for path in plot_paths:
        lines.append(f"- `{Path(path).name}`")
    lines.extend(
        [
            "",
            "## Таблицы",
            "",
            "- `patient_series_temporal_summary.csv`: одна строка на пациента/серию/разметчика.",
            "- `patient_projection_summary.csv`: агрегат по пациенту и серии.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient-level temporal and multi-projection coronary graph analysis.")
    parser.add_argument("--analysis-dir", type=Path, default=Path("3d/outputs_graph_analysis/manual_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("3d/outputs_graph_analysis/patient_context"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_image, edges, junctions = read_tables(args.analysis_dir)
    series_summary = build_series_summary(per_image)
    projection_summary = build_projection_summary(series_summary)

    series_summary.to_csv(args.output_dir / "patient_series_temporal_summary.csv", index=False)
    projection_summary.to_csv(args.output_dir / "patient_projection_summary.csv", index=False)
    save_canonical_topology(args.output_dir / "canonical_coronary_topology.png")
    plots = []
    plots.extend(plot_temporal(per_image, args.output_dir))
    plots.extend(plot_projection_comparison(projection_summary, args.output_dir))
    plots.append(str(args.output_dir / "canonical_coronary_topology.png"))
    report_path = write_report(args.output_dir, series_summary, projection_summary, plots)

    payload = {
        "analysis_dir": str(args.analysis_dir),
        "output_dir": str(args.output_dir),
        "patients": int(per_image["patient_id"].nunique()),
        "images": int(len(per_image)),
        "series_sequences": int(len(series_summary)),
        "projection_rows": int(len(projection_summary)),
        "plots": plots,
        "report_path": str(report_path),
    }
    (args.output_dir / "patient_context_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
