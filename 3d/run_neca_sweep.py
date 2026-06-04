#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE_SUMMARY = ROOT / "remote_artifacts" / "20260422" / "baseline_summary.json"
DEFAULT_CASES_DIR = ROOT / "reconstruction_cases"
DEFAULT_OUTPUT_DIR = ROOT / "neca_cases"


def parse_list(value: str, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a NeCA hyperparameter sweep on prepared coronary reconstruction cases.")
    parser.add_argument("--repo-dir", type=Path, required=True, help="Path to NeCA repo")
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", nargs="*", default=None, help="Explicit case ids to run. If omitted, top-k by baseline mean IoU are used.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--input-modes", default="mask,edt_mask,blurred_mask")
    parser.add_argument("--bundle-sizes", default="1,3")
    parser.add_argument("--detector-sizes", default="256")
    parser.add_argument("--volume-sizes", default="128")
    parser.add_argument("--volume-extents", default="160,180")
    parser.add_argument("--bounds", default="0.25,0.30")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-jobs", type=int, default=0, help="Optional hard limit on number of jobs. 0 means unlimited.")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def load_cases(args: argparse.Namespace) -> list[dict]:
    summary = None
    if args.baseline_summary.exists():
        summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    if args.cases:
        if summary is None:
            return [{"case_id": case_id, "best_phase_index": 1} for case_id in args.cases]
        wanted = set(args.cases)
        picked = [item for item in summary if item["case_id"] in wanted]
        missing = wanted.difference(item["case_id"] for item in picked)
        if missing:
            picked.extend({"case_id": case_id, "best_phase_index": 1} for case_id in sorted(missing))
        return picked
    if summary is None:
        raise SystemExit(f"Baseline summary not found: {args.baseline_summary}")
    filtered = [item for item in summary if item.get("status") == "ok"]
    filtered.sort(key=lambda item: float(item.get("best_mean_iou", 0.0)), reverse=True)
    return filtered[: max(1, args.top_k)]


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def prepare_job(args: argparse.Namespace, case_id: str, phase_index: int, input_mode: str, bundle_size: int, detector_size: int, volume_size: int, volume_extent: float, bound: float) -> Path:
    cmd = [
        args.python,
        str(ROOT / "prepare_neca_case.py"),
        "--case-id", case_id,
        "--phase-index", str(phase_index),
        "--input-mode", input_mode,
        "--cases-dir", str(args.cases_dir),
        "--output-dir", str(args.output_dir),
        "--detector-size", str(detector_size),
        "--volume-size", str(volume_size),
        "--volume-extent-mm", str(volume_extent),
        "--epochs", str(args.epochs),
        "--eval-every", str(args.eval_every),
        "--save-every", str(args.save_every),
        "--bound", str(bound),
        "--bundle-size", str(bundle_size),
    ]
    run_command(cmd, cwd=ROOT)
    return args.output_dir / case_id / f"phase_{phase_index:02d}_b{bundle_size}_{input_mode}_{detector_size}"


def train_job(args: argparse.Namespace, repo_dir: Path, case_dir: Path) -> None:
    cmd = [args.python, str(repo_dir / "train.py"), "--config", str(case_dir / "config" / "CCTA.yaml")]
    run_command(cmd, cwd=repo_dir)


def evaluate_job(args: argparse.Namespace, repo_dir: Path, case_dir: Path) -> dict:
    cmd = [
        args.python,
        str(ROOT / "evaluate_neca_case.py"),
        "--repo-dir", str(repo_dir),
        "--case-dir", str(case_dir),
    ]
    run_command(cmd, cwd=ROOT)
    summary = json.loads((case_dir / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    return summary


def main() -> None:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    cases = load_cases(args)

    input_modes = parse_list(args.input_modes, str)
    bundle_sizes = parse_list(args.bundle_sizes, int)
    detector_sizes = parse_list(args.detector_sizes, int)
    volume_sizes = parse_list(args.volume_sizes, int)
    volume_extents = parse_list(args.volume_extents, float)
    bounds = parse_list(args.bounds, float)

    job_specs = []
    for case in cases:
        for input_mode, bundle_size, detector_size, volume_size, volume_extent, bound in itertools.product(
            input_modes, bundle_sizes, detector_sizes, volume_sizes, volume_extents, bounds
        ):
            job_specs.append({
                "case_id": case["case_id"],
                "phase_index": int(case["best_phase_index"]),
                "input_mode": input_mode,
                "bundle_size": bundle_size,
                "detector_size": detector_size,
                "volume_size": volume_size,
                "volume_extent_mm": volume_extent,
                "bound": bound,
            })
    if args.max_jobs > 0:
        job_specs = job_specs[: args.max_jobs]

    results = []
    for job in job_specs:
        case_dir = prepare_job(
            args,
            case_id=job["case_id"],
            phase_index=job["phase_index"],
            input_mode=job["input_mode"],
            bundle_size=job["bundle_size"],
            detector_size=job["detector_size"],
            volume_size=job["volume_size"],
            volume_extent=job["volume_extent_mm"],
            bound=job["bound"],
        )
        train_job(args, repo_dir, case_dir)
        summary = evaluate_job(args, repo_dir, case_dir)
        row = {
            **job,
            "case_dir": str(case_dir.resolve()),
            "mean_iou_hard": summary["hard_mask_metrics"].get("mean_mean_iou_hard_mask", 0.0),
            "mean_dice_hard": summary["hard_mask_metrics"].get("mean_mean_dice_hard_mask", 0.0),
            "mean_iou_training": summary["training_target_metrics"].get("mean_mean_iou_training_target", 0.0),
            "mean_dice_training": summary["training_target_metrics"].get("mean_mean_dice_training_target", 0.0),
            "occupied_voxels_abs_0_5": summary["volume_metrics"].get("occupancy_by_threshold", {}).get("abs_0_5", {}).get("occupied_voxels", 0),
            "occupied_voxels_p999": summary["volume_metrics"].get("occupancy_by_threshold", {}).get("p999", {}).get("occupied_voxels", 0),
            "component_count_p999": summary["volume_metrics"].get("occupancy_by_threshold", {}).get("p999", {}).get("component_count", 0),
            "volume_p999": summary["volume_metrics"].get("stats", {}).get("p999", 0.0),
            "volume_max": summary["volume_metrics"].get("stats", {}).get("max", 0.0),
            "latest_volume": summary.get("latest_volume"),
            "evaluation_summary": str((case_dir / "evaluation" / "summary.json").resolve()),
        }
        results.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    results.sort(key=lambda item: (float(item["mean_dice_hard"]), float(item["mean_iou_hard"])), reverse=True)
    out_dir = ROOT / "remote_artifacts" / "20260422"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "neca_sweep_summary.json"
    csv_path = out_dir / "neca_sweep_summary.csv"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if results:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(json.dumps({
        "jobs_run": len(results),
        "summary_json": str(json_path.resolve()),
        "summary_csv": str(csv_path.resolve()),
        "best": results[0] if results else None,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
