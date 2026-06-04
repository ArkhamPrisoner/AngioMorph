#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES_DIR = ROOT / "reconstruction_cases"
OUTPUT_DIR = ROOT / "experiment_results" / "calibrated_voxel_baseline"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python3")
    args = parser.parse_args()

    summary = load_json(CASES_DIR / "summary.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for case in summary["cases"]:
        case_id = case["case_id"]
        case_manifest = CASES_DIR / case_id / "case_manifest.json"
        case_out = OUTPUT_DIR / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str(ROOT / "run_calibrated_voxel_baseline.py"),
            "--case-manifest",
            str(case_manifest),
            "--output-dir",
            str(case_out),
        ]
        completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        (case_out / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (case_out / "stderr.log").write_text(completed.stderr, encoding="utf-8")

        result_path = case_out / "results.json"
        if completed.returncode != 0 or not result_path.exists():
            rows.append(
                {
                    "case_id": case_id,
                    "patient": case["patient"],
                    "series_a": case["series_a"],
                    "series_b": case["series_b"],
                    "status": "failed",
                    "best_method": None,
                    "best_phase_index": None,
                    "best_score_total": None,
                    "best_mean_iou": None,
                    "results_json": str(result_path.resolve()),
                }
            )
            continue

        result = load_json(result_path)
        best = result["best_run"]
        rows.append(
            {
                "case_id": case_id,
                "patient": case["patient"],
                "series_a": case["series_a"],
                "series_b": case["series_b"],
                "status": "ok",
                "best_method": best["method_id"],
                "best_phase_index": best["phase_index"],
                "best_score_total": best["metrics"]["score_total"],
                "best_mean_iou": best["metrics"]["mean_iou"],
                "results_json": str(result_path.resolve()),
            }
        )

    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "patient",
                "series_a",
                "series_b",
                "status",
                "best_method",
                "best_phase_index",
                "best_score_total",
                "best_mean_iou",
                "results_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
