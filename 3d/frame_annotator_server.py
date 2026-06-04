#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


LABEL_TYPES = ["stenosis", "bifurcation", "intersection", "uncertain"]
SHAPE_TYPES = ["point", "segment", "box"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_frame_name(stem: str) -> dict[str, str]:
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected frame name: {stem}")
    patient_id = parts[0]
    series_id = parts[1]
    frame_id = "_".join(parts[2:])
    return {
        "frame_key": stem,
        "patient_id": patient_id,
        "series_id": series_id,
        "frame_id": frame_id,
    }


def build_manifest(dataset_dir: Path) -> dict[str, Any]:
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing masks dir: {masks_dir}")

    frame_entries: list[dict[str, Any]] = []
    series_order: dict[str, list[str]] = {}

    for image_path in sorted(images_dir.glob("*.png")):
        meta = parse_frame_name(image_path.stem)
        mask_path = masks_dir / image_path.name
        frame_entry = {
            **meta,
            "image_rel_path": str(image_path.relative_to(dataset_dir).as_posix()),
            "mask_rel_path": str(mask_path.relative_to(dataset_dir).as_posix()) if mask_path.exists() else None,
        }
        frame_entries.append(frame_entry)
        series_order.setdefault(meta["series_id"], []).append(meta["frame_key"])

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_name": dataset_dir.name,
        "label_types": LABEL_TYPES,
        "frames": frame_entries,
        "series_order": [
            {
                "series_id": series_id,
                "frame_keys": frame_keys,
            }
            for series_id, frame_keys in sorted(series_order.items())
        ],
    }


def load_annotations(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {
            "version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "label_types": LABEL_TYPES,
            "shape_types": SHAPE_TYPES,
            "frames": {},
        }

    frames = data.setdefault("frames", {})
    for frame in manifest["frames"]:
        frames.setdefault(
            frame["frame_key"],
            {
                "patient_id": frame["patient_id"],
                "series_id": frame["series_id"],
                "frame_id": frame["frame_id"],
                "image_rel_path": frame["image_rel_path"],
                "mask_rel_path": frame["mask_rel_path"],
                "annotations": [],
            },
        )
    data["label_types"] = LABEL_TYPES
    data["shape_types"] = SHAPE_TYPES
    return data


def normalize_geometry(annotation: dict[str, Any]) -> dict[str, Any] | None:
    shape = annotation.get("shape", "point")
    geometry = annotation.get("geometry")

    if shape == "point":
        if isinstance(geometry, dict):
            x = geometry.get("x")
            y = geometry.get("y")
        else:
            x = annotation.get("x")
            y = annotation.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        return {"shape": "point", "x": round(float(x), 3), "y": round(float(y), 3)}

    if shape == "segment":
        points = geometry.get("points") if isinstance(geometry, dict) else annotation.get("points")
        if not isinstance(points, list) or len(points) != 2:
            return None
        normalized_points = []
        for point in points:
            x = point.get("x") if isinstance(point, dict) else None
            y = point.get("y") if isinstance(point, dict) else None
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return None
            normalized_points.append({"x": round(float(x), 3), "y": round(float(y), 3)})
        return {"shape": "segment", "points": normalized_points}

    if shape == "box":
        if not isinstance(geometry, dict):
            return None
        x = geometry.get("x")
        y = geometry.get("y")
        width = geometry.get("width")
        height = geometry.get("height")
        if not all(isinstance(value, (int, float)) for value in [x, y, width, height]):
            return None
        return {
            "shape": "box",
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "width": round(float(width), 3),
            "height": round(float(height), 3),
        }

    return None


def normalize_annotations(payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    known_frames = {frame["frame_key"]: frame for frame in manifest["frames"]}
    frames_payload = payload.get("frames")
    if not isinstance(frames_payload, dict):
        raise ValueError("Payload must contain a frames object")

    normalized = {
        "version": 1,
        "created_at": payload.get("created_at") or utc_now(),
        "updated_at": utc_now(),
        "label_types": LABEL_TYPES,
        "shape_types": SHAPE_TYPES,
        "frames": {},
    }

    for frame_key, frame_payload in frames_payload.items():
        if frame_key not in known_frames:
            continue
        known_frame = known_frames[frame_key]
        annotations = frame_payload.get("annotations", [])
        normalized_annotations = []
        for annotation in annotations:
            label_type = annotation.get("type")
            if label_type not in LABEL_TYPES:
                continue
            geometry = normalize_geometry(annotation)
            if geometry is None:
                continue
            normalized_annotations.append(
                {
                    "id": str(annotation.get("id") or f"{frame_key}_{len(normalized_annotations) + 1}"),
                    "type": label_type,
                    "shape": geometry["shape"],
                    "geometry": geometry,
                    "note": str(annotation.get("note") or "").strip(),
                    "created_at": annotation.get("created_at") or utc_now(),
                    "updated_at": utc_now(),
                }
            )

        normalized["frames"][frame_key] = {
            "patient_id": known_frame["patient_id"],
            "series_id": known_frame["series_id"],
            "frame_id": known_frame["frame_id"],
            "image_rel_path": known_frame["image_rel_path"],
            "mask_rel_path": known_frame["mask_rel_path"],
            "annotations": normalized_annotations,
        }

    for frame_key, known_frame in known_frames.items():
        normalized["frames"].setdefault(
            frame_key,
            {
                "patient_id": known_frame["patient_id"],
                "series_id": known_frame["series_id"],
                "frame_id": known_frame["frame_id"],
                "image_rel_path": known_frame["image_rel_path"],
                "mask_rel_path": known_frame["mask_rel_path"],
                "annotations": [],
            },
        )

    return normalized


@dataclass
class AppState:
    dataset_dir: Path
    annotations_path: Path
    ui_dir: Path
    manifest: dict[str, Any]
    annotations: dict[str, Any]

    def save_annotations(self, payload: dict[str, Any]) -> None:
        self.annotations = normalize_annotations(payload, self.manifest)
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        self.annotations_path.write_text(
            json.dumps(self.annotations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class AnnotatorHandler(SimpleHTTPRequestHandler):
    server_version = "FrameAnnotator/0.1"

    @property
    def app_state(self) -> AppState:
        return self.server.app_state  # type: ignore[attr-defined]

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        if parsed.path.startswith("/data/"):
            requested = unquote(parsed.path.removeprefix("/data/"))
            target = (self.app_state.dataset_dir / requested).resolve()
            dataset_root = self.app_state.dataset_dir.resolve()
            if dataset_root not in target.parents and target != dataset_root:
                return str(dataset_root)
            return str(target)
        relative = parsed.path.lstrip("/") or "index.html"
        target = (self.app_state.ui_dir / relative).resolve()
        ui_root = self.app_state.ui_dir.resolve()
        if ui_root not in target.parents and target != ui_root:
            return str(ui_root / "index.html")
        return str(target)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/manifest":
            self.send_json(self.app_state.manifest)
            return
        if parsed.path == "/api/annotations":
            self.send_json(self.app_state.annotations)
            return
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "time": utc_now()})
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/annotations":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            self.app_state.save_annotations(payload)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_json(
            {
                "ok": True,
                "saved_to": str(self.app_state.annotations_path.resolve()),
                "updated_at": self.app_state.annotations["updated_at"],
            }
        )

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local frame annotator for coronary frames.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("p0001_unique"),
        help="Directory with images/ and masks/ subdirectories.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Path to JSON annotations file. Defaults to <dataset-dir>/manual_annotations.json",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    ui_dir = (Path(__file__).resolve().parent / "annotator").resolve()
    annotations_path = (args.annotations or (dataset_dir / "manual_annotations.json")).resolve()

    mimetypes.add_type("application/javascript", ".js")
    manifest = build_manifest(dataset_dir)
    annotations = load_annotations(annotations_path, manifest)
    state = AppState(
        dataset_dir=dataset_dir,
        annotations_path=annotations_path,
        ui_dir=ui_dir,
        manifest=manifest,
        annotations=annotations,
    )

    httpd = ThreadingHTTPServer((args.host, args.port), AnnotatorHandler)
    httpd.app_state = state  # type: ignore[attr-defined]
    print(f"Serving annotator on http://{args.host}:{args.port}")
    print(f"Dataset: {dataset_dir}")
    print(f"Annotations: {annotations_path}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
