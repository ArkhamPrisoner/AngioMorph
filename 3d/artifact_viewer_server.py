#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cgi
import json
import math
import mimetypes
import shutil
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = ROOT / "remote_artifacts" / "20260422"
STATIC_DIR = ROOT / "artifact_viewer"
ALLOWED_SUFFIXES = {".ply", ".npz", ".npy"}
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
UPLOAD_DIR_NAME = "_uploads"


@dataclass
class Artifact:
    rel_path: str
    kind: str
    size_bytes: int
    summary: dict[str, Any]


def load_volume(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if "occupancy" in data.files:
            return data["occupancy"]
        if data.files:
            return data[data.files[0]]
        raise ValueError(f"Empty npz: {path}")
    return np.load(path)


def summarize_volume(volume: np.ndarray) -> dict[str, Any]:
    vol = np.asarray(volume)
    summary = {
        "shape": list(vol.shape),
        "dtype": str(vol.dtype),
        "min": float(vol.min()),
        "max": float(vol.max()),
    }
    if vol.size:
        quantiles = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
        values = np.quantile(vol.astype(np.float64), quantiles).tolist()
        summary["quantiles"] = {str(q): float(v) for q, v in zip(quantiles, values)}
    return summary


def detect_artifact(path: Path, artifact_root: Path) -> Artifact | None:
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return None
    rel_path = path.relative_to(artifact_root).as_posix()
    if suffix == ".ply":
        return Artifact(rel_path=rel_path, kind="ply", size_bytes=path.stat().st_size, summary={})
    try:
        volume = load_volume(path)
        summary = summarize_volume(volume)
        return Artifact(rel_path=rel_path, kind="volume", size_bytes=path.stat().st_size, summary=summary)
    except Exception as exc:
        return Artifact(
            rel_path=rel_path,
            kind="broken",
            size_bytes=path.stat().st_size,
            summary={"error": str(exc)},
        )


def serialize_artifact(item: Artifact) -> dict[str, Any]:
    return {
        "rel_path": item.rel_path,
        "kind": item.kind,
        "size_bytes": item.size_bytes,
        "size_mb": round(item.size_bytes / (1024 * 1024), 3),
        "summary": item.summary,
        "url": f"/artifacts/{quote(item.rel_path)}",
    }


def build_manifest(artifact_root: Path) -> dict[str, Any]:
    artifacts: list[Artifact] = []
    for path in sorted(artifact_root.rglob("*")):
        if not path.is_file():
            continue
        artifact = detect_artifact(path, artifact_root)
        if artifact is not None:
            artifacts.append(artifact)
    return {
        "artifact_root": str(artifact_root.resolve()),
        "artifacts": [serialize_artifact(item) for item in artifacts],
    }


def sample_points(volume: np.ndarray, threshold: float, max_points: int) -> dict[str, Any]:
    vol = np.asarray(volume)
    mask = vol >= threshold
    coords = np.argwhere(mask)
    values = vol[mask].astype(np.float32, copy=False)
    total_points = int(coords.shape[0])
    if total_points == 0:
        return {
            "shape": list(vol.shape),
            "threshold": threshold,
            "num_points_total": 0,
            "num_points_returned": 0,
            "points": [],
            "min": float(vol.min()),
            "max": float(vol.max()),
        }

    if total_points > max_points:
        step = math.ceil(total_points / max_points)
        coords = coords[::step]
        values = values[::step]
    shape = np.array(vol.shape, dtype=np.float32)
    center = (shape - 1.0) / 2.0
    scale = float(shape.max()) or 1.0
    xyz = (coords.astype(np.float32) - center) / scale

    points = [
        [float(p[2]), float(-p[1]), float(p[0]), float(v)]
        for p, v in zip(xyz, values, strict=False)
    ]
    return {
        "shape": list(vol.shape),
        "threshold": float(threshold),
        "num_points_total": total_points,
        "num_points_returned": len(points),
        "points": points,
        "min": float(vol.min()),
        "max": float(vol.max()),
    }


class ArtifactViewerHandler(SimpleHTTPRequestHandler):
    server_version = "ArtifactViewer/0.2"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/manifest":
            self.send_json(self.server.manifest)
            return
        if parsed.path == "/api/volume_meta":
            self.handle_volume_meta(parsed)
            return
        if parsed.path == "/api/volume_points":
            self.handle_volume_points(parsed)
            return
        if parsed.path.startswith("/artifacts/"):
            self.handle_artifact_file(parsed.path)
            return
        if parsed.path in {"/", "/index.html"}:
            self.path = "/index.html"
            return super().do_GET()
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self.handle_upload()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unsupported endpoint")

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean_path = parsed.path.lstrip("/")
        return str((STATIC_DIR / clean_path).resolve())

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def artifact_path(self, rel_path: str) -> Path:
        candidate = (self.server.artifact_root / rel_path).resolve()
        root = self.server.artifact_root.resolve()
        if root not in candidate.parents and candidate != root:
            raise PermissionError(rel_path)
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(rel_path)
        return candidate

    def handle_artifact_file(self, path: str) -> None:
        rel_path = unquote(path[len("/artifacts/"):])
        try:
            candidate = self.artifact_path(rel_path)
        except (PermissionError, FileNotFoundError):
            self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found")
            return
        mime_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        data = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_volume_meta(self, parsed) -> None:
        rel_path = parse_qs(parsed.query).get("path", [""])[0]
        if not rel_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing path")
            return
        try:
            candidate = self.artifact_path(rel_path)
            volume = load_volume(candidate)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
            return
        self.send_json({"rel_path": rel_path, "summary": summarize_volume(volume)})

    def handle_volume_points(self, parsed) -> None:
        query = parse_qs(parsed.query)
        rel_path = query.get("path", [""])[0]
        if not rel_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing path")
            return
        threshold_raw = query.get("threshold", [None])[0]
        max_points_raw = query.get("max_points", ["120000"])[0]
        try:
            candidate = self.artifact_path(rel_path)
            volume = load_volume(candidate)
            vol_min = float(volume.min())
            vol_max = float(volume.max())
            if threshold_raw is None or threshold_raw == "":
                threshold = float(np.quantile(volume.astype(np.float64), 0.995))
            else:
                threshold = float(threshold_raw)
            threshold = max(min(threshold, vol_max), vol_min)
            max_points = max(1000, min(int(max_points_raw), 500000))
            payload = sample_points(volume, threshold=threshold, max_points=max_points)
            payload["rel_path"] = rel_path
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
            return
        self.send_json(payload)

    def handle_upload(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            self.send_json({"error": "Пустой запрос"}, status=400)
            return
        if content_length > MAX_UPLOAD_BYTES:
            self.send_json({"error": f"Файл слишком большой. Лимит {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}, status=413)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": str(content_length),
            },
        )
        if "file" not in form:
            self.send_json({"error": "Файл не передан"}, status=400)
            return

        file_field = form["file"]
        filename = Path(file_field.filename or "").name
        if not filename:
            self.send_json({"error": "Не удалось определить имя файла"}, status=400)
            return
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            self.send_json({"error": f"Поддерживаются только: {', '.join(sorted(ALLOWED_SUFFIXES))}"}, status=400)
            return

        upload_dir = self.server.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(filename).stem
        safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in stem).strip("._") or "artifact"
        target = upload_dir / f"{safe_stem}_{int(time.time())}{suffix}"
        with target.open("wb") as handle:
            shutil.copyfileobj(file_field.file, handle)

        artifact = detect_artifact(target, self.server.artifact_root)
        if artifact is None:
            target.unlink(missing_ok=True)
            self.send_json({"error": "Не удалось обработать загруженный файл"}, status=400)
            return

        self.server.refresh_manifest()
        self.send_json({
            "ok": True,
            "artifact": serialize_artifact(artifact),
            "manifest": self.server.manifest,
        })


class ArtifactViewerServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], artifact_root: Path):
        super().__init__(server_address, ArtifactViewerHandler)
        self.artifact_root = artifact_root.resolve()
        self.upload_dir = self.artifact_root / UPLOAD_DIR_NAME
        self.manifest = build_manifest(self.artifact_root)

    def refresh_manifest(self) -> None:
        self.manifest = build_manifest(self.artifact_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local 3D reconstruction artifacts in a browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.artifact_root.exists():
        raise FileNotFoundError(args.artifact_root)
    server = ArtifactViewerServer((args.host, args.port), args.artifact_root)
    print(f"Artifact viewer: http://{args.host}:{args.port}")
    print(f"Artifact root: {args.artifact_root.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
