"""Inspect optional dependencies for future real LeRobot export."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
import shutil
import sys
from typing import Any


DEPENDENCY_ORDER = [
    "python",
    "pyarrow",
    "pandas",
    "lerobot",
    "cv2",
    "imageio",
    "imageio_ffmpeg",
    "PIL",
    "ffmpeg",
]

IMPLEMENTED_VIDEO_ENCODER_KEYS = ("imageio_ffmpeg", "imageio", "cv2")
IMPLEMENTED_VIDEO_ENCODER_REQUIREMENT = (
    "requires one implemented video encoder: imageio_ffmpeg, imageio, or cv2"
)


def _version_from_distribution(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_from_module(module: Any, distribution_name: str | None = None) -> str | None:
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version
    if distribution_name is not None:
        return _version_from_distribution(distribution_name)
    return None


def _check_importable(module_name: str, distribution_name: str | None = None) -> dict[str, object]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # Import can fail from missing transitive dependencies too.
        return {
            "available": False,
            "version": None,
            "detail": f"not importable on this environment: {exc}",
        }

    version = _version_from_module(module, distribution_name)
    detail = f"imported module {module_name}"
    if distribution_name is not None:
        detail += f"; distribution {distribution_name}"
    return {"available": True, "version": version, "detail": detail}


def _check_python() -> dict[str, object]:
    return {
        "available": True,
        "version": platform.python_version(),
        "detail": f"{sys.executable} on {platform.platform()}",
    }


def _check_ffmpeg() -> dict[str, object]:
    path = shutil.which("ffmpeg")
    if path is not None:
        return {"available": True, "version": None, "detail": f"ffmpeg command found at {path}"}

    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "detail": f"ffmpeg command not found on PATH and imageio_ffmpeg is unavailable: {exc}",
        }

    version = _version_from_distribution("imageio-ffmpeg")
    return {
        "available": True,
        "version": version,
        "detail": f"ffmpeg command not on PATH; using imageio_ffmpeg executable at {ffmpeg_exe}",
    }


def _check_imageio_ffmpeg() -> dict[str, object]:
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "detail": f"not importable or get_ffmpeg_exe failed on this environment: {exc}",
        }

    return {
        "available": True,
        "version": _version_from_module(imageio_ffmpeg, "imageio-ffmpeg"),
        "detail": f"imageio_ffmpeg.get_ffmpeg_exe returned {ffmpeg_exe}",
    }


def check_export_dependencies() -> dict[str, dict[str, object]]:
    """Return availability details for optional real-export dependencies."""

    return {
        "python": _check_python(),
        "pyarrow": _check_importable("pyarrow"),
        "pandas": _check_importable("pandas"),
        "lerobot": _check_importable("lerobot"),
        "cv2": _check_importable("cv2", "opencv-python"),
        "imageio": _check_importable("imageio"),
        "imageio_ffmpeg": _check_imageio_ffmpeg(),
        "PIL": _check_importable("PIL", "Pillow"),
        "ffmpeg": _check_ffmpeg(),
    }


def implemented_video_backend_ready(dependencies: dict[str, dict[str, object]]) -> bool:
    """Return whether at least one implemented video encoder is available."""

    return any(
        isinstance(dependencies.get(key), dict) and dependencies[key].get("available") is True
        for key in IMPLEMENTED_VIDEO_ENCODER_KEYS
    )


def _format_version(version: object) -> str:
    if version is None:
        return "unknown"
    return str(version)


def print_dependency_report(results: dict[str, dict[str, object]]) -> None:
    """Print a readable dependency report."""

    print("Real Export Dependency Preflight")
    print("This command is read-only and does not install packages.")
    print("Missing laptop dependencies are not final ForceVLA compatibility blockers.")
    print("Run this same command later on the lab workstation inside the validated ForceVLA environment.")
    print("")

    for key in DEPENDENCY_ORDER:
        entry = results[key]
        status = "available" if entry["available"] else "missing"
        print(f"{key}: {status}")
        print(f"  version: {_format_version(entry['version'])}")
        print(f"  detail: {entry['detail']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect optional dependencies for real LeRobot export.")
    parser.parse_args(argv)

    print_dependency_report(check_export_dependencies())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
