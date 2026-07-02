"""Read-only dependency doctor for the direct LeRobot v2.1 exporter."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from typing import Any

from doosan_forcevla_data.convert.processed_to_lerobot_v21 import PYARROW_EXTRA_HELP


def _module_version(module_name: str, distribution_name: str | None = None) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if isinstance(version, str):
        return version
    if distribution_name is None:
        distribution_name = module_name
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_command(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None:
        return {"available": False, "path": None, "version": None, "detail": f"{name} not found on PATH"}
    version = None
    try:
        completed = subprocess.run([name, "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        first = (completed.stdout or completed.stderr).splitlines()[0]
        version = first.strip()
    except Exception as exc:
        return {"available": True, "path": path, "version": None, "detail": f"found but version probe failed: {exc}"}
    return {"available": True, "path": path, "version": version, "detail": "found on PATH"}


def check_lerobot_v21_dependencies() -> dict[str, Any]:
    pyarrow_version = _module_version("pyarrow")
    return {
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "detail": platform.platform(),
        },
        "pyarrow": {
            "available": pyarrow_version is not None,
            "version": pyarrow_version,
            "detail": "required for Parquet read/write" if pyarrow_version else f"missing. {PYARROW_EXTRA_HELP}",
        },
        "ffmpeg": _check_command("ffmpeg"),
        "ffprobe": _check_command("ffprobe"),
        "forbidden_runtime_dependencies": {
            "lerobot_imported": "lerobot" in sys.modules,
            "numpy_imported": "numpy" in sys.modules,
            "PIL_imported": "PIL" in sys.modules,
            "cv2_imported": "cv2" in sys.modules,
        },
    }


def print_dependency_report(report: dict[str, Any]) -> None:
    print("LeRobot v2.1 Exporter Dependency Doctor")
    for key in ["python", "pyarrow", "ffmpeg", "ffprobe"]:
        entry = report[key]
        status = "available" if entry.get("available") else "missing"
        print(f"{key}: {status}")
        print(f"  version: {entry.get('version')}")
        print(f"  detail: {entry.get('detail')}")
    print("Forbidden runtime dependency imports")
    for key, value in report["forbidden_runtime_dependencies"].items():
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check dependencies for the direct LeRobot v2.1 exporter.")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args(argv)
    report = check_lerobot_v21_dependencies()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_dependency_report(report)
    return 0 if all(report[key].get("available") for key in ["pyarrow", "ffmpeg", "ffprobe"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
