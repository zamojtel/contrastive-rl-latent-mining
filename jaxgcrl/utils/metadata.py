"""Run provenance collection and durable JSON persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import jax


SOURCE_ROOT = Path(__file__).resolve().parents[2]

PACKAGE_NAMES = (
    "jax",
    "jaxlib",
    "brax",
    "flax",
    "mujoco",
    "mujoco-mjx",
    "numpy",
    "scipy",
    "tyro",
    "wandb",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=str)]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass

    return str(value)


def _git_output(
    repo_dir: str | os.PathLike[str],
    *args: str,
) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_git_metadata(
    repo_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    repo_root = SOURCE_ROOT if repo_dir is None else Path(repo_dir)
    status = _git_output(repo_root, "status", "--porcelain")

    return {
        "commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "branch": _git_output(repo_root, "symbolic-ref", "--short", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def collect_package_versions() -> dict[str, str | None]:
    versions = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def collect_runtime_metadata() -> dict[str, Any]:
    devices = jax.devices()

    return {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "jax": {
            "backend": jax.default_backend(),
            "process_count": jax.process_count(),
            "local_device_count": jax.local_device_count(),
            "devices": [
                {
                    "id": _json_safe(device.id),
                    "platform": device.platform,
                    "device_kind": device.device_kind,
                    "process_index": device.process_index,
                    "repr": str(device),
                }
                for device in devices
            ],
        },
    }


def build_run_metadata(
    config: Any,
    *,
    repo_dir: str | os.PathLike[str] | None = None,
    command: Sequence[str] | None = None,
    derived: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    git_metadata = collect_git_metadata(repo_dir)
    argv = list(sys.argv if command is None else command)

    return {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "argv": argv,
        "config": {
            "run": _json_safe(vars(config.run)),
            "agent": {
                "type": type(config.agent).__name__,
                "parameters": _json_safe(vars(config.agent)),
            },
        },
        "derived": _json_safe({} if derived is None else derived),
        "git": git_metadata,
        "runtime": collect_runtime_metadata(),
        "packages": collect_package_versions(),
    }


def write_run_metadata(
    run_dir: str | os.PathLike[str],
    config: Any,
    *,
    repo_dir: str | os.PathLike[str] | None = None,
    command: Sequence[str] | None = None,
    derived: Mapping[str, Any] | None = None,
) -> Path:
    metadata = build_run_metadata(
        config,
        repo_dir=repo_dir,
        command=command,
        derived=derived,
    )

    output_path = Path(run_dir) / "metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    replacing_existing = output_path.exists()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    if replacing_existing:
        logging.warning("Replaced existing run metadata: %s", output_path)
    if metadata["git"]["dirty"] is True:
        logging.warning(
            "The Git worktree is dirty; this run cannot be reconstructed from its commit alone."
        )

    return output_path
