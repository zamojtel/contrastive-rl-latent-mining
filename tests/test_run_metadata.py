from dataclasses import dataclass
import json
from pathlib import Path

import jaxgcrl.utils.metadata as metadata_utils


@dataclass
class DummyRun:
    env: str = "simple_m1"
    seed: int = 3
    checkpoint_logdir: Path | None = None


@dataclass
class DummyAgent:
    repr_dim: int = 16
    n_hidden: int = 2


@dataclass
class DummyConfig:
    run: DummyRun
    agent: DummyAgent


def test_write_run_metadata_records_readable_provenance(
    tmp_path,
    monkeypatch,
):
    commit = "0123456789abcdef0123456789abcdef01234567"

    monkeypatch.setattr(
        metadata_utils,
        "_utc_now",
        lambda: "2026-08-29T12:00:00Z",
    )
    monkeypatch.setattr(
        metadata_utils,
        "collect_git_metadata",
        lambda repo_dir=None: {
            "commit": commit,
            "branch": "project/setup",
            "dirty": False,
        },
    )
    monkeypatch.setattr(
        metadata_utils,
        "collect_runtime_metadata",
        lambda: {
            "python": {"version": "3.11.13"},
            "platform": "test-platform",
            "jax": {
                "backend": "gpu",
                "devices": [{"device_kind": "NVIDIA L4"}],
            },
        },
    )
    monkeypatch.setattr(
        metadata_utils,
        "collect_package_versions",
        lambda: {"jax": "0.5.2", "jaxlib": "0.5.1"},
    )

    checkpoint_dir = tmp_path / "ckpt"
    config = DummyConfig(
        run=DummyRun(checkpoint_logdir=checkpoint_dir),
        agent=DummyAgent(),
    )
    run_dir = tmp_path / "run"

    metadata_path = metadata_utils.write_run_metadata(
        run_dir,
        config,
        repo_dir=tmp_path,
        command=["python", "run.py", "crl", "--seed", "3"],
        derived={"utd_ratio": 1.0},
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata_path == run_dir / "metadata.json"
    assert payload["schema_version"] == 1
    assert payload["created_at_utc"] == "2026-08-29T12:00:00Z"
    assert payload["argv"] == ["python", "run.py", "crl", "--seed", "3"]
    assert payload["config"]["run"]["checkpoint_logdir"] == str(checkpoint_dir)
    assert payload["config"]["agent"] == {
        "type": "DummyAgent",
        "parameters": {
            "n_hidden": 2,
            "repr_dim": 16,
        },
    }
    assert payload["derived"] == {"utd_ratio": 1.0}
    assert payload["git"]["commit"] == commit
    assert payload["runtime"]["jax"]["backend"] == "gpu"
    assert payload["packages"]["jax"] == "0.5.2"
    assert list(run_dir.glob(".metadata.json.*.tmp")) == []


def test_collect_git_metadata_preserves_detached_and_failed_states(
    tmp_path,
    monkeypatch,
):
    responses = {
        ("rev-parse", "HEAD"): "a" * 40,
        ("symbolic-ref", "--short", "HEAD"): None,
        ("status", "--porcelain"): "?? untracked.txt",
    }

    monkeypatch.setattr(
        metadata_utils,
        "_git_output",
        lambda repo_dir, *args: responses[args],
    )

    metadata = metadata_utils.collect_git_metadata(tmp_path)

    assert metadata == {
        "commit": "a" * 40,
        "branch": None,
        "dirty": True,
    }

    responses[("status", "--porcelain")] = None
    assert metadata_utils.collect_git_metadata(tmp_path)["dirty"] is None


def test_collect_package_versions_keeps_missing_entries(monkeypatch):
    def fake_version(package_name):
        if package_name == "jax":
            return "0.5.2"
        raise metadata_utils.importlib_metadata.PackageNotFoundError(package_name)

    monkeypatch.setattr(
        metadata_utils.importlib_metadata,
        "version",
        fake_version,
    )

    versions = metadata_utils.collect_package_versions()

    assert versions["jax"] == "0.5.2"
    assert versions["jaxlib"] is None
    assert set(versions) == set(metadata_utils.PACKAGE_NAMES)
