"""Export CRL representations to portable NumPy archives."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import numpy as np

from .latents import CRLLatentExtractor


_FORMAT_NAME = "jaxgcrl.crl_latents"
_SCHEMA_VERSION = 1
_PHI_ACTIONS = {
    "zero",
    "deterministic_policy",
}


def _real_numeric_observations(
    values: Any,
    expected_dim: int,
) -> np.ndarray:
    try:
        array = np.asarray(values)
    except Exception as error:
        raise TypeError(
            "observations must be a real numeric matrix"
        ) from error

    if array.ndim != 2:
        raise ValueError(
            "observations must have shape (samples, features)"
        )
    if array.shape[0] == 0:
        raise ValueError(
            "observations must contain at least one sample"
        )
    if array.shape[1] != expected_dim:
        raise ValueError(
            "observations have feature dimension "
            f"{array.shape[1]}, expected {expected_dim}"
        )
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(
            "observations must be a real numeric matrix"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(
            array,
            dtype=np.float32,
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "observations must remain finite in float32"
        )

    return array


def _output_matrix(
    values: Any,
    name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(
            np.asarray(values),
            dtype=np.float32,
        )

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {array.shape}, "
            f"expected {expected_shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} must contain only finite values"
        )

    return array


def _metadata_mapping(
    value: Any,
) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _archive_metadata(
    extractor: CRLLatentExtractor,
    *,
    samples: int,
    phi_action: str,
) -> dict[str, Any]:
    metadata = _metadata_mapping(extractor.metadata)
    config = _metadata_mapping(metadata.get("config"))
    run = _metadata_mapping(config.get("run"))
    git = _metadata_mapping(metadata.get("git"))

    return {
        "format": _FORMAT_NAME,
        "schema_version": _SCHEMA_VERSION,
        "phi_action": phi_action,
        "latent_normalization": "none",
        "checkpoint": {
            "name": extractor.checkpoint_path.name,
            "sha256": extractor.checkpoint_sha256,
        },
        "dimensions": {
            "samples": samples,
            "state": extractor.state_dim,
            "action": extractor.action_dim,
            "goal": extractor.goal_dim,
            "latent": extractor.repr_dim,
        },
        "provenance": {
            "env": run.get("env"),
            "seed": run.get("seed"),
            "maze_size_scaling": run.get(
                "maze_size_scaling"
            ),
            "git_commit": git.get("commit"),
        },
    }


def export_crl_latents_npz(
    extractor: CRLLatentExtractor,
    observations: Any,
    output_path: str | Path,
    *,
    phi_action: Literal[
        "zero",
        "deterministic_policy",
    ],
    overwrite: bool = False,
) -> Path:
    """Exports raw CRL latents without pickle or object arrays.

    The ``phi_action`` argument is required because the two modes
    answer different questions:

    - ``zero`` isolates the state-action encoder at a fixed action.
    - ``deterministic_policy`` uses tanh(mean(state, goal)).
    """
    if phi_action not in _PHI_ACTIONS:
        raise ValueError(
            "phi_action must be 'zero' or "
            "'deterministic_policy'"
        )
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".npz":
        raise ValueError(
            "output_path must use the .npz suffix"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    expected_observation_dim = (
        extractor.state_dim + extractor.goal_dim
    )
    observation_array = _real_numeric_observations(
        observations,
        expected_observation_dim,
    )

    sample_count = observation_array.shape[0]
    states = np.ascontiguousarray(
        observation_array[:, : extractor.state_dim]
    )
    goals = np.ascontiguousarray(
        observation_array[:, extractor.state_dim :]
    )

    if phi_action == "zero":
        actions = np.zeros(
            (sample_count, extractor.action_dim),
            dtype=np.float32,
        )
    else:
        actions = _output_matrix(
            extractor.deterministic_action(
                observation_array
            ),
            "actions",
            (sample_count, extractor.action_dim),
        )

        if np.any(np.abs(actions) > 1.0 + 1e-6):
            raise ValueError(
                "deterministic policy actions must be in [-1, 1]"
            )

    phi = _output_matrix(
        extractor.phi(states, actions),
        "phi",
        (sample_count, extractor.repr_dim),
    )
    psi = _output_matrix(
        extractor.psi(goals),
        "psi",
        (sample_count, extractor.repr_dim),
    )

    archive_metadata = _archive_metadata(
        extractor,
        samples=sample_count,
        phi_action=phi_action,
    )
    metadata_json = json.dumps(
        archive_metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

    arrays = {
        "schema_version": np.asarray(
            _SCHEMA_VERSION,
            dtype=np.int32,
        ),
        "sample_index": np.arange(
            sample_count,
            dtype=np.int64,
        ),
        "states": states,
        "goals": goals,
        "actions": actions,
        "phi": phi,
        "psi": psi,
        "metadata_json": np.asarray(
            metadata_json,
            dtype=np.str_,
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez_compressed(
                output,
                **arrays,
            )
            output.flush()
            os.fsync(output.fileno())

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}"
            )

        os.replace(
            temporary_path,
            output_path,
        )
    finally:
        temporary_path.unlink(missing_ok=True)

    return output_path
