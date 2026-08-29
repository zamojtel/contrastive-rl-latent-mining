"""Load CRL checkpoints and expose raw latent representations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import jax.numpy as jnp

from .crl import CRL, load_params
from .networks import Encoder


_REQUIRED_ENCODER_FIELDS = {
    "repr_dim",
    "h_dim",
    "n_hidden",
    "skip_connections",
    "use_relu",
    "use_ln",
}
_CHECKPOINT_PATTERN = re.compile(r"step_(\d+)\.pkl")


def _feature_array(values: Any, name: str) -> jnp.ndarray:
    array = jnp.asarray(values)
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one feature dimension")
    return array


def _dense_kernel_shape(
    variables: Mapping[str, Any],
    layer_name: str,
    component_name: str,
) -> tuple[int, int]:
    if not isinstance(variables, Mapping):
        raise ValueError(f"{component_name} must be a Flax variables mapping")

    parameters = variables.get("params")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{component_name} is missing the Flax 'params' collection")

    layer = parameters.get(layer_name)
    if not isinstance(layer, Mapping) or "kernel" not in layer:
        raise ValueError(f"{component_name} is missing {layer_name}/kernel")

    shape = getattr(layer["kernel"], "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{component_name} {layer_name}/kernel must be a matrix")

    return int(shape[0]), int(shape[1])


def _load_agent_from_metadata(
    metadata: Mapping[str, Any],
    metadata_path: Path,
) -> CRL:
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported metadata schema in {metadata_path}")

    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"Missing config in {metadata_path}")

    agent_config = config.get("agent")
    if not isinstance(agent_config, Mapping):
        raise ValueError(f"Missing agent config in {metadata_path}")

    if agent_config.get("type") != "CRL":
        raise ValueError(
            f"Expected CRL metadata, found {agent_config.get('type')!r}"
        )

    parameters = agent_config.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Missing CRL parameters in {metadata_path}")

    missing = sorted(_REQUIRED_ENCODER_FIELDS - set(parameters))
    if missing:
        raise ValueError(
            f"Missing CRL encoder parameters in {metadata_path}: {missing}"
        )

    try:
        return CRL(**dict(parameters))
    except TypeError as error:
        raise ValueError(f"Invalid CRL parameters in {metadata_path}") from error


def _read_run_metadata(run_dir: Path) -> tuple[dict[str, Any], CRL]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata does not exist: {metadata_path}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read run metadata: {metadata_path}") from error

    if not isinstance(metadata, dict):
        raise ValueError(f"Run metadata must be a JSON object: {metadata_path}")

    return metadata, _load_agent_from_metadata(metadata, metadata_path)


def _resolve_checkpoint_path(
    run_dir: Path,
    checkpoint: str | int,
) -> Path:
    if isinstance(checkpoint, bool):
        raise TypeError("checkpoint must be 'final', a file name, or an integer step")

    if isinstance(checkpoint, int):
        if checkpoint < 0:
            raise ValueError("checkpoint step must be non-negative")
        checkpoint_name = f"step_{checkpoint}.pkl"
    elif isinstance(checkpoint, str):
        checkpoint_name = checkpoint
    else:
        raise TypeError("checkpoint must be 'final', a file name, or an integer step")

    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint must be a file name inside the run ckpt directory")

    checkpoint_path = run_dir / "ckpt" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CRL checkpoint does not exist: {checkpoint_path}")

    return checkpoint_path


@dataclass(frozen=True)
class CRLLatentExtractor:
    """Raw CRL encoders reconstructed from a trusted run checkpoint."""

    encoder: Encoder
    sa_encoder_params: Any
    g_encoder_params: Any
    metadata: dict[str, Any]
    checkpoint_path: Path
    state_dim: int
    action_dim: int
    goal_dim: int
    repr_dim: int

    def split_observation(
        self,
        observation: Any,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Splits a goal-conditioned observation into state and desired goal."""
        observation = _feature_array(observation, "observation")
        expected_dim = self.state_dim + self.goal_dim

        if observation.shape[-1] != expected_dim:
            raise ValueError(
                "observation has feature dimension "
                f"{observation.shape[-1]}, expected {expected_dim}"
            )

        return (
            observation[..., : self.state_dim],
            observation[..., self.state_dim :],
        )

    def phi(self, state: Any, action: Any) -> jnp.ndarray:
        """Returns the raw phi(state, action) representation."""
        state = _feature_array(state, "state")
        action = _feature_array(action, "action")

        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "state and action must have identical leading dimensions"
            )
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state has feature dimension {state.shape[-1]}, "
                f"expected {self.state_dim}"
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action has feature dimension {action.shape[-1]}, "
                f"expected {self.action_dim}"
            )

        state_action = jnp.concatenate([state, action], axis=-1)
        return self.encoder.apply(self.sa_encoder_params, state_action)

    def psi(self, goal: Any) -> jnp.ndarray:
        """Returns the raw psi(goal) representation."""
        goal = _feature_array(goal, "goal")

        if goal.shape[-1] != self.goal_dim:
            raise ValueError(
                f"goal has feature dimension {goal.shape[-1]}, "
                f"expected {self.goal_dim}"
            )

        return self.encoder.apply(self.g_encoder_params, goal)


def list_crl_checkpoint_steps(run_dir: str | Path) -> list[int]:
    """Returns available step checkpoints in numeric order."""
    checkpoint_dir = Path(run_dir) / "ckpt"
    steps = []

    for path in checkpoint_dir.glob("step_*.pkl"):
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            steps.append(int(match.group(1)))

    return sorted(steps)


def load_crl_latent_extractor(
    run_dir: str | Path,
    checkpoint: str | int = "final",
) -> CRLLatentExtractor:
    """Loads raw CRL encoders from a trusted checkpoint and metadata file.

    Checkpoints use pickle internally. Never load artifacts from an untrusted
    source because pickle deserialization can execute arbitrary code.
    """
    run_dir = Path(run_dir)
    metadata, agent = _read_run_metadata(run_dir)
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint)

    try:
        checkpoint_params = load_params(str(checkpoint_path))
    except Exception as error:
        raise ValueError(
            f"Could not load trusted CRL checkpoint: {checkpoint_path}"
        ) from error

    if (
        not isinstance(checkpoint_params, (tuple, list))
        or len(checkpoint_params) != 3
    ):
        raise ValueError(
            f"CRL checkpoint must contain three parameter components: {checkpoint_path}"
        )

    _, actor_params, critic_params = checkpoint_params

    if not isinstance(critic_params, Mapping):
        raise ValueError(
            f"CRL critic parameters must be a mapping: {checkpoint_path}"
        )
    if "sa_encoder" not in critic_params or "g_encoder" not in critic_params:
        raise ValueError(
            "CRL critic parameters must contain sa_encoder and g_encoder"
        )

    sa_encoder_params = critic_params["sa_encoder"]
    g_encoder_params = critic_params["g_encoder"]

    encoder_final_layer = f"Dense_{agent.n_hidden}"
    actor_mean_layer = f"Dense_{agent.n_hidden}"

    sa_input_shape = _dense_kernel_shape(
        sa_encoder_params,
        "Dense_0",
        "sa_encoder",
    )
    goal_input_shape = _dense_kernel_shape(
        g_encoder_params,
        "Dense_0",
        "g_encoder",
    )
    sa_output_shape = _dense_kernel_shape(
        sa_encoder_params,
        encoder_final_layer,
        "sa_encoder",
    )
    goal_output_shape = _dense_kernel_shape(
        g_encoder_params,
        encoder_final_layer,
        "g_encoder",
    )
    actor_input_shape = _dense_kernel_shape(
        actor_params,
        "Dense_0",
        "actor",
    )
    actor_mean_shape = _dense_kernel_shape(
        actor_params,
        actor_mean_layer,
        "actor",
    )

    goal_dim = goal_input_shape[0]
    observation_dim = actor_input_shape[0]
    action_dim = actor_mean_shape[1]
    state_dim = observation_dim - goal_dim

    if state_dim <= 0:
        raise ValueError("Checkpoint implies a non-positive state dimension")
    if action_dim <= 0:
        raise ValueError("Checkpoint implies a non-positive action dimension")
    if sa_input_shape[0] != state_dim + action_dim:
        raise ValueError(
            "Checkpoint dimensions disagree between actor and sa_encoder"
        )

    sa_repr_dim = sa_output_shape[1]
    goal_repr_dim = goal_output_shape[1]
    if sa_repr_dim != goal_repr_dim or sa_repr_dim != agent.repr_dim:
        raise ValueError(
            "Checkpoint latent dimensions disagree with CRL metadata"
        )

    return CRLLatentExtractor(
        encoder=agent.make_encoder(),
        sa_encoder_params=sa_encoder_params,
        g_encoder_params=g_encoder_params,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        repr_dim=sa_repr_dim,
    )
