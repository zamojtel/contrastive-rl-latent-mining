import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgcrl.agents.crl import (
    CRL,
    list_crl_checkpoint_steps,
    load_crl_latent_extractor,
)
from jaxgcrl.agents.crl.crl import save_params
from jaxgcrl.agents.crl.networks import Actor


def write_test_run(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "ckpt"
    checkpoint_dir.mkdir(parents=True)

    state_dim = 4
    action_dim = 2
    goal_dim = 2

    agent = CRL(
        repr_dim=3,
        h_dim=8,
        n_hidden=1,
        skip_connections=0,
        use_relu=False,
        use_ln=False,
    )
    encoder = agent.make_encoder()
    actor = Actor(
        action_size=action_dim,
        network_width=agent.h_dim,
        network_depth=agent.n_hidden,
        skip_connections=agent.skip_connections,
        use_relu=agent.use_relu,
    )

    actor_params = actor.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, state_dim + goal_dim)),
    )
    sa_encoder_params = encoder.init(
        jax.random.PRNGKey(1),
        jnp.ones((1, state_dim + action_dim)),
    )
    g_encoder_params = encoder.init(
        jax.random.PRNGKey(2),
        jnp.ones((1, goal_dim)),
    )

    checkpoint_params = (
        {"log_alpha": jnp.asarray(0.0)},
        actor_params,
        {
            "sa_encoder": sa_encoder_params,
            "g_encoder": g_encoder_params,
        },
    )

    save_params(checkpoint_dir / "final", checkpoint_params)
    save_params(checkpoint_dir / "step_10.pkl", checkpoint_params)
    save_params(checkpoint_dir / "step_2.pkl", checkpoint_params)

    metadata = {
        "schema_version": 1,
        "config": {
            "agent": {
                "type": "CRL",
                "parameters": dict(vars(agent)),
            },
            "run": {
                "env": "simple_m1",
                "seed": 0,
            },
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return (
        run_dir,
        encoder,
        sa_encoder_params,
        g_encoder_params,
    )


def test_crl_latent_extractor_matches_training_encoder_path(tmp_path):
    (
        run_dir,
        encoder,
        sa_encoder_params,
        g_encoder_params,
    ) = write_test_run(tmp_path)

    extractor = load_crl_latent_extractor(run_dir)

    assert extractor.state_dim == 4
    assert extractor.action_dim == 2
    assert extractor.goal_dim == 2
    assert extractor.repr_dim == 3
    assert extractor.checkpoint_path.name == "final"

    states = jnp.asarray(
        [
            [1.0, 2.0, 0.1, 0.2],
            [3.0, 4.0, 0.3, 0.4],
        ]
    )
    actions = jnp.asarray(
        [
            [0.5, -0.5],
            [-0.25, 0.75],
        ]
    )
    goals = jnp.asarray(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    phi = extractor.phi(states, actions)
    psi = extractor.psi(goals)

    expected_phi = encoder.apply(
        sa_encoder_params,
        jnp.concatenate([states, actions], axis=-1),
    )
    expected_psi = encoder.apply(g_encoder_params, goals)

    assert phi.shape == (2, 3)
    assert psi.shape == (2, 3)
    np.testing.assert_allclose(phi, expected_phi, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(psi, expected_psi, rtol=1e-6, atol=1e-6)

    single_phi = extractor.phi(states[0], actions[0])
    np.testing.assert_allclose(single_phi, phi[0], rtol=1e-6, atol=1e-6)

    compiled_psi = jax.jit(lambda value: extractor.psi(value))(goals)
    np.testing.assert_allclose(compiled_psi, psi, rtol=1e-6, atol=1e-6)

    observations = jnp.concatenate([states, goals], axis=-1)
    split_states, split_goals = extractor.split_observation(observations)
    np.testing.assert_array_equal(split_states, states)
    np.testing.assert_array_equal(split_goals, goals)

    time_batched_phi = extractor.phi(
        states.reshape(1, 2, 4),
        actions.reshape(1, 2, 2),
    )
    assert time_batched_phi.shape == (1, 2, 3)


def test_checkpoint_steps_are_sorted_numerically_and_loadable(tmp_path):
    run_dir, _, _, _ = write_test_run(tmp_path)

    assert list_crl_checkpoint_steps(run_dir) == [2, 10]

    extractor = load_crl_latent_extractor(run_dir, checkpoint=2)

    assert extractor.checkpoint_path.name == "step_2.pkl"


def test_crl_latent_extractor_validates_input_shapes(tmp_path):
    run_dir, _, _, _ = write_test_run(tmp_path)
    extractor = load_crl_latent_extractor(run_dir)

    with pytest.raises(ValueError, match="state has feature dimension"):
        extractor.phi(
            jnp.zeros((2, 3)),
            jnp.zeros((2, 2)),
        )

    with pytest.raises(ValueError, match="action has feature dimension"):
        extractor.phi(
            jnp.zeros((2, 4)),
            jnp.zeros((2, 3)),
        )

    with pytest.raises(ValueError, match="identical leading dimensions"):
        extractor.phi(
            jnp.zeros((2, 4)),
            jnp.zeros((3, 2)),
        )

    with pytest.raises(ValueError, match="goal has feature dimension"):
        extractor.psi(jnp.zeros((2, 3)))

    with pytest.raises(ValueError, match="observation has feature dimension"):
        extractor.split_observation(jnp.zeros((2, 5)))


def test_crl_latent_extractor_rejects_invalid_artifacts(tmp_path):
    run_dir, _, _, _ = write_test_run(tmp_path)
    metadata_path = run_dir / "metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"]["agent"]["type"] = "PPO"
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected CRL metadata"):
        load_crl_latent_extractor(run_dir)

    metadata["config"]["agent"]["type"] = "CRL"
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    save_params(
        run_dir / "ckpt" / "broken.pkl",
        ("alpha", "actor", {}),
    )

    with pytest.raises(ValueError, match="sa_encoder and g_encoder"):
        load_crl_latent_extractor(
            run_dir,
            checkpoint="broken.pkl",
        )
