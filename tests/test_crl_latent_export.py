import hashlib
import importlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgcrl.agents.crl import (
    CRL,
    export_crl_latents_npz,
    load_crl_latent_extractor,
)
from jaxgcrl.agents.crl.crl import save_params


EXPECTED_KEYS = {
    "schema_version",
    "sample_index",
    "states",
    "goals",
    "actions",
    "phi",
    "psi",
    "metadata_json",
}


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
    actor = agent.make_actor(action_dim)

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

    save_params(
        checkpoint_dir / "final",
        (
            {"log_alpha": jnp.asarray(0.0)},
            actor_params,
            {
                "sa_encoder": sa_encoder_params,
                "g_encoder": g_encoder_params,
            },
        ),
    )

    metadata = {
        "schema_version": 1,
        "config": {
            "agent": {
                "type": "CRL",
                "parameters": dict(vars(agent)),
            },
            "run": {
                "env": "simple_m1_\u2603",
                "seed": 7,
                "maze_size_scaling": 2.0,
            },
        },
        "git": {
            "commit": "abc123",
            "branch": "test",
            "dirty": False,
        },
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return run_dir


def observations():
    values = np.asarray(
        [
            [1.0, 2.0, 0.1, 0.2, 5.0, 6.0],
            [3.0, 4.0, 0.3, 0.4, 7.0, 8.0],
        ],
        dtype=np.float64,
    )

    padded = np.zeros((2, 12), dtype=np.float64)
    padded[:, ::2] = values
    result = padded[:, ::2]

    assert not result.flags.c_contiguous
    return result


def read_metadata(archive):
    metadata_array = archive["metadata_json"]

    assert metadata_array.shape == ()
    assert metadata_array.dtype.kind == "U"

    return json.loads(metadata_array.item())


def test_zero_action_export_is_portable_and_complete(tmp_path):
    run_dir = write_test_run(tmp_path)
    extractor = load_crl_latent_extractor(run_dir)
    input_observations = observations()

    output_path = (
        tmp_path / "nested" / "zero_latents.npz"
    )

    result = export_crl_latents_npz(
        extractor,
        input_observations,
        output_path,
        phi_action="zero",
    )

    assert result == output_path
    assert output_path.is_file()

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        assert set(archive.files) == EXPECTED_KEYS

        for key in archive.files:
            assert archive[key].dtype.kind != "O"

        assert archive["schema_version"].shape == ()
        assert archive["schema_version"].dtype == np.int32
        assert archive["schema_version"].item() == 1

        np.testing.assert_array_equal(
            archive["sample_index"],
            np.arange(2, dtype=np.int64),
        )
        np.testing.assert_allclose(
            archive["states"],
            input_observations[:, :4].astype(np.float32),
        )
        np.testing.assert_allclose(
            archive["goals"],
            input_observations[:, 4:].astype(np.float32),
        )
        np.testing.assert_array_equal(
            archive["actions"],
            np.zeros((2, 2), dtype=np.float32),
        )

        expected_phi = extractor.phi(
            input_observations[:, :4].astype(np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )
        expected_psi = extractor.psi(
            input_observations[:, 4:].astype(np.float32)
        )

        np.testing.assert_allclose(
            archive["phi"],
            expected_phi,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            archive["psi"],
            expected_psi,
            rtol=1e-6,
            atol=1e-6,
        )

        assert archive["states"].dtype == np.float32
        assert archive["goals"].dtype == np.float32
        assert archive["actions"].dtype == np.float32
        assert archive["phi"].dtype == np.float32
        assert archive["psi"].dtype == np.float32

        metadata = read_metadata(archive)

    expected_sha256 = hashlib.sha256(
        (run_dir / "ckpt/final").read_bytes()
    ).hexdigest()

    assert metadata == {
        "format": "jaxgcrl.crl_latents",
        "schema_version": 1,
        "phi_action": "zero",
        "latent_normalization": "none",
        "checkpoint": {
            "name": "final",
            "sha256": expected_sha256,
        },
        "dimensions": {
            "samples": 2,
            "state": 4,
            "action": 2,
            "goal": 2,
            "latent": 3,
        },
        "provenance": {
            "env": "simple_m1_\u2603",
            "seed": 7,
            "maze_size_scaling": 2.0,
            "git_commit": "abc123",
        },
    }

    assert str(tmp_path) not in json.dumps(metadata)


def test_policy_export_matches_deterministic_actor(tmp_path):
    run_dir = write_test_run(tmp_path)
    extractor = load_crl_latent_extractor(run_dir)
    input_observations = observations()

    first_path = tmp_path / "policy_first.npz"
    second_path = tmp_path / "policy_second.npz"

    export_crl_latents_npz(
        extractor,
        input_observations,
        first_path,
        phi_action="deterministic_policy",
    )
    export_crl_latents_npz(
        extractor,
        input_observations,
        second_path,
        phi_action="deterministic_policy",
    )

    expected_actions = np.asarray(
        extractor.deterministic_action(
            input_observations.astype(np.float32)
        )
    )
    expected_phi = np.asarray(
        extractor.phi(
            input_observations[:, :4].astype(np.float32),
            expected_actions,
        )
    )

    with (
        np.load(first_path, allow_pickle=False) as first,
        np.load(second_path, allow_pickle=False) as second,
    ):
        np.testing.assert_allclose(
            first["actions"],
            expected_actions,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            first["phi"],
            expected_phi,
            rtol=1e-6,
            atol=1e-6,
        )

        for key in EXPECTED_KEYS:
            np.testing.assert_array_equal(
                first[key],
                second[key],
            )

        metadata = read_metadata(first)

    assert metadata["phi_action"] == (
        "deterministic_policy"
    )
    assert np.abs(expected_actions).max() <= 1.0


@pytest.mark.parametrize(
    "bad_observations",
    [
        np.zeros(6),
        np.zeros((1, 1, 6)),
        np.empty((0, 6)),
        np.zeros((2, 5)),
        np.full((2, 6), np.nan),
        np.full((2, 6), np.inf),
        np.full((2, 6), 1e100),
        np.full((2, 6), "text"),
        np.full((2, 6), 1, dtype=object),
        np.ones((2, 6), dtype=np.complex64),
        np.ones((2, 6), dtype=np.bool_),
    ],
    ids=[
        "rank-one",
        "rank-three",
        "empty",
        "wrong-features",
        "nan",
        "infinity",
        "float32-overflow",
        "string",
        "object",
        "complex",
        "boolean",
    ],
)
def test_export_rejects_invalid_observations(
    tmp_path,
    bad_observations,
):
    extractor = load_crl_latent_extractor(
        write_test_run(tmp_path)
    )
    output_path = tmp_path / "invalid.npz"

    with pytest.raises((TypeError, ValueError)):
        export_crl_latents_npz(
            extractor,
            bad_observations,
            output_path,
            phi_action="zero",
        )

    assert not output_path.exists()


def test_export_validates_arguments_and_overwrite(tmp_path):
    extractor = load_crl_latent_extractor(
        write_test_run(tmp_path)
    )
    input_observations = observations()

    with pytest.raises(ValueError, match="phi_action"):
        export_crl_latents_npz(
            extractor,
            input_observations,
            tmp_path / "bad_mode.npz",
            phi_action="sample",
        )

    with pytest.raises(ValueError, match=".npz suffix"):
        export_crl_latents_npz(
            extractor,
            input_observations,
            tmp_path / "latents.bin",
            phi_action="zero",
        )

    output_path = tmp_path / "latents.npz"

    export_crl_latents_npz(
        extractor,
        input_observations,
        output_path,
        phi_action="zero",
    )
    original_bytes = output_path.read_bytes()

    with pytest.raises(FileExistsError):
        export_crl_latents_npz(
            extractor,
            input_observations,
            output_path,
            phi_action="deterministic_policy",
        )

    assert output_path.read_bytes() == original_bytes

    export_crl_latents_npz(
        extractor,
        input_observations,
        output_path,
        phi_action="deterministic_policy",
        overwrite=True,
    )

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        assert read_metadata(archive)["phi_action"] == (
            "deterministic_policy"
        )

    assert not list(
        tmp_path.glob(".latents.npz.*.tmp")
    )
    assert not list(
        tmp_path.glob("*.tmp.npz")
    )


def test_atomic_failure_preserves_existing_target(
    tmp_path,
    monkeypatch,
):
    export_module = importlib.import_module(
        "jaxgcrl.agents.crl.latent_export"
    )
    extractor = load_crl_latent_extractor(
        write_test_run(tmp_path)
    )
    output_path = tmp_path / "existing.npz"
    output_path.write_bytes(b"original")

    def failing_save(output, **arrays):
        output.write(b"partial")
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(
        export_module.np,
        "savez_compressed",
        failing_save,
    )

    with pytest.raises(
        RuntimeError,
        match="injected write failure",
    ):
        export_module.export_crl_latents_npz(
            extractor,
            observations(),
            output_path,
            phi_action="zero",
            overwrite=True,
        )

    assert output_path.read_bytes() == b"original"
    assert not list(
        tmp_path.glob(".existing.npz.*.tmp")
    )
    assert not list(
        tmp_path.glob("*.tmp.npz")
    )


def test_export_uses_checksum_of_loaded_checkpoint(
    tmp_path,
):
    run_dir = write_test_run(tmp_path)
    checkpoint_path = run_dir / "ckpt/final"

    expected_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    extractor = load_crl_latent_extractor(run_dir)

    assert extractor.checkpoint_sha256 == expected_sha256

    checkpoint_path.write_bytes(
        b"checkpoint replaced after loading"
    )

    assert hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest() != expected_sha256

    output_path = tmp_path / "checksum.npz"

    export_crl_latents_npz(
        extractor,
        observations(),
        output_path,
        phi_action="zero",
    )

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        metadata = read_metadata(archive)

    assert metadata["checkpoint"]["sha256"] == (
        expected_sha256
    )
