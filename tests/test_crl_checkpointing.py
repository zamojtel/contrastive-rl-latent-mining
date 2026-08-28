from types import SimpleNamespace

import numpy as np

from jaxgcrl.agents.crl.crl import (
    _extract_params,
    load_params,
    save_params,
)


def test_extract_params_preserves_component_order():
    training_state = SimpleNamespace(
        alpha_state=SimpleNamespace(params="alpha"),
        actor_state=SimpleNamespace(params="actor"),
        critic_state=SimpleNamespace(params="critic"),
    )

    assert _extract_params(training_state) == (
        "alpha",
        "actor",
        "critic",
    )


def test_save_params_creates_parent_and_round_trips(tmp_path):
    path = tmp_path / "nested" / "checkpoints" / "params.pkl"
    params = (
        {"alpha": np.asarray(0.5)},
        {"actor": np.arange(4, dtype=np.float32)},
        {"critic": np.eye(2, dtype=np.float32)},
    )

    save_params(path, params)

    assert path.is_file()

    loaded = load_params(path)

    assert loaded[0]["alpha"] == params[0]["alpha"]
    np.testing.assert_array_equal(
        loaded[1]["actor"],
        params[1]["actor"],
    )
    np.testing.assert_array_equal(
        loaded[2]["critic"],
        params[2]["critic"],
    )
