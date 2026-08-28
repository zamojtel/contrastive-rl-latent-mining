import csv

import jax.numpy as jnp
import pytest

from jaxgcrl.utils import env as env_utils


def read_metrics_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


def test_metrics_recorder_writes_fixed_csv_without_external_logging(
    tmp_path,
    monkeypatch,
):
    def unexpected_render(*args, **kwargs):
        pytest.fail("Rendering should be disabled.")

    def unexpected_wandb_log(*args, **kwargs):
        pytest.fail("WandB logging should be disabled.")

    monkeypatch.setattr(env_utils, "render", unexpected_render)
    monkeypatch.setattr(env_utils.wandb, "log", unexpected_wandb_log)

    recorder = env_utils.MetricsRecorder(
        total_env_steps=100,
        metrics_to_collect=["metric/a", "metric/b"],
        exp_dir=tmp_path,
        exp_name="test",
        mode="online",
        render_enabled=False,
        wandb_enabled=False,
    )

    recorder.progress(
        10,
        {
            "metric/a": jnp.asarray(1.25),
            "ignored": 99.0,
        },
        make_policy=None,
        params=None,
        env=None,
        do_render=True,
    )
    recorder.progress(
        20,
        {
            "metric/a": 2.5,
            "metric/b": jnp.asarray(3.5),
        },
        make_policy=None,
        params=None,
        env=None,
        do_render=True,
    )

    fieldnames, rows = read_metrics_csv(recorder.metrics_path)

    assert fieldnames == ["step", "metric/a", "metric/b"]
    assert rows == [
        {
            "step": "10",
            "metric/a": "1.25",
            "metric/b": "0",
        },
        {
            "step": "20",
            "metric/a": "2.5",
            "metric/b": "3.5",
        },
    ]


def test_metrics_recorder_preserves_rendering_by_default(
    tmp_path,
    monkeypatch,
):
    render_calls = []

    monkeypatch.setattr(
        env_utils,
        "render",
        lambda *args, **kwargs: render_calls.append((args, kwargs)),
    )

    recorder = env_utils.MetricsRecorder(
        total_env_steps=10,
        metrics_to_collect=["metric"],
        exp_dir=tmp_path,
        exp_name="test",
        mode="online",
        wandb_enabled=False,
    )

    recorder.progress(
        1,
        {"metric": 1.0},
        make_policy=object(),
        params=object(),
        env=object(),
    )

    assert len(render_calls) == 1


def test_metrics_recorder_truncates_csv_for_a_new_run(tmp_path):
    first = env_utils.MetricsRecorder(
        total_env_steps=10,
        metrics_to_collect=["metric"],
        exp_dir=tmp_path,
        exp_name="test",
        mode="online",
        render_enabled=False,
        wandb_enabled=False,
    )
    first.progress(
        1,
        {"metric": 1.0},
        make_policy=None,
        params=None,
        env=None,
        do_render=False,
    )

    second = env_utils.MetricsRecorder(
        total_env_steps=10,
        metrics_to_collect=["metric"],
        exp_dir=tmp_path,
        exp_name="test",
        mode="online",
        render_enabled=False,
        wandb_enabled=False,
    )

    fieldnames, rows = read_metrics_csv(second.metrics_path)

    assert fieldnames == ["step", "metric"]
    assert rows == []
