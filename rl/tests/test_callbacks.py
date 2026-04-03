import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import pytest

from rl.callbacks.training_callbacks import _write_metrics_json, LiveMetricsCallback


def test_write_metrics_json_creates_file(tmp_path):
    dest = tmp_path / "metrics.json"
    _write_metrics_json(dest, {"key": "value", "num": 42})
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["key"] == "value"
    assert data["num"] == 42


def test_write_metrics_json_overwrites(tmp_path):
    dest = tmp_path / "metrics.json"
    _write_metrics_json(dest, {"v": 1})
    _write_metrics_json(dest, {"v": 2})
    data = json.loads(dest.read_text())
    assert data["v"] == 2


def _make_callback(tmp_path) -> LiveMetricsCallback:
    cb = LiveMetricsCallback(
        metrics_path=str(tmp_path / "live_metrics.json"),
        verbose=0,
    )
    cb.model = MagicMock()
    return cb


def test_live_metrics_callback_step_accumulates_latency(tmp_path):
    cb = _make_callback(tmp_path)
    cb.locals = {"infos": [{"step_latency_ms": 12.5}, {"step_latency_ms": 7.0}]}
    cb._on_step()
    assert list(cb._step_latencies) == [12.5, 7.0]


def test_live_metrics_callback_step_accumulates_resources(tmp_path):
    cb = _make_callback(tmp_path)
    cb.locals = {"infos": [{"resources": {"copper": 150, "lead": 30}}]}
    cb._on_step()
    assert cb._last_resources == {"copper": 150.0, "lead": 30.0}


def test_live_metrics_callback_writes_on_rollout_end(tmp_path):
    cb = _make_callback(tmp_path)
    metrics_file = tmp_path / "live_metrics.json"

    buf = MagicMock()
    buf.actions = np.zeros((256, 1, 2), dtype=np.int32)
    buf.values = np.ones((256, 1))
    buf.action_masks = np.ones((256, 1, 16), dtype=bool)
    cb.model.rollout_buffer = buf
    cb.num_timesteps = 1024

    cb._on_rollout_end()

    assert metrics_file.exists()
    data = json.loads(metrics_file.read_text())
    assert "policy" in data
    assert "world" in data
    assert "pipeline" in data
    assert "training" in data
    assert data["training"]["total_timesteps"] == 1024
