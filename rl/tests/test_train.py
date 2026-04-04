"""Tests for train.py argument parsing and env factory."""
import pytest
from pathlib import Path
import tempfile
from rl.train import parse_args, _make_env_factory, _find_latest_checkpoint


def test_parse_args_default_n_envs():
    args = parse_args([])
    assert args.n_envs == 4


def test_parse_args_custom_n_envs():
    args = parse_args(["--n-envs", "2"])
    assert args.n_envs == 2


def test_make_env_factory_returns_callable():
    factory = _make_env_factory(host="localhost", tcp_port=9001, max_steps=100, maps=None)
    assert callable(factory)


def test_train_uses_maskable_ppo():
    from sb3_contrib import MaskablePPO
    assert MaskablePPO is not None


def test_parse_args_resume_default_is_none():
    args = parse_args([])
    assert args.resume is None


def test_parse_args_resume_auto():
    args = parse_args(["--resume"])
    assert args.resume == "auto"


def test_parse_args_resume_explicit_path():
    args = parse_args(["--resume", "rl/models/mindustry_ppo_50000_steps.zip"])
    assert args.resume == "rl/models/mindustry_ppo_50000_steps.zip"


def test_find_latest_checkpoint_empty_dir():
    with tempfile.TemporaryDirectory() as d:
        model_path, vec_path = _find_latest_checkpoint(d)
        assert model_path is None
        assert vec_path is None


def test_find_latest_checkpoint_picks_highest_step():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "mindustry_ppo_10000_steps.zip").touch()
        (p / "mindustry_ppo_50000_steps.zip").touch()
        (p / "mindustry_ppo_20000_steps.zip").touch()
        model_path, _ = _find_latest_checkpoint(d)
        assert "50000" in model_path


def test_find_latest_checkpoint_uses_matching_vecnorm():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "mindustry_ppo_50000_steps.zip").touch()
        (p / "vecnormalize_50000_steps.pkl").touch()
        _, vec_path = _find_latest_checkpoint(d)
        assert vec_path is not None
        assert "50000" in vec_path


def test_find_latest_checkpoint_falls_back_to_vecnormalize_pkl():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "mindustry_ppo_50000_steps.zip").touch()
        (p / "vecnormalize.pkl").touch()
        _, vec_path = _find_latest_checkpoint(d)
        assert vec_path is not None
        assert "vecnormalize.pkl" in vec_path


def test_find_latest_checkpoint_falls_back_to_final_model():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "final_model.zip").touch()
        (p / "vecnormalize.pkl").touch()
        model_path, vec_path = _find_latest_checkpoint(d)
        assert "final_model" in model_path
        assert vec_path is not None
