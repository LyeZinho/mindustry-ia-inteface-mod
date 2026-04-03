"""Tests for train.py argument parsing and env factory."""
import pytest
from unittest.mock import patch, MagicMock
from rl.train import parse_args, _make_env_factory


def test_parse_args_default_n_envs():
    args = parse_args([])
    assert args.n_envs == 4


def test_parse_args_custom_n_envs():
    args = parse_args(["--n-envs", "2"])
    assert args.n_envs == 2


def test_make_env_factory_returns_callable():
    factory = _make_env_factory(host="localhost", tcp_port=9001, max_steps=100, maps=None)
    assert callable(factory)
