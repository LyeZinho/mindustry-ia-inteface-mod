"""Unit tests for MimiClient — no live Mindustry required."""
import json
import socket
from unittest.mock import MagicMock, patch
import pytest
from rl.env.mimi_client import MimiClient


def make_mock_socket(json_payload: dict) -> MagicMock:
    """Return a mock socket whose makefile().readline() yields payload."""
    raw = (json.dumps(json_payload) + "\n").encode()
    mock_file = MagicMock()
    mock_file.readline.return_value = raw.decode()
    mock_sock = MagicMock()
    mock_sock.makefile.return_value = mock_file
    return mock_sock


def test_receive_state_parses_json():
    """receive_state() returns parsed dict from newline-delimited JSON."""
    payload = {"tick": 1, "wave": 3, "core": {"hp": 0.9, "x": 10, "y": 10}}
    client = MimiClient.__new__(MimiClient)
    client._sock = make_mock_socket(payload)
    client._file = client._sock.makefile()
    state = client.receive_state()
    assert state["wave"] == 3
    assert state["core"]["hp"] == pytest.approx(0.9)


def test_send_command_appends_newline():
    """send_command() sends cmd + newline as UTF-8 bytes."""
    client = MimiClient.__new__(MimiClient)
    client._sock = MagicMock()
    client.send_command("BUILD;duo;15;20;0")
    client._sock.send.assert_called_once_with(b"BUILD;duo;15;20;0\n")


def test_send_build():
    """build() sends correct BUILD command string."""
    client = MimiClient.__new__(MimiClient)
    client._sock = MagicMock()
    client.build("duo", 15, 20, rotation=0)
    client._sock.send.assert_called_once_with(b"BUILD;duo;15;20;0\n")


def test_send_move_unit():
    client = MimiClient.__new__(MimiClient)
    client._sock = MagicMock()
    client.move_unit(unit_id=5, x=10, y=12)
    client._sock.send.assert_called_once_with(b"UNIT_MOVE;5;10;12\n")


def test_send_attack():
    client = MimiClient.__new__(MimiClient)
    client._sock = MagicMock()
    client.attack(unit_id=3, x=25, y=30)
    client._sock.send.assert_called_once_with(b"ATTACK;3;25;30\n")


def test_send_factory():
    client = MimiClient.__new__(MimiClient)
    client._sock = MagicMock()
    client.spawn_unit(factory_x=10, factory_y=12, unit_type="poly")
    client._sock.send.assert_called_once_with(b"FACTORY;10;12;poly\n")


def test_send_repair():
    client = MimiClient.__new__(MimiClient)
    client._sock = MagicMock()
    client.repair(x=11, y=21)
    client._sock.send.assert_called_once_with(b"REPAIR;11;21\n")


def test_receive_state_returns_none_on_empty():
    """receive_state() returns None when server sends empty line."""
    client = MimiClient.__new__(MimiClient)
    mock_file = MagicMock()
    mock_file.readline.return_value = ""
    client._sock = MagicMock()
    client._file = mock_file
    assert client.receive_state() is None
