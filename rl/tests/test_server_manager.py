"""Tests for MindustryServer process manager."""
import subprocess
import threading
import time
import pytest
from unittest.mock import patch, MagicMock
from rl.server.manager import MindustryServer


def test_is_running_false_before_start():
    server = MindustryServer(jar_path="server-release.jar")
    assert server.is_running() is False


def test_stop_noop_when_not_running():
    """stop() on an unstarted server should not raise."""
    server = MindustryServer(jar_path="server-release.jar")
    server.stop()  # must not raise


def test_start_raises_if_jar_not_found():
    server = MindustryServer(jar_path="/nonexistent/server.jar")
    with pytest.raises(FileNotFoundError):
        server.start(timeout=1)


def test_start_detects_ready_line(tmp_path):
    """start() returns when stdout emits the ready sentinel."""
    server = MindustryServer(jar_path="fake.jar", data_dir=str(tmp_path))

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None  # still running
    fake_proc.stdout = MagicMock()

    # Simulate stdout: noise then ready line
    lines = [b"[INFO] Loading...\n", b"[INFO] Server loaded\n"]
    fake_proc.stdout.__iter__ = lambda self: iter(lines)

    with patch("subprocess.Popen", return_value=fake_proc):
        with patch("pathlib.Path.exists", return_value=True):
            server.start(timeout=5)

    assert server.is_running() is True


def test_stop_terminates_process(tmp_path):
    """stop() terminates the subprocess."""
    server = MindustryServer(jar_path="fake.jar", data_dir=str(tmp_path))

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.__iter__ = lambda self: iter([b"[INFO] Server loaded\n"])

    with patch("subprocess.Popen", return_value=fake_proc):
        with patch("pathlib.Path.exists", return_value=True):
            server.start(timeout=5)

    server.stop()
    fake_proc.terminate.assert_called_once()
    assert server.is_running() is False
