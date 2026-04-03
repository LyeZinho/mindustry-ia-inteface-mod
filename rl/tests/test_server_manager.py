"""Tests for MindustryServer process manager."""
import subprocess
import threading
import time
import pytest
from unittest.mock import patch, MagicMock
from rl.server.manager import MindustryServer, start_n_servers


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
    lines = [b"[INFO] Loading...\n", b"[Mimi Gateway] Aguardando conexao...\n"]
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
    fake_proc.stdout.__iter__ = lambda self: iter([b"[Mimi Gateway] Aguardando conexao...\n"])

    with patch("subprocess.Popen", return_value=fake_proc):
        with patch("pathlib.Path.exists", return_value=True):
            server.start(timeout=5)

    server.stop()
    fake_proc.terminate.assert_called_once()
    assert server.is_running() is False


def test_start_n_servers_returns_n_instances(tmp_path):
    """start_n_servers creates N MindustryServer objects."""
    mod_zip = tmp_path / "mimi-gateway-v1.0.4.zip"
    mod_zip.write_bytes(b"fake")

    started = []

    def fake_start(self, timeout=30):
        started.append(self)
        self._proc = MagicMock()
        self._proc.poll.return_value = None
        self._ready.set()

    with patch.object(MindustryServer, "start", fake_start):
        servers = start_n_servers(
            n=3,
            base_tcp_port=9100,
            base_game_port=6700,
            base_data_dir=str(tmp_path / "servers"),
            jar_path="fake.jar",
            mod_zip=str(mod_zip),
        )

    assert len(servers) == 3
    assert len(started) == 3


def test_start_n_servers_writes_port_files(tmp_path):
    """Each instance's data dir gets a mimi_port.txt with the right port."""
    mod_zip = tmp_path / "mimi-gateway-v1.0.4.zip"
    mod_zip.write_bytes(b"fake")

    def fake_start(self, timeout=30):
        self._proc = MagicMock()
        self._proc.poll.return_value = None
        self._ready.set()

    with patch.object(MindustryServer, "start", fake_start):
        start_n_servers(
            n=2,
            base_tcp_port=9200,
            base_game_port=6800,
            base_data_dir=str(tmp_path / "servers"),
            jar_path="fake.jar",
            mod_zip=str(mod_zip),
        )

    port_file_0 = tmp_path / "servers" / "instance_0" / "mimi_port.txt"
    port_file_1 = tmp_path / "servers" / "instance_1" / "mimi_port.txt"
    assert port_file_0.read_text().strip() == "9200"
    assert port_file_1.read_text().strip() == "9201"


def test_game_port_passed_as_jvm_arg(tmp_path):
    """MindustryServer with game_port passes -Dmindustry.port to JVM."""
    server = MindustryServer(
        jar_path="fake.jar",
        data_dir=str(tmp_path),
        port=9000,
        game_port=6570,
    )
    # game_port should appear in _java_args
    assert any("6570" in arg for arg in server._java_args)
