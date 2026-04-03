"""
MindustryServer — manages the server-release.jar subprocess lifecycle.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


READY_SENTINEL = "[Mimi Gateway] Aguardando conex"
STARTUP_TIMEOUT = 30


class MindustryServer:
    """Spawns and manages a Mindustry dedicated server process."""

    def __init__(
        self,
        jar_path: str = "server-release.jar",
        data_dir: str = "rl/server_data",
        port: int = 9000,
        game_port: Optional[int] = None,
        java_args: Optional[list[str]] = None,
    ) -> None:
        self._jar_path = Path(jar_path)
        self._data_dir = Path(data_dir)
        self._port = port
        extra_args: list[str] = []
        if game_port is not None:
            extra_args.append(f"-Dmindustry.port={game_port}")
        self._java_args = (java_args or []) + extra_args
        self._proc: Optional[subprocess.Popen] = None
        self._ready = threading.Event()
        self._failed = threading.Event()

    def start(self, timeout: float = STARTUP_TIMEOUT) -> None:
        """Spawn the server and block until it signals ready."""
        if not self._jar_path.exists():
            raise FileNotFoundError(f"Server jar not found: {self._jar_path}")

        self._kill_port_holder()

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._ready.clear()
        self._failed.clear()

        cmd = ["java"] + self._java_args + ["-jar", str(self._jar_path.resolve())]
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._data_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
        )

        t = threading.Thread(target=self._monitor_stdout, daemon=True)
        t.start()

        start_time = time.time()
        while not self._ready.wait(timeout=0.1):
            if self._failed.is_set():
                self.stop()
                raise RuntimeError("Server process exited before becoming ready (check server logs)")
            if time.time() - start_time > timeout:
                self.stop()
                raise TimeoutError(f"Server did not start within {timeout}s")

        def _send_fps_hint(self_ref=self) -> None:
            time.sleep(1.0)
            self_ref.send_stdin("fps 120")
        threading.Thread(target=_send_fps_hint, daemon=True).start()

    def stop(self) -> None:
        """Terminate the server process."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def send_stdin(self, cmd: str) -> None:
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write((cmd + "\n").encode())
                self._proc.stdin.flush()
            except Exception:
                pass

    def _kill_port_holder(self) -> None:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{self._port}"],
                capture_output=True, text=True, timeout=5,
            )
            pids = [p for p in result.stdout.strip().split() if p.isdigit()]
            for pid_str in pids:
                try:
                    os.kill(int(pid_str), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if pids:
                time.sleep(1.0)
        except Exception:
            pass

    def _monitor_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        ready_seen = False
        try:
            for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                print(f"[server] {text}")
                if not ready_seen and READY_SENTINEL in text:
                    self._ready.set()
                    ready_seen = True
            if not ready_seen:
                exit_code = proc.poll()
                print(f"[server] Process exited with code {exit_code} before emitting ready sentinel")
                self._failed.set()
        except Exception:
            pass


def start_n_servers(
    n: int,
    base_tcp_port: int = 9000,
    base_game_port: int = 6567,
    base_data_dir: str = "rl/server_data",
    jar_path: str = "server-release.jar",
    mod_zip: str = "mimi-gateway-v1.0.4.zip",
) -> list[MindustryServer]:
    """Start N Mindustry server instances in parallel and return them all ready."""
    import shutil as _shutil
    from pathlib import Path as _Path

    servers: list[MindustryServer] = []
    for i in range(n):
        inst_dir = _Path(base_data_dir) / f"instance_{i}"
        mods_dir = inst_dir / "config" / "mods"
        mods_dir.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(mod_zip, mods_dir / "mimi-gateway.zip")
        (inst_dir / "mimi_port.txt").write_text(str(base_tcp_port + i))
        server = MindustryServer(
            jar_path=jar_path,
            data_dir=str(inst_dir),
            port=base_tcp_port + i,
            game_port=base_game_port + i,
        )
        servers.append(server)

    errors: list[Exception] = []
    threads: list[threading.Thread] = []

    def _start(srv: MindustryServer) -> None:
        try:
            srv.start()
        except Exception as e:
            errors.append(e)

    for srv in servers:
        t = threading.Thread(target=_start, args=(srv,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if errors:
        for srv in servers:
            try:
                srv.stop()
            except Exception:
                pass
        raise RuntimeError(f"Failed to start {len(errors)} server(s): {errors}")

    return servers
