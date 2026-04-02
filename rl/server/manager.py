"""
MindustryServer — manages the server-release.jar subprocess lifecycle.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


READY_SENTINEL = "Server loaded"
STARTUP_TIMEOUT = 30


class MindustryServer:
    """Spawns and manages a Mindustry dedicated server process."""

    def __init__(
        self,
        jar_path: str = "server-release.jar",
        data_dir: str = "rl/server_data",
        java_args: Optional[list[str]] = None,
    ) -> None:
        self._jar_path = Path(jar_path)
        self._data_dir = Path(data_dir)
        self._java_args = java_args or []
        self._proc: Optional[subprocess.Popen] = None
        self._ready = threading.Event()
        self._failed = threading.Event()

    def start(self, timeout: float = STARTUP_TIMEOUT) -> None:
        """Spawn the server and block until it signals ready."""
        if not self._jar_path.exists():
            raise FileNotFoundError(f"Server jar not found: {self._jar_path}")

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

    def _monitor_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                print(f"[server] {text}")
                if READY_SENTINEL in text:
                    self._ready.set()
                    return
            exit_code = proc.poll()
            print(f"[server] Process exited with code {exit_code} before emitting ready sentinel")
            self._failed.set()
        except Exception:
            pass
