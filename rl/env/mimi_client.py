"""
MimiClient — TCP client for the Mimi Gateway Mindustry mod.

Connects to localhost:9000 by default. Reads newline-delimited JSON state
frames from the server and sends newline-terminated command strings.

The constructor accepts an optional pre-built socket for unit testing.
"""
from __future__ import annotations

import json
import logging
import socket
import time
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)

_MAX_CONNECT_RETRIES = 10
_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 30.0


class MimiClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        timeout: float = 30.0,
        _sock: Optional[socket.socket] = None,
    ) -> None:
        if _sock is not None:
            self._sock = _sock
        else:
            self._sock = self._connect_with_retry(host, port, timeout)
        self._file = self._sock.makefile("r", encoding="utf-8")

    @staticmethod
    def _connect_with_retry(
        host: str, port: int, timeout: float
    ) -> socket.socket:
        backoff = _INITIAL_BACKOFF
        for attempt in range(1, _MAX_CONNECT_RETRIES + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                sock.connect((host, port))
                if attempt > 1:
                    _log.info("Connected to %s:%s on attempt %d", host, port, attempt)
                return sock
            except OSError as exc:
                sock.close()
                if attempt == _MAX_CONNECT_RETRIES:
                    raise
                _log.warning(
                    "Connect to %s:%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    host, port, attempt, _MAX_CONNECT_RETRIES, exc, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)
        raise RuntimeError("unreachable")

    def receive_state(self) -> Optional[Dict[str, Any]]:
        line = self._file.readline()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            _log.warning("Malformed JSON from server (%.200r): %s", line, exc)
            return None

    def send_command(self, cmd: str) -> None:
        self._sock.send(f"{cmd}\n".encode("utf-8"))

    def build(self, block: str, x: int, y: int, rotation: int = 0) -> None:
        self.send_command(f"BUILD;{block};{x};{y};{rotation}")

    def move_unit(self, unit_id: int, x: int, y: int) -> None:
        self.send_command(f"UNIT_MOVE;{unit_id};{x};{y}")

    def attack(self, unit_id: int, x: int, y: int) -> None:
        self.send_command(f"ATTACK;{unit_id};{x};{y}")

    def spawn_unit(self, factory_x: int, factory_y: int, unit_type: str = "poly") -> None:
        self.send_command(f"FACTORY;{factory_x};{factory_y};{unit_type}")

    def repair(self, x: int, y: int) -> None:
        self.send_command(f"REPAIR;{x};{y}")

    def delete(self, x: int, y: int) -> None:
        self.send_command(f"DELETE;{x};{y}")

    def stop(self, unit_id: Optional[int] = None) -> None:
        if unit_id is not None:
            self.send_command(f"STOP;{unit_id}")
        else:
            self.send_command("STOP")

    def message(self, text: str) -> None:
        self.send_command(f"MSG;{text}")

    def close(self) -> None:
        self._file.close()
        self._sock.close()
