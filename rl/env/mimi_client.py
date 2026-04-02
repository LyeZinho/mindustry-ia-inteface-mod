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
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)


class MimiClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        timeout: float = 10.0,
        _sock: Optional[socket.socket] = None,
    ) -> None:
        if _sock is not None:
            self._sock = _sock
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(timeout)
            self._sock.connect((host, port))
        self._file = self._sock.makefile("r", encoding="utf-8")

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
