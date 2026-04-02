# Server Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the RL training pipeline from Mindustry client mode to dedicated server mode (`server-release.jar`), with `train.py` managing the server process and the mod handling episode resets via a new `RESET;[mapname]` TCP command.

**Architecture:** `train.py` spawns `server-release.jar` as a subprocess, monitors stdout to detect readiness, then connects the existing `MimiClient` TCP channel. The Mimi Gateway mod gains a `RESET;[mapname]` command that reloads the map on the game thread. `MindustryEnv.reset()` sends this command and cycles through a configurable map list.

**Tech Stack:** Python 3, subprocess, threading, stable-baselines3, Gymnasium, Rhino JS (Mindustry mod)

---

## Task 1: `MindustryServer` class — process lifecycle management

**Files:**
- Create: `rl/server/__init__.py`
- Create: `rl/server/manager.py`
- Create: `rl/tests/test_server_manager.py`

**Step 1: Write the failing tests**

```python
# rl/tests/test_server_manager.py
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
```

**Step 2: Run tests to confirm they fail**

```bash
cd /home/pedro/repo/mindustry-ia-inteface-mod
source rl/venv/bin/activate.fish
python -m pytest rl/tests/test_server_manager.py -v
```
Expected: `ModuleNotFoundError: No module named 'rl.server'`

**Step 3: Create `rl/server/__init__.py`**

```python
# rl/server/__init__.py
```
(empty file)

**Step 4: Implement `rl/server/manager.py`**

```python
"""
MindustryServer — manages the server-release.jar subprocess lifecycle.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


# Sentinel string in server stdout that signals the server is ready
READY_SENTINEL = "Server loaded"
STARTUP_TIMEOUT = 30  # seconds


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

    def start(self, timeout: float = STARTUP_TIMEOUT) -> None:
        """Spawn the server and block until it signals ready."""
        if not self._jar_path.exists():
            raise FileNotFoundError(f"Server jar not found: {self._jar_path}")

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._ready.clear()

        cmd = ["java"] + self._java_args + ["-jar", str(self._jar_path.resolve())]
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._data_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
        )

        # Monitor stdout in daemon thread
        t = threading.Thread(target=self._monitor_stdout, daemon=True)
        t.start()

        if not self._ready.wait(timeout=timeout):
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
        try:
            for line in self._proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                print(f"[server] {text}")
                if READY_SENTINEL in text:
                    self._ready.set()
        except Exception:
            pass
```

**Step 5: Run tests to confirm they pass**

```bash
python -m pytest rl/tests/test_server_manager.py -v
```
Expected: 5 passed

**Step 6: Run full suite to confirm no regressions**

```bash
python -m pytest rl/tests/ -v
```
Expected: 36 passed (31 old + 5 new)

**Step 7: Commit**

```bash
git add rl/server/__init__.py rl/server/manager.py rl/tests/test_server_manager.py
git commit -m "feat: add MindustryServer subprocess manager"
```

---

## Task 2: `RESET` command in `scripts/main.js`

**Files:**
- Modify: `scripts/main.js`

No Python tests for this task — tested manually with the live server.

**Step 1: Add `RESET` to `validateCommand`**

Find this line in `validateCommand`:
```javascript
let validCommands = ["BUILD", "UNIT_MOVE", "MSG", "ATTACK", "STOP", "FACTORY", "REPAIR", "DELETE", "UPGRADE"];
```
Replace with:
```javascript
let validCommands = ["BUILD", "UNIT_MOVE", "MSG", "ATTACK", "STOP", "FACTORY", "REPAIR", "DELETE", "UPGRADE", "RESET"];
```

**Step 2: Add `RESET` case to `processCommand` switch**

Find the switch block. Add before the closing `}`:
```javascript
            case "RESET":
                handleResetCommand(parts);
                break;
```

**Step 3: Add `handleResetCommand` function**

Add this function before `// PHASE 5: Event Triggers`:

```javascript
function handleResetCommand(parts) {
    let mapName = parts[1] ? parts[1].trim() : null;
    
    Log.info("[Mimi Gateway] RESET solicitado" + (mapName ? ": " + mapName : " (mapa padrão)"));
    
    // Must execute map load on the game thread
    Core.app.post(() => {
        try {
            let map = null;
            
            if (mapName != null) {
                // Try exact name match first, then file name match
                map = Maps.all().find(m => 
                    m.name() === mapName || 
                    (m.file != null && m.file.nameWithoutExtension() === mapName)
                );
            }
            
            if (map == null) {
                map = Maps.all().first();
                if (map != null) {
                    Log.info("[Mimi Gateway] Mapa '" + mapName + "' não encontrado, usando: " + map.name());
                }
            }
            
            if (map == null) {
                Log.err("[Mimi Gateway] RESET: nenhum mapa disponível");
                return;
            }
            
            let rules = map.applyRules(Gamemode.survival);
            Vars.logic.reset();
            Vars.world.loadMap(map, rules);
            Vars.state.set(State.playing);
            Vars.logic.play();
            
            // Reset tick counter so state is sent promptly after load
            tickCounter = config.updateInterval;
            
            Log.info("[Mimi Gateway] RESET completo: " + map.name());
        } catch (e) {
            Log.err("[Mimi Gateway] Erro no RESET: " + e);
            if (config.debug) Log.err(e.stack);
        }
    });
}
```

**Step 4: Verify no syntax errors** — read the file and check visually that all braces are balanced.

**Step 5: Commit**

```bash
git add scripts/main.js
git commit -m "feat: add RESET;[mapname] command to mod protocol"
```

---

## Task 3: Update `MindustryEnv` to send RESET and cycle maps

**Files:**
- Modify: `rl/env/mindustry_env.py`
- Modify: `rl/tests/test_env.py`

**Step 1: Write new/updated failing tests**

Add these tests to `rl/tests/test_env.py`:

```python
DEFAULT_MAPS = ["Ancient Caldera", "Windswept Islands"]


def test_reset_sends_reset_command_with_map():
    """reset() sends RESET;[mapname] to cycle through maps."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client, maps=DEFAULT_MAPS)
    env.reset()
    client.send_command.assert_called_with("RESET;Ancient Caldera")


def test_reset_cycles_maps_on_successive_resets():
    """Successive reset() calls cycle through the maps list."""
    client = MagicMock()
    client.receive_state.return_value = MOCK_STATE
    env = MindustryEnv(client=client, maps=DEFAULT_MAPS)
    
    env.reset()
    client.send_command.assert_called_with("RESET;Ancient Caldera")
    
    env.reset()
    client.send_command.assert_called_with("RESET;Windswept Islands")
    
    env.reset()  # wraps around
    client.send_command.assert_called_with("RESET;Ancient Caldera")


def test_reset_uses_default_maps_when_none_provided():
    """reset() works without explicit maps list (uses built-in defaults)."""
    client = make_mock_client(states=[MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()  # should not raise
    client.send_command.assert_called_once()
    call_arg = client.send_command.call_args[0][0]
    assert call_arg.startswith("RESET;")
```

**Step 2: Run tests to confirm they fail**

```bash
python -m pytest rl/tests/test_env.py::test_reset_sends_reset_command_with_map rl/tests/test_env.py::test_reset_cycles_maps_on_successive_resets rl/tests/test_env.py::test_reset_uses_default_maps_when_none_provided -v
```
Expected: FAIL (TypeError or AttributeError — `maps` param doesn't exist yet)

**Step 3: Update `MimiClient` to expose `send_command`**

Check `rl/env/mimi_client.py` — if `send_command` is private/named differently, expose it or add a public alias. The tests mock it directly, so ensure the method is called `send_command`.

**Step 4: Update `rl/env/mindustry_env.py`**

```python
# Add to top of file
DEFAULT_TRAINING_MAPS = [
    "Ancient Caldera",
    "Windswept Islands",
    "Tarpit Depths",
    "Craters",
    "Fungal Pass",
    "Nuclear Complex",
]
```

Update `__init__`:
```python
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        max_steps: int = 5000,
        client: Optional[MimiClient] = None,
        maps: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.observation_space = make_obs_space()
        self.action_space = make_action_space()
        self.max_steps = max_steps

        self._host = host
        self._port = port
        self._client: Optional[MimiClient] = client
        self._maps: list[str] = maps if maps is not None else DEFAULT_TRAINING_MAPS
        self._map_index: int = 0

        self._prev_state: Optional[Dict[str, Any]] = None
        self._step_count: int = 0
```

Update `reset()`:
```python
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        if self._client is None:
            self._client = MimiClient(self._host, self._port)

        map_name = self._maps[self._map_index % len(self._maps)]
        self._map_index += 1

        self._client.send_command(f"RESET;{map_name}")

        state = self._client.receive_state()
        if state is None:
            raise RuntimeError("Failed to receive initial state from Mindustry server")
        self._prev_state = state
        self._step_count = 0

        obs = parse_observation(state)
        return obs, {}
```

**Step 5: Run tests to confirm they pass**

```bash
python -m pytest rl/tests/test_env.py -v
```
Expected: all env tests pass (6 old + 3 new = 9 total)

**Step 6: Run full suite**

```bash
python -m pytest rl/tests/ -v
```
Expected: all pass

**Step 7: Commit**

```bash
git add rl/env/mindustry_env.py rl/tests/test_env.py
git commit -m "feat: MindustryEnv sends RESET;[mapname] and cycles through map list"
```

---

## Task 4: Update `train.py` to spawn and manage the server

**Files:**
- Modify: `rl/train.py`

No new unit tests — `MindustryServer` is already tested. Integration tested manually.

**Step 1: Update `parse_args`**

Add to the argument parser:
```python
    p.add_argument(
        "--server-jar",
        default="server-release.jar",
        dest="server_jar",
        help="Path to server-release.jar (default: server-release.jar in cwd)",
    )
    p.add_argument(
        "--maps",
        default=None,
        help="Comma-separated list of map names to cycle (default: built-in list)",
    )
    p.add_argument(
        "--no-server",
        action="store_true",
        dest="no_server",
        help="Skip spawning server (connect to already-running server)",
    )
    p.add_argument(
        "--server-data-dir",
        default="rl/server_data",
        dest="server_data_dir",
        help="Directory for Mindustry server data (saves, config)",
    )
```

**Step 2: Update `main`**

```python
def main() -> None:
    args = parse_args()

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    maps = [m.strip() for m in args.maps.split(",")] if args.maps else None

    server: Optional["MindustryServer"] = None
    if not args.no_server:
        from rl.server.manager import MindustryServer
        server = MindustryServer(jar_path=args.server_jar, data_dir=args.server_data_dir)
        print(f"Starting Mindustry server ({args.server_jar})...")
        server.start()
        print("Server ready. Connecting agent...")
        # Brief pause to let the mod initialize its TCP server
        import time
        time.sleep(3)

    try:
        env = Monitor(
            MindustryEnv(host=args.host, port=args.port, max_steps=args.max_steps, maps=maps),
            filename=f"{args.logs_dir}/monitor",
        )

        model = A2C(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=args.logs_dir,
        )

        callbacks = make_callbacks(save_path=args.models_dir)

        print(f"Starting A2C training for {args.timesteps:,} timesteps...")
        model.learn(total_timesteps=args.timesteps, callback=callbacks)

        final_path = f"{args.models_dir}/final_model"
        model.save(final_path)
        print(f"Training complete. Model saved to {final_path}.zip")

    finally:
        if server is not None:
            print("Stopping server...")
            server.stop()
```

Also add `Optional` import at the top:
```python
from typing import Optional
```

**Step 3: Run full test suite to confirm no regressions**

```bash
python -m pytest rl/tests/ -v
```
Expected: all pass

**Step 4: Commit**

```bash
git add rl/train.py
git commit -m "feat: train.py spawns server-release.jar and passes maps list to env"
```

---

## Task 5: Verify `MimiClient.send_command` is public

**Files:**
- Read + possibly modify: `rl/env/mimi_client.py`

**Step 1: Read the file**

```bash
cat rl/env/mimi_client.py
```

**Step 2: Check if `send_command` is already public**

If the method is named `send_command`, you're done.
If it's named `_send_command` (private), add a public alias:
```python
def send_command(self, cmd: str) -> None:
    self._send_command(cmd)
```

**Step 3: If changed, run tests**

```bash
python -m pytest rl/tests/ -v
```
Expected: all pass

**Step 4: Commit if changed**

```bash
git add rl/env/mimi_client.py
git commit -m "fix: expose send_command as public method on MimiClient"
```

---

## Task 6: Repackage mod and final verification

**Files:**
- Update: `mimi-gateway-v1.0.4.zip`

**Step 1: Run full test suite**

```bash
python -m pytest rl/tests/ -v
```
Expected: all pass (36+ tests)

**Step 2: Repackage mod**

```bash
python3 -c "
import zipfile, os
with zipfile.ZipFile('mimi-gateway-v1.0.4.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('mod.hjson')
    for root, dirs, files in os.walk('scripts'):
        for f in files:
            z.write(os.path.join(root, f))
print('OK')
"
```

**Step 3: Commit**

```bash
git add mimi-gateway-v1.0.4.zip
git commit -m "release: v1.0.4 — server mode support with RESET command"
```

**Step 4: Smoke test (requires live server)**

```bash
source rl/venv/bin/activate.fish
python -m rl.train --server-jar server-release.jar --timesteps 100
```

Expected: server spawns, "Server ready", agent connects, first episode begins.

---

## Summary of changes

| File | Change |
|---|---|
| `rl/server/__init__.py` | New (empty) |
| `rl/server/manager.py` | New — MindustryServer class |
| `rl/tests/test_server_manager.py` | New — 5 tests |
| `scripts/main.js` | Add RESET command handler |
| `rl/env/mindustry_env.py` | Add maps cycling, send RESET on reset() |
| `rl/tests/test_env.py` | Add 3 new tests |
| `rl/train.py` | Add --server-jar, --maps, --no-server args |
| `mimi-gateway-v1.0.4.zip` | Updated package |
