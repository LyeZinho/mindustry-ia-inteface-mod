# RL System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete RL training pipeline in `/rl/` that trains an A2C agent to play Mindustry via the Mimi Gateway TCP mod.

**Architecture:** Gymnasium environment wraps a TCP client (`mimi_client.py`) that connects to Mindustry on `localhost:9000`. Observations are a Dict with a `(4, 31, 31)` grid tensor (CNN) + a `(43,)` features vector (MLP). Actions are a Dict with `action_type` (Discrete 8) + integer `x`, `y` coords. Stable Baselines3 A2C with `MultiInputPolicy` trains end-to-end. Checkpoints and TensorBoard logs saved in `rl/models/` and `rl/logs/`.

**Tech Stack:** Python 3.10+, Gymnasium ≥ 0.29, Stable Baselines3 ≥ 2.0, PyTorch ≥ 2.0, TensorBoard ≥ 2.13, pytest

---

## Reference Files

| File | Why it matters |
|---|---|
| `docs/plans/2026-04-02-rl-system-design.md` | Primary spec — spaces, rewards, hyperparams |
| `test_mimi_client.py` | TCP protocol — recv/send patterns, JSON structure |
| `scripts/main.js` | Authoritative state JSON format (grid, resources, enemies…) |

---

## Task 1: Project scaffold + requirements

**Files:**
- Create: `rl/requirements.txt`
- Create: `rl/__init__.py` (empty)
- Create: `rl/env/__init__.py` (empty)
- Create: `rl/rewards/__init__.py` (empty)
- Create: `rl/callbacks/__init__.py` (empty)
- Create: `rl/models/.gitkeep`
- Create: `rl/logs/.gitkeep`

**Step 1: Create directory structure**

```bash
mkdir -p rl/env rl/rewards rl/callbacks rl/models rl/logs
touch rl/__init__.py rl/env/__init__.py rl/rewards/__init__.py rl/callbacks/__init__.py
touch rl/models/.gitkeep rl/logs/.gitkeep
```

**Step 2: Write `rl/requirements.txt`**

```
stable-baselines3[extra]>=2.0.0
gymnasium>=0.29.0
torch>=2.0.0
numpy>=1.24.0
tensorboard>=2.13.0
pytest>=7.0.0
```

**Step 3: Install dependencies**

```bash
pip install -r rl/requirements.txt
```

Expected: all packages install without error.

**Step 4: Commit**

```bash
git add rl/
git commit -m "feat(rl): scaffold /rl directory structure and requirements"
```

---

## Task 2: `mimi_client.py` — TCP client

**Files:**
- Create: `rl/env/mimi_client.py`
- Create: `rl/tests/__init__.py` (empty)
- Create: `rl/tests/test_mimi_client.py`

**Goal:** A `MimiClient` that connects to `localhost:9000`, reads newline-delimited JSON state frames, and sends `\n`-terminated command strings. Must be mockable for unit tests (dependency-injected socket).

**Step 1: Write failing tests**

Create `rl/tests/__init__.py` (empty), then create `rl/tests/test_mimi_client.py`:

```python
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
```

**Step 2: Run tests — verify they FAIL**

```bash
cd /home/pedro/repo/mindustry-ia-inteface-mod
python -m pytest rl/tests/test_mimi_client.py -v
```

Expected: `ModuleNotFoundError: No module named 'rl.env.mimi_client'`

**Step 3: Implement `rl/env/mimi_client.py`**

```python
"""
MimiClient — TCP client for the Mimi Gateway Mindustry mod.

Connects to localhost:9000 by default. Reads newline-delimited JSON state
frames from the server and sends newline-terminated command strings.

The constructor accepts an optional pre-built socket for unit testing.
"""
from __future__ import annotations

import json
import socket
from typing import Any, Dict, Optional


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

    # ------------------------------------------------------------------ #
    # Receiving state                                                       #
    # ------------------------------------------------------------------ #

    def receive_state(self) -> Optional[Dict[str, Any]]:
        """Block until next state frame arrives. Returns None on disconnect."""
        line = self._file.readline()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------ #
    # Sending commands                                                      #
    # ------------------------------------------------------------------ #

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
```

**Step 4: Run tests — verify they PASS**

```bash
python -m pytest rl/tests/test_mimi_client.py -v
```

Expected: all 8 tests PASS.

**Step 5: Commit**

```bash
git add rl/env/mimi_client.py rl/tests/
git commit -m "feat(rl): add MimiClient TCP wrapper with unit tests"
```

---

## Task 3: `spaces.py` — observation & action space definitions

**Files:**
- Create: `rl/env/spaces.py`
- Create: `rl/tests/test_spaces.py`

**Goal:** Two pure functions: `make_obs_space()` → `spaces.Dict` and `make_action_space()` → `spaces.Dict`. Also a `parse_observation(state)` function that converts a raw Mimi Gateway state dict into the numpy arrays expected by the obs space.

**Step 1: Write failing tests**

```python
"""Tests for observation/action space definitions."""
import numpy as np
import pytest
from gymnasium import spaces
from rl.env.spaces import make_obs_space, make_action_space, parse_observation


MINIMAL_STATE = {
    "tick": 1000,
    "time": 500,
    "wave": 3,
    "waveTime": 300,
    "resources": {"copper": 450, "lead": 120, "graphite": 75, "titanium": 50, "thorium": 0},
    "power": {"produced": 120.5, "consumed": 80.2, "stored": 500, "capacity": 1000},
    "core": {"hp": 0.95, "x": 15, "y": 15, "size": 3},
    "player": {"x": 15, "y": 15},
    "enemies": [],
    "friendlyUnits": [],
    "buildings": [],
    "grid": [{"x": i % 31, "y": i // 31, "block": "air", "floor": "stone",
               "team": "neutral", "hp": 0.0, "rotation": 0} for i in range(961)],
}


def test_obs_space_shape():
    obs = make_obs_space()
    assert isinstance(obs, spaces.Dict)
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (43,)


def test_action_space_structure():
    act = make_action_space()
    assert isinstance(act, spaces.Dict)
    assert isinstance(act["action_type"], spaces.Discrete)
    assert act["action_type"].n == 8
    assert act["x"].shape == (1,)
    assert act["y"].shape == (1,)


def test_parse_observation_returns_correct_shapes():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (43,)


def test_parse_observation_grid_dtype():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["grid"].dtype == np.float32


def test_parse_observation_features_dtype():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["features"].dtype == np.float32


def test_parse_observation_grid_within_range():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["grid"].min() >= 0.0
    assert obs["grid"].max() <= 1.0


def test_parse_observation_zero_pads_missing_enemies():
    """With no enemies, enemy slots in features should be zero."""
    obs = parse_observation(MINIMAL_STATE)
    # enemies occupy features[11:31] (5 enemies × 4 features each)
    enemy_block = obs["features"][11:31]
    assert np.all(enemy_block == 0.0)


def test_obs_space_contains_parsed_obs():
    obs_space = make_obs_space()
    obs = parse_observation(MINIMAL_STATE)
    # gymnasium contains() checks shape + dtype + bounds
    assert obs_space["grid"].contains(obs["grid"])
    # features has unbounded range, just check shape/dtype
    assert obs["features"].shape == obs_space["features"].shape
```

**Step 2: Run tests — verify they FAIL**

```bash
python -m pytest rl/tests/test_spaces.py -v
```

Expected: `ModuleNotFoundError: No module named 'rl.env.spaces'`

**Step 3: Implement `rl/env/spaces.py`**

The features vector layout (43 total):

```
[0]       core_hp                          (1)
[1–5]     resources: copper, lead, graphite, titanium, thorium  /1000  (5)
[6–9]     power: produced, consumed, stored, capacity  /1000   (4)
[10]      wave number / 100                (1)
[11–30]   top-5 enemies: hp, x/30, y/30, type_enc per enemy    (20)  ← zeros if absent
[31–39]   top-3 friendly units: hp, x/30, y/30                  (9)  ← zeros if absent
[40]      waveTime / 3600                  (1)
[41]      core_x / 30                      (1)
[42]      core_y / 30                      (1)
```

Grid channel mapping from grid tile:

```
ch0 = block_type / 50.0   (int-encoded block name hash, clamped 0–50)
ch1 = tile hp
ch2 = team_enc  (neutral=0.0, sharded/ally=0.5, crux/enemy=1.0)
ch3 = rotation / 3.0
```

Block name → int encoding: use `hash(block_name) % 50` normalized to [0,1].

```python
"""
Observation and action space definitions for the Mindustry RL environment.

Observation: Dict with two tensors:
  "grid":     float32 (4, 31, 31)  — CNN input
  "features": float32 (43,)        — MLP input

Action: Dict with:
  "action_type": Discrete(8)
  "x":           Box(0, 30, (1,), int32)
  "y":           Box(0, 30, (1,), int32)
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces

# ------------------------------------------------------------------ #
# Constants                                                             #
# ------------------------------------------------------------------ #

GRID_SIZE = 31
OBS_FEATURES_DIM = 43
NUM_ACTIONS = 8
MAX_ENEMIES = 5
MAX_FRIENDLY = 3
ENEMY_FEATURES = 4   # hp, x, y, type_enc
FRIENDLY_FEATURES = 3  # hp, x, y

ALLY_TEAMS = {"sharded", "player"}
ENEMY_TEAMS = {"crux"}


# ------------------------------------------------------------------ #
# Space constructors                                                   #
# ------------------------------------------------------------------ #

def make_obs_space() -> spaces.Dict:
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(4, GRID_SIZE, GRID_SIZE), dtype=np.float32),
        "features": spaces.Box(-np.inf, np.inf, shape=(OBS_FEATURES_DIM,), dtype=np.float32),
    })


def make_action_space() -> spaces.Dict:
    return spaces.Dict({
        "action_type": spaces.Discrete(NUM_ACTIONS),
        "x": spaces.Box(0, GRID_SIZE - 1, shape=(1,), dtype=np.int32),
        "y": spaces.Box(0, GRID_SIZE - 1, shape=(1,), dtype=np.int32),
    })


# ------------------------------------------------------------------ #
# Observation parser                                                   #
# ------------------------------------------------------------------ #

def _encode_team(team: str) -> float:
    if team in ALLY_TEAMS:
        return 0.5
    if team in ENEMY_TEAMS:
        return 1.0
    return 0.0


def _encode_block(block: str) -> float:
    return (abs(hash(block)) % 50) / 50.0


def parse_observation(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert raw Mimi Gateway state dict into gym-compatible observation."""
    grid_arr = _parse_grid(state.get("grid", []))
    features = _parse_features(state)
    return {
        "grid": grid_arr.astype(np.float32),
        "features": features.astype(np.float32),
    }


def _parse_grid(grid: List[Dict[str, Any]]) -> np.ndarray:
    arr = np.zeros((4, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for tile in grid:
        x = int(tile.get("x", 0))
        y = int(tile.get("y", 0))
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            continue
        arr[0, y, x] = _encode_block(tile.get("block", "air"))
        arr[1, y, x] = float(tile.get("hp", 0.0))
        arr[2, y, x] = _encode_team(tile.get("team", "neutral"))
        arr[3, y, x] = float(tile.get("rotation", 0)) / 3.0
    return arr


def _parse_features(state: Dict[str, Any]) -> np.ndarray:
    feat = np.zeros(OBS_FEATURES_DIM, dtype=np.float32)

    core = state.get("core", {})
    feat[0] = float(core.get("hp", 0.0))

    res = state.get("resources", {})
    feat[1] = res.get("copper", 0.0) / 1000.0
    feat[2] = res.get("lead", 0.0) / 1000.0
    feat[3] = res.get("graphite", 0.0) / 1000.0
    feat[4] = res.get("titanium", 0.0) / 1000.0
    feat[5] = res.get("thorium", 0.0) / 1000.0

    power = state.get("power", {})
    max_power = max(float(power.get("capacity", 1.0)), 1.0)
    feat[6] = float(power.get("produced", 0.0)) / max_power
    feat[7] = float(power.get("consumed", 0.0)) / max_power
    feat[8] = float(power.get("stored", 0.0)) / max_power
    feat[9] = 1.0  # capacity normalized = 1

    feat[10] = float(state.get("wave", 0)) / 100.0

    # Enemies (top MAX_ENEMIES, zero-padded)
    enemies = state.get("enemies", [])[:MAX_ENEMIES]
    base = 11
    for i, e in enumerate(enemies):
        offset = base + i * ENEMY_FEATURES
        feat[offset] = float(e.get("hp", 0.0))
        feat[offset + 1] = float(e.get("x", 0)) / (GRID_SIZE - 1)
        feat[offset + 2] = float(e.get("y", 0)) / (GRID_SIZE - 1)
        feat[offset + 3] = (abs(hash(e.get("type", ""))) % 20) / 20.0

    # Friendly units (top MAX_FRIENDLY, zero-padded)
    friendly = state.get("friendlyUnits", [])[:MAX_FRIENDLY]
    base = 31
    for i, u in enumerate(friendly):
        offset = base + i * FRIENDLY_FEATURES
        feat[offset] = float(u.get("hp", 0.0))
        feat[offset + 1] = float(u.get("x", 0)) / (GRID_SIZE - 1)
        feat[offset + 2] = float(u.get("y", 0)) / (GRID_SIZE - 1)

    feat[40] = float(state.get("waveTime", 0)) / 3600.0
    feat[41] = float(core.get("x", 0)) / (GRID_SIZE - 1)
    feat[42] = float(core.get("y", 0)) / (GRID_SIZE - 1)

    return feat
```

**Step 4: Run tests — verify they PASS**

```bash
python -m pytest rl/tests/test_spaces.py -v
```

Expected: all 8 tests PASS.

**Step 5: Commit**

```bash
git add rl/env/spaces.py rl/tests/test_spaces.py
git commit -m "feat(rl): add observation/action spaces and parse_observation"
```

---

## Task 4: `multi_objective.py` — reward function

**Files:**
- Create: `rl/rewards/multi_objective.py`
- Create: `rl/tests/test_reward.py`

**Goal:** A pure `compute_reward(prev_state, curr_state, done)` function — no side effects, fully unit-testable.

**Step 1: Write failing tests**

```python
"""Tests for multi-objective reward function."""
import pytest
from rl.rewards.multi_objective import compute_reward

BASE = {
    "core": {"hp": 1.0},
    "wave": 1,
    "resources": {"copper": 0, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0},
    "friendlyUnits": [],
    "enemies": [],
}


def make_state(**overrides):
    import copy
    s = copy.deepcopy(BASE)
    for k, v in overrides.items():
        if isinstance(v, dict) and k in s:
            s[k].update(v)
        else:
            s[k] = v
    return s


def test_reward_core_hp_loss():
    """Losing core HP is penalized."""
    prev = make_state(core={"hp": 1.0})
    curr = make_state(core={"hp": 0.8})
    r = compute_reward(prev, curr, done=False)
    assert r < 0


def test_reward_core_hp_gain():
    """Maintaining full HP gives a small positive contribution."""
    prev = make_state(core={"hp": 0.9})
    curr = make_state(core={"hp": 0.9})
    r = compute_reward(prev, curr, done=False)
    # time penalty applies, so result is slightly negative
    assert r == pytest.approx(-0.001, abs=1e-4)


def test_reward_wave_survived():
    """Completing a wave gives +0.20 bonus."""
    prev = make_state(wave=1)
    curr = make_state(wave=2)
    r = compute_reward(prev, curr, done=False)
    assert r > 0


def test_reward_resource_accumulation():
    """Accumulating resources gives a small positive reward."""
    prev = make_state(resources={"copper": 0, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0})
    curr = make_state(resources={"copper": 500, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0})
    r = compute_reward(prev, curr, done=False)
    assert r > 0


def test_reward_terminal_penalty():
    """Core destroyed applies -1.0 terminal penalty."""
    prev = make_state(core={"hp": 0.1})
    curr = make_state(core={"hp": 0.0})
    r = compute_reward(prev, curr, done=True)
    # -1.0 terminal + core_hp_delta penalty
    assert r <= -1.0


def test_reward_friendly_units_ratio():
    """More friendly units alive → higher reward."""
    prev_state = make_state()
    curr_no_units = make_state(friendlyUnits=[], enemies=[{"hp": 1.0}])
    curr_with_units = make_state(friendlyUnits=[{"hp": 1.0}, {"hp": 1.0}], enemies=[{"hp": 1.0}])
    r_no = compute_reward(prev_state, curr_no_units, done=False)
    r_with = compute_reward(prev_state, curr_with_units, done=False)
    assert r_with > r_no
```

**Step 2: Run tests — verify they FAIL**

```bash
python -m pytest rl/tests/test_reward.py -v
```

Expected: `ModuleNotFoundError: No module named 'rl.rewards.multi_objective'`

**Step 3: Implement `rl/rewards/multi_objective.py`**

```python
"""
Multi-objective reward function for the Mindustry RL agent.

reward = 0.50 * core_hp_delta
       + 0.20 * wave_survived_bonus
       + 0.15 * resources_delta / 500
       + 0.15 * friendly_units_ratio
       - 0.001  (time penalty)

Terminal: -1.0 if core_destroyed.
"""
from __future__ import annotations

from typing import Any, Dict


def compute_reward(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    done: bool,
) -> float:
    # --- Core HP delta ---
    prev_hp = float(prev_state.get("core", {}).get("hp", 0.0))
    curr_hp = float(curr_state.get("core", {}).get("hp", 0.0))
    core_hp_delta = curr_hp - prev_hp

    # --- Wave survived bonus ---
    prev_wave = int(prev_state.get("wave", 0))
    curr_wave = int(curr_state.get("wave", 0))
    wave_survived_bonus = 1.0 if curr_wave > prev_wave else 0.0

    # --- Resources delta (sum all, normalized) ---
    def _total_resources(state: Dict[str, Any]) -> float:
        res = state.get("resources", {})
        return sum(float(v) for v in res.values())

    resources_delta = _total_resources(curr_state) - _total_resources(prev_state)

    # --- Friendly units ratio ---
    friendly = curr_state.get("friendlyUnits", [])
    enemies = curr_state.get("enemies", [])
    total_units = len(friendly) + len(enemies)
    friendly_ratio = len(friendly) / total_units if total_units > 0 else 0.0

    # --- Compose reward ---
    reward = (
        0.50 * core_hp_delta
        + 0.20 * wave_survived_bonus
        + 0.15 * (resources_delta / 500.0)
        + 0.15 * friendly_ratio
        - 0.001
    )

    # --- Terminal penalty ---
    if done and curr_hp <= 0.0:
        reward -= 1.0

    return float(reward)
```

**Step 4: Run tests — verify they PASS**

```bash
python -m pytest rl/tests/test_reward.py -v
```

Expected: all 6 tests PASS.

**Step 5: Commit**

```bash
git add rl/rewards/multi_objective.py rl/tests/test_reward.py
git commit -m "feat(rl): add multi-objective reward function with unit tests"
```

---

## Task 5: `mindustry_env.py` — Gymnasium environment

**Files:**
- Create: `rl/env/mindustry_env.py`
- Create: `rl/tests/test_env.py`

**Goal:** A `MindustryEnv(gym.Env)` that wraps `MimiClient` and `compute_reward`. The client must be injectable (pass `client=` to `__init__`) for tests. `reset()` sends `MSG;RESET` and returns first parsed observation. `step(action)` executes the action command, receives next state, computes reward.

**Action type → command mapping:**

| action_type | command sent |
|---|---|
| 0 (WAIT) | `MSG;WAIT` (no-op signal) |
| 1 (BUILD_TURRET) | `BUILD;duo;{x};{y};0` |
| 2 (BUILD_WALL) | `BUILD;wall;{x};{y};0` |
| 3 (BUILD_SOLAR) | `BUILD;solar-panel;{x};{y};0` |
| 4 (REPAIR) | `REPAIR;{x};{y}` |
| 5 (MOVE_UNIT) | `UNIT_MOVE;{first_friendly_id};{x};{y}` |
| 6 (ATTACK) | `ATTACK;{first_friendly_id};{x};{y}` |
| 7 (SPAWN_UNIT) | `FACTORY;{x};{y};poly` |

**Step 1: Write failing tests**

```python
"""Tests for MindustryEnv — uses a mock MimiClient (no live game)."""
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from rl.env.mindustry_env import MindustryEnv

MOCK_STATE = {
    "tick": 1000, "time": 500, "wave": 1, "waveTime": 300,
    "resources": {"copper": 100, "lead": 50, "graphite": 0, "titanium": 0, "thorium": 0},
    "power": {"produced": 10.0, "consumed": 5.0, "stored": 100, "capacity": 1000},
    "core": {"hp": 1.0, "x": 15, "y": 15, "size": 3},
    "player": {"x": 15, "y": 15},
    "enemies": [],
    "friendlyUnits": [],
    "buildings": [],
    "grid": [{"x": i % 31, "y": i // 31, "block": "air", "floor": "stone",
               "team": "neutral", "hp": 0.0, "rotation": 0} for i in range(961)],
}


def make_mock_client(states=None):
    """Mock MimiClient returning predefined states."""
    client = MagicMock()
    if states is None:
        states = [MOCK_STATE, MOCK_STATE]
    client.receive_state.side_effect = states
    return client


def test_reset_returns_valid_obs():
    """reset() returns obs dict with correct shapes."""
    env = MindustryEnv(client=make_mock_client())
    obs, info = env.reset()
    assert "grid" in obs and "features" in obs
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (43,)
    assert isinstance(info, dict)


def test_step_returns_five_tuple():
    """step() returns (obs, reward, terminated, truncated, info)."""
    env = MindustryEnv(client=make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE]))
    env.reset()
    action = {"action_type": 0, "x": np.array([15], dtype=np.int32), "y": np.array([15], dtype=np.int32)}
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs["grid"].shape == (4, 31, 31)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_step_build_turret_sends_build_command():
    """action_type=1 sends BUILD;duo;x;y;0."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = {"action_type": 1, "x": np.array([10], dtype=np.int32), "y": np.array([12], dtype=np.int32)}
    env.step(action)
    client.build.assert_called_with("duo", 10, 12, rotation=0)


def test_step_wait_sends_msg():
    """action_type=0 (WAIT) sends message command."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = {"action_type": 0, "x": np.array([0], dtype=np.int32), "y": np.array([0], dtype=np.int32)}
    env.step(action)
    client.message.assert_called_once()


def test_episode_terminates_on_core_destroyed():
    """terminated=True when core hp <= 0."""
    dead_state = {**MOCK_STATE, "core": {"hp": 0.0, "x": 15, "y": 15, "size": 3}}
    client = make_mock_client(states=[MOCK_STATE, dead_state])
    env = MindustryEnv(client=client)
    env.reset()
    action = {"action_type": 0, "x": np.array([0], dtype=np.int32), "y": np.array([0], dtype=np.int32)}
    _, _, terminated, _, _ = env.step(action)
    assert terminated is True


def test_episode_truncates_on_max_steps():
    """truncated=True when step count >= max_steps."""
    from itertools import repeat
    states = list(repeat(MOCK_STATE, 12))
    client = MagicMock()
    client.receive_state.side_effect = states
    env = MindustryEnv(client=client, max_steps=5)
    env.reset()
    action = {"action_type": 0, "x": np.array([0], dtype=np.int32), "y": np.array([0], dtype=np.int32)}
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(action)
    assert not truncated
    _, _, terminated, truncated, _ = env.step(action)
    assert truncated is True
```

**Step 2: Run tests — verify they FAIL**

```bash
python -m pytest rl/tests/test_env.py -v
```

Expected: `ModuleNotFoundError: No module named 'rl.env.mindustry_env'`

**Step 3: Implement `rl/env/mindustry_env.py`**

```python
"""
MindustryEnv — Gymnasium environment wrapping the Mimi Gateway TCP mod.

Observation: Dict{"grid": (4,31,31), "features": (43,)}
Action:      Dict{"action_type": Discrete(8), "x": Box(1,), "y": Box(1,)}
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from rl.env.mimi_client import MimiClient
from rl.env.spaces import make_obs_space, make_action_space, parse_observation
from rl.rewards.multi_objective import compute_reward


class MindustryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        max_steps: int = 5000,
        client: Optional[MimiClient] = None,
    ) -> None:
        super().__init__()
        self.observation_space = make_obs_space()
        self.action_space = make_action_space()
        self.max_steps = max_steps

        self._host = host
        self._port = port
        self._client: Optional[MimiClient] = client  # injectable for tests

        self._prev_state: Optional[Dict[str, Any]] = None
        self._step_count: int = 0

    # ------------------------------------------------------------------ #
    # Gymnasium API                                                        #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        if self._client is None:
            self._client = MimiClient(self._host, self._port)

        self._client.message("RESET")
        state = self._client.receive_state()
        self._prev_state = state
        self._step_count = 0

        obs = parse_observation(state)
        return obs, {}

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action_type = int(action["action_type"])
        x = int(action["x"][0])
        y = int(action["y"][0])

        self._execute_action(action_type, x, y)

        state = self._client.receive_state()
        self._step_count += 1

        obs = parse_observation(state)
        core_hp = float(state.get("core", {}).get("hp", 0.0))
        terminated = core_hp <= 0.0
        truncated = self._step_count >= self.max_steps

        reward = compute_reward(self._prev_state, state, done=terminated)
        self._prev_state = state

        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------ #
    # Action dispatch                                                      #
    # ------------------------------------------------------------------ #

    def _execute_action(self, action_type: int, x: int, y: int) -> None:
        if action_type == 0:  # WAIT
            self._client.message("WAIT")
        elif action_type == 1:  # BUILD_TURRET
            self._client.build("duo", x, y, rotation=0)
        elif action_type == 2:  # BUILD_WALL
            self._client.build("wall", x, y, rotation=0)
        elif action_type == 3:  # BUILD_SOLAR
            self._client.build("solar-panel", x, y, rotation=0)
        elif action_type == 4:  # REPAIR
            self._client.repair(x, y)
        elif action_type == 5:  # MOVE_UNIT
            unit_id = self._get_first_friendly_id()
            if unit_id is not None:
                self._client.move_unit(unit_id, x, y)
        elif action_type == 6:  # ATTACK
            unit_id = self._get_first_friendly_id()
            if unit_id is not None:
                self._client.attack(unit_id, x, y)
        elif action_type == 7:  # SPAWN_UNIT
            self._client.spawn_unit(x, y, unit_type="poly")

    def _get_first_friendly_id(self) -> Optional[int]:
        if self._prev_state is None:
            return None
        units = self._prev_state.get("friendlyUnits", [])
        if units:
            return int(units[0].get("id", 0))
        return None
```

**Step 4: Run tests — verify they PASS**

```bash
python -m pytest rl/tests/test_env.py -v
```

Expected: all 6 tests PASS.

**Step 5: Commit**

```bash
git add rl/env/mindustry_env.py rl/tests/test_env.py
git commit -m "feat(rl): add MindustryEnv Gymnasium environment with mock tests"
```

---

## Task 6: `training_callbacks.py` — SB3 callbacks

**Files:**
- Create: `rl/callbacks/training_callbacks.py`

**Goal:** Bundle `CheckpointCallback` and a custom `RewardLoggerCallback` that logs mean reward and wave number to TensorBoard on each rollout end.

No isolated unit tests for this file (depends on SB3 internals) — it will be verified during training smoke test.

**Step 1: Implement `rl/callbacks/training_callbacks.py`**

```python
"""
Training callbacks for Stable Baselines3.

Usage in train.py:
    from rl.callbacks.training_callbacks import make_callbacks
    callbacks = make_callbacks(save_path="rl/models", log_path="rl/logs")
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)


class RewardLoggerCallback(BaseCallback):
    """Logs mean rollout reward and current wave to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self) -> None:
        if self._episode_rewards:
            mean_r = sum(self._episode_rewards) / len(self._episode_rewards)
            self.logger.record("rollout/mean_episode_reward", mean_r)
            self._episode_rewards.clear()


def make_callbacks(
    save_path: str = "rl/models",
    save_freq: int = 10_000,
    verbose: int = 1,
) -> CallbackList:
    Path(save_path).mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix="mindustry_a2c",
        verbose=verbose,
    )
    reward_logger = RewardLoggerCallback(verbose=verbose)
    return CallbackList([checkpoint_cb, reward_logger])
```

**Step 2: Commit**

```bash
git add rl/callbacks/training_callbacks.py
git commit -m "feat(rl): add SB3 training callbacks (checkpoint + reward logger)"
```

---

## Task 7: `train.py` — training entry point

**Files:**
- Create: `rl/train.py`

**Goal:** CLI script that creates `MindustryEnv`, wraps it with SB3 `A2C("MultiInputPolicy")`, attaches callbacks, and calls `model.learn(total_timesteps)`. Saves final model to `rl/models/final_model`.

**Step 1: Implement `rl/train.py`**

```python
"""
Training entry point for the Mindustry A2C agent.

Usage:
    python -m rl.train
    python -m rl.train --timesteps 500000 --host localhost --port 9000

Requires:
    - Mindustry running with Mimi Gateway mod loaded
    - pip install -r rl/requirements.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from rl.env.mindustry_env import MindustryEnv
from rl.callbacks.training_callbacks import make_callbacks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Mindustry A2C agent")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--max-steps", type=int, default=5000, dest="max_steps")
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--n-steps", type=int, default=128, dest="n_steps")
    p.add_argument("--models-dir", default="rl/models")
    p.add_argument("--logs-dir", default="rl/logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    env = Monitor(
        MindustryEnv(host=args.host, port=args.port, max_steps=args.max_steps),
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


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add rl/train.py
git commit -m "feat(rl): add A2C training entry point (train.py)"
```

---

## Task 8: `evaluate.py` — model evaluation

**Files:**
- Create: `rl/evaluate.py`

**Goal:** CLI script to load a saved `.zip` model and run N episodes, printing mean reward and max wave reached.

**Step 1: Implement `rl/evaluate.py`**

```python
"""
Evaluate a saved Mindustry A2C model.

Usage:
    python -m rl.evaluate --model rl/models/final_model
    python -m rl.evaluate --model rl/models/final_model --episodes 5
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import A2C

from rl.env.mindustry_env import MindustryEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved Mindustry A2C model")
    p.add_argument("--model", required=True, help="Path to model .zip (without extension)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=5000, dest="max_steps")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = A2C.load(args.model)
    env = MindustryEnv(host=args.host, port=args.port, max_steps=args.max_steps)

    episode_rewards = []
    max_waves = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        max_wave = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1}: reward={total_reward:.3f}")

    print(f"\nMean reward over {args.episodes} episodes: {np.mean(episode_rewards):.3f}")
    env.close()


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add rl/evaluate.py
git commit -m "feat(rl): add model evaluation script (evaluate.py)"
```

---

## Task 9: Full test suite + smoke test

**Goal:** Run all unit tests at once to confirm everything passes. Then confirm `train.py` imports without error.

**Step 1: Run full test suite**

```bash
python -m pytest rl/tests/ -v
```

Expected: All tests pass (0 failures). If any fail, fix before proceeding.

**Step 2: Smoke test — import check (no live game needed)**

```bash
python -c "from rl.train import main; print('train.py imports OK')"
python -c "from rl.evaluate import main; print('evaluate.py imports OK')"
```

Expected: both print "OK" with no import errors.

**Step 3: Commit**

```bash
git add .
git commit -m "test(rl): verify full unit test suite passes (all green)"
```

---

## Done Criteria

- [ ] `python -m pytest rl/tests/ -v` → all green
- [ ] `python -c "from rl.train import main"` → no errors
- [ ] `python -c "from rl.evaluate import main"` → no errors
- [ ] All 9 git commits created (one per task)
- [ ] `rl/` structure matches design doc exactly

## To start training (requires live Mindustry + mod)

```bash
# 1. Start Mindustry with Mimi Gateway mod loaded
# 2. Run training:
python -m rl.train --timesteps 1000000

# 3. Monitor with TensorBoard:
tensorboard --logdir rl/logs/
```
