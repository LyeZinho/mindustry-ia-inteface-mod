# Player Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers/executing-plans to implement this plan task-by-task.

**Goal:** Replace the abstract building bot with a real-unit-as-player agent: a `poly` unit is spawned after each RESET; the agent moves it around the map and builds near it, looking like a real player to observers connecting on `localhost:6567`.

**Architecture:** The mod spawns a `poly` unit at the core after each map load and tracks its ID as `playerUnitId`. Python sends `PLAYER_MOVE` (8-directional) or `PLAYER_BUILD` (block near unit) commands. The Python env has a slim action space (`MultiDiscrete([7, 9])`) and a richer observation that includes the unit's position. Episode ends when the core is destroyed or the player unit is dead.

**Tech Stack:** Mindustry JS mod (Rhino), Python 3, Gymnasium, stable-baselines3 A2C, pytest.

---

## Context

### Key files
| File | Role |
|---|---|
| `scripts/main.js` | Mod — all game-side logic |
| `rl/env/mindustry_env.py` | Gym env — step/reset/execute |
| `rl/env/spaces.py` | Observation & action space definitions + `parse_observation` |
| `rl/rewards/multi_objective.py` | `compute_reward()` |
| `rl/tests/test_env.py` | Env unit tests |
| `rl/tests/test_spaces.py` | Spaces unit tests |
| `rl/tests/test_reward.py` | Reward unit tests |
| `mimi-gateway-v1.0.4.zip` | Mod zip (repackage after every JS change) |
| `rl/server_data/config/mods/mimi-gateway.zip` | Installed copy (keep in sync) |

### Repackage command (run after EVERY mod JS change)
```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('mimi-gateway-v1.0.4.zip', 'w', zipfile.ZIP_DEFLATED)
z.write('mod.hjson'); z.write('scripts/main.js')
z.close()
"
cp mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
```

### Run tests
```bash
source rl/venv/bin/activate.fish
python -m pytest rl/tests/ -v
```

---

## New Design Summary

### Action space: `MultiDiscrete([7, 9])`
| action[0] | Meaning | action[1] |
|---|---|---|
| 0 | WAIT | ignored (pass 0) |
| 1 | MOVE | direction 0-7 (N/NE/E/SE/S/SW/W/NW) |
| 2 | BUILD_TURRET (`duo`) | relative slot 0-8 (3×3 grid around unit) |
| 3 | BUILD_WALL (`copper-wall`) | relative slot 0-8 |
| 4 | BUILD_POWER (`solar-panel`) | relative slot 0-8 |
| 5 | BUILD_DRILL (`mechanical-drill`) | relative slot 0-8 |
| 6 | REPAIR | relative slot 0-8 |

Relative slot mapping (3×3 around unit, row-major):
```
0 1 2
3 4 5
6 7 8
```
slot → (dx, dy): slot 0 = (-1,+1), slot 1 = (0,+1), slot 2 = (+1,+1),
                 slot 3 = (-1, 0), slot 4 = (0, 0), slot 5 = (+1, 0),
                 slot 6 = (-1,-1), slot 7 = (0,-1), slot 8 = (+1,-1)

### Observation space: `Dict{"grid": (4,31,31), "features": (47,)}`
`features` grows from 43 → 47 by adding at the end:
- feat[43]: player_x relative to core, normalised ÷ 15
- feat[44]: player_y relative to core, normalised ÷ 15
- feat[45]: player alive (1.0) or dead (0.0)
- feat[46]: player hp (0.0–1.0, 0.0 if dead)

Player info comes from `state["player"]` and the first entry in `state["friendlyUnits"]` whose `type == "poly"` (used for hp).

### Reward redesign
```
reward = 0.40 * core_hp_delta
       + 0.20 * wave_survived_bonus
       + 0.10 * resources_delta / 500
       + 0.10 * power_balance_bonus        # (produced - consumed) / max(produced, 1) clamped [0,1]
       + 0.10 * build_efficiency_bonus     # +0.1 per new friendly building placed (capped at 1.0)
       + 0.10 * player_alive_bonus         # +0.1 if player unit still alive
       - 0.001                             # time penalty
```
Terminal conditions:
- `terminated = core_hp <= 0 OR player_dead`
- `done=True, core_dead` → `-1.0`
- `done=True, player_dead, core_alive` → `-0.5`

### New mod commands
```
PLAYER_MOVE;direction        direction=0..7 (N/NE/E/SE/S/SW/W/NW)
PLAYER_BUILD;block;slot      slot=0..8 relative to unit
```

The mod also:
1. Spawns a `poly` unit at the core after `Vars.logic.play()` in `handleResetCommand`
2. Stores `playerUnitId` globally
3. Exposes `state.player.hp` (0 if unit dead/not found) and `state.player.alive` (bool)
4. `PLAYER_MOVE` moves the unit `MOVE_STEP = 3` tiles in the given direction
5. `PLAYER_BUILD` builds a block at (unit_tile + slot_offset), using `tile.setNet(block, Team.sharded, 0)`

---

## Task 1 — Mod: spawn player unit on RESET

**Files:**
- Modify: `scripts/main.js`

**Step 1: Add global `playerUnitId = -1` near other globals (line ~25)**

```javascript
let playerUnitId = -1;
```

**Step 2: In `handleResetCommand`, after `Vars.logic.play();` add unit spawn**

```javascript
// Spawn player unit (poly) at core position
Core.app.post(() => {
    try {
        let coreData = Team.sharded.data();
        let core = coreData != null ? coreData.core() : null;
        if (core != null) {
            let polyType = Vars.content.units().find(u => u.name === "poly");
            if (polyType != null) {
                let spawnedUnit = Unit.create(polyType, Team.sharded);
                spawnedUnit.set(core.x, core.y);
                spawnedUnit.add();
                playerUnitId = spawnedUnit.id;
                Log.info("[Mimi Gateway] Player unit spawned id=" + playerUnitId);
            }
        }
    } catch (e) {
        Log.err("[Mimi Gateway] Erro ao spawnar player unit: " + e);
    }
});
```

Note: The outer `Core.app.post` in `handleResetCommand` already wraps map loading. Nest this second `Core.app.post` *inside* the existing one, after `Vars.logic.play();`.

**Step 3: In `captureGameState`, replace the `player` block**

Old:
```javascript
if (Vars.player != null) {
    state.player.x = Math.floor(Vars.player.x / 8);
    state.player.y = Math.floor(Vars.player.y / 8);
}
```

New:
```javascript
state.player.alive = false;
state.player.hp = 0.0;
if (playerUnitId >= 0) {
    // Find player unit by id
    let found = false;
    let allTeamData = Team.sharded.data();
    if (allTeamData != null && allTeamData.units != null) {
        allTeamData.units.forEach(u => {
            if (u != null && u.id === playerUnitId) {
                state.player.x = Math.floor(u.x / 8);
                state.player.y = Math.floor(u.y / 8);
                state.player.hp = Math.floor((u.health / u.maxHealth) * 100) / 100;
                state.player.alive = true;
                found = true;
            }
        });
    }
    if (!found) {
        // Unit is dead — episode should end
        state.player.alive = false;
    }
}
```

Also update the initial `state.player` definition (line ~146) to include `alive` and `hp`:
```javascript
player: { x: 0, y: 0, alive: false, hp: 0.0 },
```

**Step 4: Repackage zip**

```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('mimi-gateway-v1.0.4.zip', 'w', zipfile.ZIP_DEFLATED)
z.write('mod.hjson'); z.write('scripts/main.js')
z.close()
"
cp mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
```

**Step 5: Commit**
```bash
git add scripts/main.js mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
git commit -m "feat(mod): spawn player poly unit on RESET, expose player.alive and player.hp"
```

---

## Task 2 — Mod: PLAYER_MOVE and PLAYER_BUILD commands

**Files:**
- Modify: `scripts/main.js`

**Step 1: Add `PLAYER_MOVE` and `PLAYER_BUILD` to the valid commands list**

In `validateCommand`, change:
```javascript
let validCommands = ["BUILD", "UNIT_MOVE", "MSG", "ATTACK", "STOP", "FACTORY", "REPAIR", "DELETE", "UPGRADE", "RESET"];
```
to:
```javascript
let validCommands = ["BUILD", "UNIT_MOVE", "MSG", "ATTACK", "STOP", "FACTORY", "REPAIR", "DELETE", "UPGRADE", "RESET", "PLAYER_MOVE", "PLAYER_BUILD"];
```

**Step 2: Add cases to `processCommand` switch**

```javascript
case "PLAYER_MOVE":
    handlePlayerMoveCommand(parts);
    break;
case "PLAYER_BUILD":
    handlePlayerBuildCommand(parts);
    break;
```

**Step 3: Implement `handlePlayerMoveCommand`**

```javascript
const MOVE_STEP = 3; // tiles per move action
// direction encoding: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
const DIR_DX = [0, 1, 1, 1, 0, -1, -1, -1];
const DIR_DY = [1, 1, 0, -1, -1, -1, 0, 1];

function handlePlayerMoveCommand(parts) {
    if (parts.length < 2) {
        Log.info("[Mimi Gateway] PLAYER_MOVE: parâmetros insuficientes");
        return;
    }
    let dir = parseInt(parts[1]);
    if (dir < 0 || dir > 7) {
        Log.info("[Mimi Gateway] PLAYER_MOVE: direção inválida " + dir);
        return;
    }

    let unit = findPlayerUnit();
    if (unit == null) {
        Log.info("[Mimi Gateway] PLAYER_MOVE: player unit não encontrada (id=" + playerUnitId + ")");
        return;
    }

    let newX = unit.x + DIR_DX[dir] * MOVE_STEP * 8;
    let newY = unit.y + DIR_DY[dir] * MOVE_STEP * 8;
    unit.set(newX, newY);
    Log.info("[Mimi Gateway] PLAYER_MOVE dir=" + dir + " -> (" + Math.floor(newX/8) + "," + Math.floor(newY/8) + ")");
}
```

**Step 4: Implement `handlePlayerBuildCommand`**

Slot → (dx, dy) mapping (3×3 around unit):
```
slot 0=(-1,+1), 1=(0,+1), 2=(+1,+1)
slot 3=(-1, 0), 4=(0, 0), 5=(+1, 0)
slot 6=(-1,-1), 7=(0,-1), 8=(+1,-1)
```

```javascript
const SLOT_DX = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
const SLOT_DY = [1, 1, 1, 0, 0, 0, -1, -1, -1];

function handlePlayerBuildCommand(parts) {
    // PLAYER_BUILD;block_name;slot
    if (parts.length < 3) {
        Log.info("[Mimi Gateway] PLAYER_BUILD: parâmetros insuficientes");
        return;
    }
    let blockName = parts[1];
    let slot = parseInt(parts[2]);
    if (slot < 0 || slot > 8) {
        Log.info("[Mimi Gateway] PLAYER_BUILD: slot inválido " + slot);
        return;
    }

    let unit = findPlayerUnit();
    if (unit == null) {
        Log.info("[Mimi Gateway] PLAYER_BUILD: player unit não encontrada");
        return;
    }

    let unitTileX = Math.floor(unit.x / 8);
    let unitTileY = Math.floor(unit.y / 8);
    let targetTileX = unitTileX + SLOT_DX[slot];
    let targetTileY = unitTileY + SLOT_DY[slot];

    let blockType = Vars.content.blocks().find(b => b.name === blockName);
    if (blockType == null) {
        Log.info("[Mimi Gateway] PLAYER_BUILD: bloco não encontrado: " + blockName);
        return;
    }

    let tile = Vars.world.tile(targetTileX, targetTileY);
    if (tile == null) {
        Log.info("[Mimi Gateway] PLAYER_BUILD: tile inválido (" + targetTileX + "," + targetTileY + ")");
        return;
    }

    try {
        tile.setNet(blockType, Team.sharded, 0);
        Log.info("[Mimi Gateway] PLAYER_BUILD: " + blockName + " em (" + targetTileX + "," + targetTileY + ")");
    } catch (e) {
        Log.err("[Mimi Gateway] Erro ao construir: " + e);
    }
}
```

**Step 5: Implement `findPlayerUnit` helper**

```javascript
function findPlayerUnit() {
    if (playerUnitId < 0) return null;
    let result = null;
    let data = Team.sharded.data();
    if (data != null && data.units != null) {
        data.units.forEach(u => {
            if (u != null && u.id === playerUnitId) {
                result = u;
            }
        });
    }
    return result;
}
```

**Step 6: Repackage zip + commit**

```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('mimi-gateway-v1.0.4.zip', 'w', zipfile.ZIP_DEFLATED)
z.write('mod.hjson'); z.write('scripts/main.js')
z.close()
"
cp mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
git add scripts/main.js mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
git commit -m "feat(mod): add PLAYER_MOVE and PLAYER_BUILD commands"
```

---

## Task 3 — Python: new action space + spaces.py

**Files:**
- Modify: `rl/env/spaces.py`
- Modify: `rl/tests/test_spaces.py`

**Step 1: Write the failing tests first**

Replace all tests in `rl/tests/test_spaces.py`:

```python
"""Tests for observation/action space definitions."""
import numpy as np
import pytest
from gymnasium import spaces
from rl.env.spaces import (
    make_obs_space, make_action_space, parse_observation,
    BLOCK_TURRET, BLOCK_WALL, BLOCK_POWER, BLOCK_DRILL,
    NUM_ACTION_TYPES, NUM_SLOTS,
)

MINIMAL_STATE = {
    "tick": 1000, "time": 500, "wave": 3, "waveTime": 300,
    "resources": {"copper": 450, "lead": 120, "graphite": 75, "titanium": 50, "thorium": 0},
    "power": {"produced": 120.5, "consumed": 80.2, "stored": 500, "capacity": 1000},
    "core": {"hp": 0.95, "x": 15, "y": 15, "size": 3},
    "player": {"x": 16, "y": 17, "alive": True, "hp": 0.8},
    "enemies": [],
    "friendlyUnits": [],
    "buildings": [],
    "grid": [{"x": i % 31, "y": i // 31, "block": "air", "floor": "stone",
               "team": "neutral", "hp": 0.0, "rotation": 0} for i in range(961)],
}


def test_action_space_structure():
    act = make_action_space()
    assert isinstance(act, spaces.MultiDiscrete)
    assert list(act.nvec) == [NUM_ACTION_TYPES, NUM_SLOTS]


def test_obs_space_shape():
    obs = make_obs_space()
    assert isinstance(obs, spaces.Dict)
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (47,)


def test_parse_observation_returns_correct_shapes():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (47,)


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


def test_parse_observation_player_features():
    """feat[43..46] encode player position relative to core and alive/hp."""
    obs = parse_observation(MINIMAL_STATE)
    feat = obs["features"]
    # player at (16,17), core at (15,15) → dx=1, dy=2, both ÷15
    assert feat[43] == pytest.approx(1.0 / 15.0, abs=1e-5)
    assert feat[44] == pytest.approx(2.0 / 15.0, abs=1e-5)
    assert feat[45] == pytest.approx(1.0)   # alive
    assert feat[46] == pytest.approx(0.8)   # hp


def test_parse_observation_player_dead():
    """feat[45]=0 and feat[46]=0 when player not alive."""
    state = {**MINIMAL_STATE, "player": {"x": 0, "y": 0, "alive": False, "hp": 0.0}}
    obs = parse_observation(state)
    assert obs["features"][45] == pytest.approx(0.0)
    assert obs["features"][46] == pytest.approx(0.0)


def test_parse_observation_zero_pads_missing_enemies():
    obs = parse_observation(MINIMAL_STATE)
    enemy_block = obs["features"][11:31]
    assert np.all(enemy_block == 0.0)


def test_obs_space_contains_parsed_obs():
    obs_space = make_obs_space()
    obs = parse_observation(MINIMAL_STATE)
    assert obs_space["grid"].contains(obs["grid"])
    assert obs["features"].shape == obs_space["features"].shape
```

**Step 2: Run tests — verify they FAIL**

```bash
python -m pytest rl/tests/test_spaces.py -v
```
Expected: multiple failures because `NUM_ACTION_TYPES`, `NUM_SLOTS`, `BLOCK_DRILL` don't exist yet and features shape is 43 not 47.

**Step 3: Update `rl/env/spaces.py`**

Full replacement:

```python
"""
Observation and action space definitions for the Mindustry RL environment.

Observation: Dict with two tensors:
  "grid":     float32 (4, 31, 31)  — CNN input
  "features": float32 (47,)        — MLP input (was 43, +4 player fields)

Action: MultiDiscrete([7, 9])
  action[0]: action_type  ∈ {0..6}  (WAIT, MOVE, BUILD_TURRET, BUILD_WALL, BUILD_POWER, BUILD_DRILL, REPAIR)
  action[1]: arg          ∈ {0..8}  (direction 0-7 for MOVE; relative slot 0-8 for build/repair; ignored for WAIT)
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces

# ------------------------------------------------------------------ #
# Constants                                                             #
# ------------------------------------------------------------------ #

GRID_SIZE = 31
OBS_FEATURES_DIM = 47   # was 43, +4 for player (dx, dy, alive, hp)
NUM_ACTION_TYPES = 7    # WAIT, MOVE, BUILD_TURRET, BUILD_WALL, BUILD_POWER, BUILD_DRILL, REPAIR
NUM_SLOTS = 9           # 3x3 relative grid around unit (also covers 8 directions + 0 for WAIT)
MAX_ENEMIES = 5
MAX_FRIENDLY = 3
ENEMY_FEATURES = 4      # hp, x, y, type_enc
FRIENDLY_FEATURES = 3   # hp, x, y

ALLY_TEAMS = {"sharded", "player"}
ENEMY_TEAMS = {"crux"}

BLOCK_TURRET = "duo"
BLOCK_WALL = "copper-wall"
BLOCK_POWER = "solar-panel"
BLOCK_DRILL = "mechanical-drill"

# Slot (0-8) → (dx, dy) in tiles, relative to player unit
# Row-major 3×3: top-left=0, top-center=1, ... bottom-right=8
# y+ = north on Mindustry map
SLOT_DX = [-1,  0,  1, -1,  0,  1, -1,  0,  1]
SLOT_DY = [ 1,  1,  1,  0,  0,  0, -1, -1, -1]

# Direction (0-7) for MOVE: N/NE/E/SE/S/SW/W/NW
MOVE_DIR_DX = [0, 1, 1,  1,  0, -1, -1, -1]
MOVE_DIR_DY = [1, 1, 0, -1, -1, -1,  0,  1]


# ------------------------------------------------------------------ #
# Space constructors                                                   #
# ------------------------------------------------------------------ #

def make_obs_space() -> spaces.Dict:
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(4, GRID_SIZE, GRID_SIZE), dtype=np.float32),
        "features": spaces.Box(-np.inf, np.inf, shape=(OBS_FEATURES_DIM,), dtype=np.float32),
    })


def make_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([NUM_ACTION_TYPES, NUM_SLOTS])


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

    # Player unit position relative to core (feat[43..46])
    player = state.get("player", {})
    core_x = float(core.get("x", 0))
    core_y = float(core.get("y", 0))
    player_alive = bool(player.get("alive", False))
    if player_alive:
        feat[43] = (float(player.get("x", core_x)) - core_x) / 15.0
        feat[44] = (float(player.get("y", core_y)) - core_y) / 15.0
        feat[45] = 1.0
        feat[46] = float(player.get("hp", 0.0))
    # else stays 0.0

    return feat
```

**Step 4: Run tests — verify they PASS**

```bash
python -m pytest rl/tests/test_spaces.py -v
```
Expected: all pass.

**Step 5: Commit**

```bash
git add rl/env/spaces.py rl/tests/test_spaces.py
git commit -m "feat(spaces): new action space MultiDiscrete([7,9]), obs features 43→47 with player fields"
```

---

## Task 4 — Python: new reward function

**Files:**
- Modify: `rl/rewards/multi_objective.py`
- Modify: `rl/tests/test_reward.py`

**Step 1: Write the failing tests first**

Replace `rl/tests/test_reward.py`:

```python
"""Tests for multi-objective reward function."""
import pytest
from rl.rewards.multi_objective import compute_reward

BASE = {
    "core": {"hp": 1.0},
    "wave": 1,
    "resources": {"copper": 0, "lead": 0, "graphite": 0, "titanium": 0, "thorium": 0},
    "power": {"produced": 0.0, "consumed": 0.0},
    "friendlyUnits": [],
    "enemies": [],
    "player": {"alive": True, "hp": 1.0},
    "buildings": [],
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


def test_reward_wave_survived():
    """Completing a wave gives positive bonus."""
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


def test_reward_terminal_core_destroyed():
    """Core destroyed applies -1.0 terminal penalty."""
    prev = make_state(core={"hp": 0.1})
    curr = make_state(core={"hp": 0.0})
    r = compute_reward(prev, curr, done=True)
    assert r <= -1.0


def test_reward_terminal_player_dead_core_alive():
    """-0.5 terminal penalty when player dies but core is alive."""
    prev = make_state(player={"alive": True, "hp": 0.3})
    curr = make_state(player={"alive": False, "hp": 0.0})
    r = compute_reward(prev, curr, done=True)
    assert r <= -0.5
    # should NOT apply -1.0 (core still alive)
    assert r > -1.0


def test_reward_player_alive_bonus():
    """Player alive contributes positively when core HP stable."""
    prev = make_state()
    curr = make_state()
    r = compute_reward(prev, curr, done=False)
    # Should include +0.10 player alive bonus minus time penalty → positive
    assert r > 0


def test_reward_power_balance_bonus():
    """Positive power balance (produced > consumed) gives bonus."""
    prev = make_state()
    curr_balanced = make_state(power={"produced": 10.0, "consumed": 5.0})
    curr_deficit = make_state(power={"produced": 5.0, "consumed": 10.0})
    r_balanced = compute_reward(prev, curr_balanced, done=False)
    r_deficit = compute_reward(prev, curr_deficit, done=False)
    assert r_balanced > r_deficit


def test_reward_done_without_core_or_player_destroyed():
    """done=True with both core and player alive (truncation) → no extra penalty."""
    prev = make_state()
    curr = make_state()
    r = compute_reward(prev, curr, done=True)
    # time penalty + player alive bonus; no terminal penalty
    assert r > -0.5
```

**Step 2: Run tests — verify they FAIL (since reward function unchanged)**

```bash
python -m pytest rl/tests/test_reward.py -v
```
Expected: several failures (player dead test, power balance test, etc.)

**Step 3: Replace `rl/rewards/multi_objective.py`**

```python
"""
Multi-objective reward function for the Mindustry RL player agent.

reward = 0.40 * core_hp_delta
       + 0.20 * wave_survived_bonus
       + 0.10 * resources_delta / 500
       + 0.10 * power_balance_bonus
       + 0.10 * build_efficiency_bonus
       + 0.10 * player_alive_bonus
       - 0.001  (time penalty)

Terminal penalties:
  core destroyed        → -1.0
  player dead, core ok  → -0.5
"""
from __future__ import annotations

from typing import Any, Dict


def compute_reward(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    done: bool,
) -> float:
    prev_hp = float(prev_state.get("core", {}).get("hp", 0.0))
    curr_hp = float(curr_state.get("core", {}).get("hp", 0.0))
    core_hp_delta = curr_hp - prev_hp

    prev_wave = int(prev_state.get("wave", 0))
    curr_wave = int(curr_state.get("wave", 0))
    wave_survived_bonus = 1.0 if curr_wave > prev_wave else 0.0

    def _total_resources(state: Dict[str, Any]) -> float:
        return sum(float(v) for v in state.get("resources", {}).values())

    resources_delta = _total_resources(curr_state) - _total_resources(prev_state)

    # Power balance bonus: how much surplus power we produce, clamped [0, 1]
    power = curr_state.get("power", {})
    produced = float(power.get("produced", 0.0))
    consumed = float(power.get("consumed", 0.0))
    if produced > 0:
        power_balance_bonus = max(0.0, min(1.0, (produced - consumed) / produced))
    else:
        power_balance_bonus = 0.0

    # Build efficiency bonus: count friendly buildings added this step (capped at 1.0)
    prev_buildings = len(prev_state.get("buildings", []))
    curr_buildings = len(curr_state.get("buildings", []))
    new_buildings = max(0, curr_buildings - prev_buildings)
    build_efficiency_bonus = min(1.0, new_buildings * 0.1)

    # Player alive bonus
    player_alive = bool(curr_state.get("player", {}).get("alive", False))
    player_alive_bonus = 1.0 if player_alive else 0.0

    reward = (
        0.40 * core_hp_delta
        + 0.20 * wave_survived_bonus
        + 0.10 * (resources_delta / 500.0)
        + 0.10 * power_balance_bonus
        + 0.10 * build_efficiency_bonus
        + 0.10 * player_alive_bonus
        - 0.001
    )

    if done:
        if curr_hp <= 0.0:
            reward -= 1.0
        elif not player_alive:
            reward -= 0.5

    return float(reward)
```

**Step 4: Run tests — verify they PASS**

```bash
python -m pytest rl/tests/test_reward.py -v
```
Expected: all pass.

**Step 5: Commit**

```bash
git add rl/rewards/multi_objective.py rl/tests/test_reward.py
git commit -m "feat(reward): redesign with power balance, build efficiency, player alive bonus; player death = -0.5"
```

---

## Task 5 — Python: update MindustryEnv

**Files:**
- Modify: `rl/env/mindustry_env.py`
- Modify: `rl/tests/test_env.py`

**Step 1: Write the failing tests first**

Replace `rl/tests/test_env.py`:

```python
"""Tests for MindustryEnv — uses a mock MimiClient (no live game)."""
import numpy as np
import pytest
from unittest.mock import MagicMock
from rl.env.mindustry_env import MindustryEnv

MOCK_STATE = {
    "tick": 1000, "time": 500, "wave": 1, "waveTime": 300,
    "resources": {"copper": 100, "lead": 50, "graphite": 0, "titanium": 0, "thorium": 0},
    "power": {"produced": 10.0, "consumed": 5.0, "stored": 100, "capacity": 1000},
    "core": {"hp": 1.0, "x": 15, "y": 15, "size": 3},
    "player": {"x": 15, "y": 15, "alive": True, "hp": 1.0},
    "enemies": [],
    "friendlyUnits": [],
    "buildings": [],
    "grid": [{"x": i % 31, "y": i // 31, "block": "air", "floor": "stone",
               "team": "neutral", "hp": 0.0, "rotation": 0} for i in range(961)],
}


def make_mock_client(states=None):
    client = MagicMock()
    if states is None:
        states = [MOCK_STATE, MOCK_STATE]
    client.receive_state.side_effect = states
    return client


def test_reset_returns_valid_obs():
    env = MindustryEnv(client=make_mock_client())
    obs, info = env.reset()
    assert "grid" in obs and "features" in obs
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (47,)
    assert isinstance(info, dict)


def test_step_returns_five_tuple():
    env = MindustryEnv(client=make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE]))
    env.reset()
    action = np.array([0, 0], dtype=np.int64)  # WAIT, arg=0
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (47,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_step_wait_does_not_send_movement():
    """action_type=0 (WAIT) should not call send_command with PLAYER_MOVE."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([0, 0], dtype=np.int64)
    env.step(action)
    for call in client.send_command.call_args_list:
        assert not call[0][0].startswith("PLAYER_MOVE")


def test_step_move_sends_player_move_command():
    """action_type=1 sends PLAYER_MOVE;direction."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([1, 3], dtype=np.int64)  # MOVE direction=3 (SE)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_MOVE;3")


def test_step_build_turret_sends_player_build():
    """action_type=2 sends PLAYER_BUILD;duo;slot."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([2, 4], dtype=np.int64)  # BUILD_TURRET slot=4
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;duo;4")


def test_step_build_wall_sends_player_build():
    """action_type=3 sends PLAYER_BUILD;copper-wall;slot."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([3, 0], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;copper-wall;0")


def test_step_build_power_sends_player_build():
    """action_type=4 sends PLAYER_BUILD;solar-panel;slot."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([4, 1], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;solar-panel;1")


def test_step_build_drill_sends_player_build():
    """action_type=5 sends PLAYER_BUILD;mechanical-drill;slot."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([5, 2], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;mechanical-drill;2")


def test_step_repair_sends_player_build_repair():
    """action_type=6 sends PLAYER_BUILD;repair;slot (repair handled by mod)."""
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([6, 7], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("REPAIR_SLOT;7")


def test_episode_terminates_on_core_destroyed():
    dead_state = {**MOCK_STATE, "core": {"hp": 0.0, "x": 15, "y": 15, "size": 3},
                  "player": {"x": 15, "y": 15, "alive": True, "hp": 1.0}}
    client = make_mock_client(states=[MOCK_STATE, dead_state])
    env = MindustryEnv(client=client)
    env.reset()
    _, _, terminated, _, _ = env.step(np.array([0, 0], dtype=np.int64))
    assert terminated is True


def test_episode_terminates_on_player_dead():
    dead_player_state = {**MOCK_STATE, "player": {"x": 0, "y": 0, "alive": False, "hp": 0.0}}
    client = make_mock_client(states=[MOCK_STATE, dead_player_state])
    env = MindustryEnv(client=client)
    env.reset()
    _, _, terminated, _, _ = env.step(np.array([0, 0], dtype=np.int64))
    assert terminated is True


def test_episode_truncates_on_max_steps():
    from itertools import repeat
    states = list(repeat(MOCK_STATE, 12))
    client = MagicMock()
    client.receive_state.side_effect = states
    env = MindustryEnv(client=client, max_steps=5)
    env.reset()
    action = np.array([0, 0], dtype=np.int64)
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(action)
    assert not truncated
    _, _, terminated, truncated, _ = env.step(action)
    assert truncated is True


DEFAULT_MAPS = ["Ancient Caldera", "Windswept Islands"]


def test_reset_sends_reset_command_with_map():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client, maps=DEFAULT_MAPS)
    env.reset()
    client.send_command.assert_called_with("RESET;Ancient Caldera")


def test_reset_cycles_maps_on_successive_resets():
    client = MagicMock()
    client.receive_state.return_value = MOCK_STATE
    env = MindustryEnv(client=client, maps=DEFAULT_MAPS)

    env.reset()
    client.send_command.assert_called_with("RESET;Ancient Caldera")

    env.reset()
    client.send_command.assert_called_with("RESET;Windswept Islands")

    env.reset()
    client.send_command.assert_called_with("RESET;Ancient Caldera")


def test_reset_uses_default_maps_when_none_provided():
    client = make_mock_client(states=[MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    client.send_command.assert_called_once()
    call_arg = client.send_command.call_args[0][0]
    assert call_arg.startswith("RESET;")
```

**Step 2: Run tests — verify they FAIL**

```bash
python -m pytest rl/tests/test_env.py -v
```
Expected: several failures (wrong action shape, player death not terminating, etc.)

**Step 3: Update `rl/env/mindustry_env.py`**

Key changes:
- `action_space = MultiDiscrete([7, 9])` (from spaces)
- `_execute_action(action_type, arg)` replaces `(action_type, x, y)`
- `terminated = core_hp <= 0.0 OR not player_alive`
- Action dispatch:
  - 0 = WAIT (no-op)
  - 1 = PLAYER_MOVE;{arg}
  - 2 = PLAYER_BUILD;duo;{arg}
  - 3 = PLAYER_BUILD;copper-wall;{arg}
  - 4 = PLAYER_BUILD;solar-panel;{arg}
  - 5 = PLAYER_BUILD;mechanical-drill;{arg}
  - 6 = REPAIR_SLOT;{arg}  ← new command the mod needs to handle (see Task 6)

Full replacement:

```python
"""
MindustryEnv — Gymnasium environment wrapping the Mimi Gateway TCP mod.

Observation: Dict{"grid": (4,31,31), "features": (47,)}
Action:      MultiDiscrete([7, 9]) — [action_type, arg]

action_type:
  0 = WAIT
  1 = MOVE (arg = direction 0-7)
  2 = BUILD_TURRET  (arg = slot 0-8)
  3 = BUILD_WALL    (arg = slot 0-8)
  4 = BUILD_POWER   (arg = slot 0-8)
  5 = BUILD_DRILL   (arg = slot 0-8)
  6 = REPAIR        (arg = slot 0-8)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from rl.env.mimi_client import MimiClient
from rl.env.spaces import (
    make_obs_space, make_action_space, parse_observation,
    BLOCK_TURRET, BLOCK_WALL, BLOCK_POWER, BLOCK_DRILL,
)
from rl.rewards.multi_objective import compute_reward


DEFAULT_TRAINING_MAPS = [
    "Ancient Caldera", "Archipelago", "Debris Field", "Domain", "Fork", "Fortress",
    "Glacier", "Islands", "Labyrinth", "Maze", "Molten Lake", "Mud Flats",
    "Passage", "Shattered", "Tendrils", "Triad", "Veins", "Wasteland",
]


class MindustryEnv(gym.Env):
    metadata = {"render_modes": []}

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

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("Must call reset() before step()")

        action_type = int(action[0])
        arg = int(action[1])

        self._execute_action(action_type, arg)

        state = self._client.receive_state()
        if state is None:
            raise RuntimeError("Lost connection to Mindustry server during step")
        self._step_count += 1

        obs = parse_observation(state)
        core_hp = float(state.get("core", {}).get("hp", 0.0))
        player_alive = bool(state.get("player", {}).get("alive", False))
        terminated = core_hp <= 0.0 or not player_alive
        truncated = self._step_count >= self.max_steps

        reward = compute_reward(self._prev_state, state, done=terminated)
        self._prev_state = state

        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _execute_action(self, action_type: int, arg: int) -> None:
        if action_type == 0:
            pass  # WAIT — do nothing
        elif action_type == 1:
            self._client.send_command(f"PLAYER_MOVE;{arg}")
        elif action_type == 2:
            self._client.send_command(f"PLAYER_BUILD;{BLOCK_TURRET};{arg}")
        elif action_type == 3:
            self._client.send_command(f"PLAYER_BUILD;{BLOCK_WALL};{arg}")
        elif action_type == 4:
            self._client.send_command(f"PLAYER_BUILD;{BLOCK_POWER};{arg}")
        elif action_type == 5:
            self._client.send_command(f"PLAYER_BUILD;{BLOCK_DRILL};{arg}")
        elif action_type == 6:
            self._client.send_command(f"REPAIR_SLOT;{arg}")
        else:
            raise ValueError(f"Invalid action_type: {action_type}. Must be 0-6")
```

**Step 4: Run tests — verify they PASS**

```bash
python -m pytest rl/tests/test_env.py -v
```
Expected: all pass.

**Step 5: Run all tests to confirm nothing broke**

```bash
python -m pytest rl/tests/ -v
```
Expected: all pass.

**Step 6: Commit**

```bash
git add rl/env/mindustry_env.py rl/tests/test_env.py
git commit -m "feat(env): player-unit action space [7,9], terminate on player death, new _execute_action"
```

---

## Task 6 — Mod: add REPAIR_SLOT command

**Files:**
- Modify: `scripts/main.js`

**Step 1: Add `REPAIR_SLOT` to valid commands list in `validateCommand`**

```javascript
let validCommands = [..., "PLAYER_MOVE", "PLAYER_BUILD", "REPAIR_SLOT"];
```

**Step 2: Add case in `processCommand`**

```javascript
case "REPAIR_SLOT":
    handleRepairSlotCommand(parts);
    break;
```

**Step 3: Implement `handleRepairSlotCommand`**

Uses the same `SLOT_DX/SLOT_DY` constants (already defined in Task 2).

```javascript
function handleRepairSlotCommand(parts) {
    // REPAIR_SLOT;slot  — repairs building at slot relative to player unit
    if (parts.length < 2) {
        Log.info("[Mimi Gateway] REPAIR_SLOT: parâmetros insuficientes");
        return;
    }
    let slot = parseInt(parts[1]);
    if (slot < 0 || slot > 8) {
        Log.info("[Mimi Gateway] REPAIR_SLOT: slot inválido " + slot);
        return;
    }

    let unit = findPlayerUnit();
    if (unit == null) {
        Log.info("[Mimi Gateway] REPAIR_SLOT: player unit não encontrada");
        return;
    }

    let unitTileX = Math.floor(unit.x / 8);
    let unitTileY = Math.floor(unit.y / 8);
    let targetTileX = unitTileX + SLOT_DX[slot];
    let targetTileY = unitTileY + SLOT_DY[slot];

    let tile = Vars.world.tile(targetTileX, targetTileY);
    if (tile == null || tile.build == null) {
        Log.info("[Mimi Gateway] REPAIR_SLOT: bloco não encontrado em slot " + slot);
        return;
    }

    let build = tile.build;
    if (build.health < build.maxHealth) {
        build.heal(build.maxHealth * 0.5);
        Log.info("[Mimi Gateway] REPAIR_SLOT: reparado em (" + targetTileX + "," + targetTileY + ")");
    }
}
```

**Step 4: Repackage zip + commit**

```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('mimi-gateway-v1.0.4.zip', 'w', zipfile.ZIP_DEFLATED)
z.write('mod.hjson'); z.write('scripts/main.js')
z.close()
"
cp mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
git add scripts/main.js mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
git commit -m "feat(mod): add REPAIR_SLOT command for relative-slot repairs"
```

---

## Task 7 — Full test suite pass + final verification

**Step 1: Run the full test suite**

```bash
python -m pytest rl/tests/ -v
```
Expected: all tests pass (previously 39, now more).

**Step 2: Verify no import errors or lint issues**

```bash
python -c "from rl.env.mindustry_env import MindustryEnv; from rl.env.spaces import make_action_space, make_obs_space; print('OK')"
```

**Step 3: Commit if anything was missed**

```bash
git add -A
git status
# only commit if there are changes
git commit -m "chore: ensure all tests pass for player-agent redesign"
```

---

## Summary of all new mod commands

| Command | Format | Description |
|---|---|---|
| `PLAYER_MOVE` | `PLAYER_MOVE;dir` | Moves player unit 3 tiles in direction 0-7 |
| `PLAYER_BUILD` | `PLAYER_BUILD;block;slot` | Builds block at slot 0-8 around player unit |
| `REPAIR_SLOT` | `REPAIR_SLOT;slot` | Repairs building at slot 0-8 around player unit |

## Summary of Python changes

| File | Change |
|---|---|
| `rl/env/spaces.py` | `OBS_FEATURES_DIM` 43→47, `make_action_space` → `[7,9]`, +player fields in `_parse_features` |
| `rl/env/mindustry_env.py` | `_execute_action(type, arg)`, `terminated` checks `player.alive`, new action dispatch |
| `rl/rewards/multi_objective.py` | Power balance + build efficiency + player alive bonus; player death → -0.5 penalty |
| `rl/tests/*.py` | All tests updated to match new shapes, commands, and termination logic |
