# Option D Expansion — Actions + Observations + Dashboard

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the RL agent's action space from 7 to 12 actions, add 4 new resource observations, and update the dashboard — all using a registry-driven design so adding future actions/resources requires touching a single constant.

**Architecture:** Introduce `ACTION_REGISTRY` (a list of `ActionDef` named tuples) in `spaces.py` as the single source of truth for action count, block names, and mask conditions. `NUM_ACTION_TYPES` is derived from it automatically. New resources are appended at the end of the feature vector (indices 79–82) to avoid shifting existing indices. Dashboard gets pie charts for both action dist panels and a new "Extended Resources" line chart.

**Tech Stack:** Python, NumPy, Gymnasium, Matplotlib, Stable-Baselines3, pytest

---

## Current State

```
Action space:  MultiDiscrete([7, 9])   → mask shape (16,)
Obs features:  79 dims                 → feat[1-7]   = 7 resources
Test suite:    135 tests passing
```

## Target State

```
Action space:  MultiDiscrete([12, 9])  → mask shape (21,)
Obs features:  83 dims                 → feat[1-7]  = 7 resources (unchanged)
                                        feat[79-82] = silicon, oil, water, metaglass
Test suite:    all tests passing
```

---

## Task 1 — ACTION_REGISTRY in spaces.py

**Files:**
- Modify: `rl/env/spaces.py`

**What to build:**

Replace the scattered `BLOCK_TURRET/WALL/POWER/DRILL` constants and if-chains with a registry-driven design.

Add these imports at the top of `spaces.py` (after existing imports):
```python
from typing import Callable, NamedTuple

class ActionDef(NamedTuple):
    name: str
    block: str | None
    mask_fn: Callable[[dict], bool]
```

Replace the four `BLOCK_*` constants and `NUM_ACTION_TYPES = 7` with:
```python
ACTION_REGISTRY: list[ActionDef] = [
    ActionDef("WAIT",                None,                    lambda r: True),
    ActionDef("MOVE",                None,                    lambda r: True),
    ActionDef("BUILD_TURRET",        "duo",                   lambda r: r.get("copper", 0) >= 35),
    ActionDef("BUILD_WALL",          "copper-wall",           lambda r: r.get("copper", 0) >= 6),
    ActionDef("BUILD_POWER",         "solar-panel",           lambda r: r.get("copper", 0) >= 40 and r.get("lead", 0) >= 35),
    ActionDef("BUILD_DRILL",         "mechanical-drill",      lambda r: r.get("copper", 0) >= 12),
    ActionDef("REPAIR",              None,                    lambda r: True),
    ActionDef("BUILD_CONVEYOR",      "conveyor",              lambda r: r.get("copper", 0) >= 1),
    ActionDef("BUILD_GRAPHITE_PRESS","graphite-press",        lambda r: r.get("copper", 0) >= 75),
    ActionDef("BUILD_SILICON_SMELTER","silicon-smelter",      lambda r: r.get("copper", 0) >= 30 and r.get("lead", 0) >= 30),
    ActionDef("BUILD_COMBUSTION_GEN","combustion-generator",  lambda r: r.get("copper", 0) >= 25 and r.get("lead", 0) >= 15),
    ActionDef("BUILD_PNEUMATIC_DRILL","pneumatic-drill",      lambda r: r.get("copper", 0) >= 12 and r.get("graphite", 0) >= 10),
]

NUM_ACTION_TYPES: int = len(ACTION_REGISTRY)

ACTION_NAMES: list[str] = [a.name for a in ACTION_REGISTRY]

def _action_idx(name: str) -> int:
    for i, a in enumerate(ACTION_REGISTRY):
        if a.name == name:
            return i
    raise ValueError(f"Unknown action: {name}")

ACTION_WAIT   = _action_idx("WAIT")
ACTION_MOVE   = _action_idx("MOVE")
ACTION_REPAIR = _action_idx("REPAIR")

BLOCK_TURRET = ACTION_REGISTRY[_action_idx("BUILD_TURRET")].block
BLOCK_WALL   = ACTION_REGISTRY[_action_idx("BUILD_WALL")].block
BLOCK_POWER  = ACTION_REGISTRY[_action_idx("BUILD_POWER")].block
BLOCK_DRILL  = ACTION_REGISTRY[_action_idx("BUILD_DRILL")].block
```

Add new block IDs to `BLOCK_IDS` dict (after existing entries):
```python
"conveyor": 11,           # already present — keep
"graphite-press": 31,
"silicon-smelter": 32,
"combustion-generator": 33,
"pneumatic-drill": 10,    # already present — keep
```
Only add the ones that don't already exist: `graphite-press` (31), `silicon-smelter` (32), `combustion-generator` (33).

Replace `compute_action_mask` body with registry-driven logic:
```python
def compute_action_mask(state: dict) -> np.ndarray:
    mask = np.ones(NUM_ACTION_TYPES + NUM_SLOTS, dtype=np.bool_)

    player = state.get("player", {})
    if not player.get("alive", False):
        mask[1:NUM_ACTION_TYPES] = False
        return mask

    resources = state.get("resources", {})
    buildings = state.get("buildings", [])

    for i, action in enumerate(ACTION_REGISTRY):
        if i in (ACTION_WAIT, ACTION_MOVE):
            continue
        if i == ACTION_REPAIR:
            if len(buildings) == 0:
                mask[i] = False
            continue
        if not action.mask_fn(resources):
            mask[i] = False

    return mask
```

**Step 1: Write the failing test**

In `rl/tests/test_spaces.py`, add (before running):
```python
def test_num_action_types_matches_registry():
    from rl.env.spaces import ACTION_REGISTRY, NUM_ACTION_TYPES
    assert NUM_ACTION_TYPES == len(ACTION_REGISTRY)
    assert NUM_ACTION_TYPES == 12

def test_action_names_length_matches_num_action_types():
    from rl.env.spaces import ACTION_NAMES, NUM_ACTION_TYPES
    assert len(ACTION_NAMES) == NUM_ACTION_TYPES

def test_new_action_masks_no_resources():
    from rl.env.spaces import compute_action_mask, NUM_ACTION_TYPES
    broke_state = {
        **MINIMAL_STATE,
        "resources": {"copper": 0, "lead": 0, "graphite": 0},
        "buildings": [],
    }
    mask = compute_action_mask(broke_state)
    assert mask.shape == (NUM_ACTION_TYPES + NUM_SLOTS,)
    assert mask[7]  == False  # BUILD_CONVEYOR (needs copper >= 1)
    assert mask[8]  == False  # BUILD_GRAPHITE_PRESS
    assert mask[9]  == False  # BUILD_SILICON_SMELTER
    assert mask[10] == False  # BUILD_COMBUSTION_GEN
    assert mask[11] == False  # BUILD_PNEUMATIC_DRILL

def test_new_action_masks_with_resources():
    from rl.env.spaces import compute_action_mask
    rich_state = {
        **MINIMAL_STATE,
        "resources": {"copper": 100, "lead": 100, "graphite": 50},
        "buildings": [{"block": "duo", "hp": 0.5}],
    }
    mask = compute_action_mask(rich_state)
    assert mask[7]  == True  # BUILD_CONVEYOR
    assert mask[8]  == True  # BUILD_GRAPHITE_PRESS (100 >= 75)
    assert mask[9]  == True  # BUILD_SILICON_SMELTER (100 >= 30, 100 >= 30)
    assert mask[10] == True  # BUILD_COMBUSTION_GEN (100 >= 25, 100 >= 15)
    assert mask[11] == True  # BUILD_PNEUMATIC_DRILL (100 >= 12, 50 >= 10)

def test_pneumatic_drill_needs_graphite():
    from rl.env.spaces import compute_action_mask
    state = {**MINIMAL_STATE, "resources": {"copper": 100, "lead": 100, "graphite": 0}}
    mask = compute_action_mask(state)
    assert mask[11] == False  # BUILD_PNEUMATIC_DRILL blocked by graphite == 0
```

**Step 2: Run to verify FAIL**
```bash
rl/venv/bin/python -m pytest rl/tests/test_spaces.py -k "new_action" -v
```
Expected: ImportError or AssertionError

**Step 3: Apply changes to spaces.py** as described above.

**Step 4: Run to verify PASS**
```bash
rl/venv/bin/python -m pytest rl/tests/test_spaces.py -v
```
Expected: all pass (some existing tests will need updating in Task 7 — run just the new tests here)

**Step 5: Commit**
```bash
git add rl/env/spaces.py rl/tests/test_spaces.py
git commit -m "feat: action registry — 7→12 actions, registry-driven masks"
```

---

## Task 2 — Extended Observation (79 → 83 features)

**Files:**
- Modify: `rl/env/spaces.py`

**What to build:**

Add a constant for the 4 new resources and update `OBS_FEATURES_DIM`:
```python
EXTENDED_RESOURCES = ["silicon", "oil", "water", "metaglass"]
OBS_FEATURES_DIM = 83   # was 79; 4 new resources appended at feat[79-82]
```

In `_parse_features`, append after the existing sand line (feat[7]):
```python
for i, rname in enumerate(EXTENDED_RESOURCES):
    feat[79 + i] = res.get(rname, 0.0) / 1000.0
```

Update the module docstring comment on `OBS_FEATURES_DIM`:
```python
OBS_FEATURES_DIM = 83   # 1(core_hp) + 7(resources) + 4(power) + 1(wave) + 20(enemies) + 9(friendly) + 7(player/core) + 15(nearby_ores) + 15(nearby_enemies) + 4(extended_resources) = 83
```

**Step 1: Write failing test**
```python
def test_obs_features_dim_is_83():
    from rl.env.spaces import OBS_FEATURES_DIM
    assert OBS_FEATURES_DIM == 83

def test_extended_resources_in_obs():
    from rl.env.spaces import parse_observation, EXTENDED_RESOURCES
    state = {**MINIMAL_STATE, "resources": {
        "copper": 500, "lead": 200, "graphite": 100,
        "titanium": 50, "thorium": 10, "coal": 30, "sand": 80,
        "silicon": 150, "oil": 75, "water": 200, "metaglass": 40,
    }}
    obs = parse_observation(state)
    assert obs["features"].shape == (83,)
    assert obs["features"][79] == pytest.approx(150 / 1000.0)  # silicon
    assert obs["features"][80] == pytest.approx(75  / 1000.0)  # oil
    assert obs["features"][81] == pytest.approx(200 / 1000.0)  # water
    assert obs["features"][82] == pytest.approx(40  / 1000.0)  # metaglass

def test_extended_resources_zero_when_absent():
    from rl.env.spaces import parse_observation
    obs = parse_observation(MINIMAL_STATE)  # no silicon/oil/water/metaglass
    assert obs["features"][79] == pytest.approx(0.0)
    assert obs["features"][82] == pytest.approx(0.0)
```

**Step 2: Run to verify FAIL**
```bash
rl/venv/bin/python -m pytest rl/tests/test_spaces.py -k "extended_resources or dim_is_83" -v
```

**Step 3: Apply changes** as described above.

**Step 4: Run to verify PASS**
```bash
rl/venv/bin/python -m pytest rl/tests/test_spaces.py -k "extended_resources or dim_is_83" -v
```

**Step 5: Commit**
```bash
git add rl/env/spaces.py rl/tests/test_spaces.py
git commit -m "feat: obs expanded 79→83, add silicon/oil/water/metaglass features"
```

---

## Task 3 — Update mindustry_env.py

**Files:**
- Modify: `rl/env/mindustry_env.py`

**What to build:**

1. Update imports — add `ACTION_REGISTRY, ACTION_NAMES, ACTION_WAIT, ACTION_MOVE, ACTION_REPAIR`:
```python
from rl.env.spaces import (
    make_obs_space, make_action_space, parse_observation,
    compute_action_mask, NUM_ACTION_TYPES, NUM_SLOTS,
    ACTION_REGISTRY, ACTION_WAIT, ACTION_MOVE, ACTION_REPAIR,
    BLOCK_TURRET, BLOCK_WALL, BLOCK_POWER, BLOCK_DRILL,
)
```

2. Update module docstring to reflect 12 actions (list all 12 with indices).

3. Replace `_execute_action` with registry-driven dispatch:
```python
def _execute_action(self, action_type: int, arg: int) -> None:
    if action_type == ACTION_WAIT:
        return
    if action_type == ACTION_MOVE:
        self._client.send_command(f"PLAYER_MOVE;{arg % 8}")
        return
    if action_type == ACTION_REPAIR:
        self._client.send_command(f"REPAIR_SLOT;{arg}")
        return
    if 0 <= action_type < len(ACTION_REGISTRY):
        block = ACTION_REGISTRY[action_type].block
        if block is not None:
            self._client.send_command(f"PLAYER_BUILD;{block};{arg}")
            return
    raise ValueError(f"Invalid action_type: {action_type}")
```

4. Update `action_masks` docstring: `Shape: (21,) = 12 action_types + 9 slots`.

**Step 1: Write failing tests for new actions**
```python
def test_step_build_conveyor_sends_player_build():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([7, 3], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;conveyor;3")

def test_step_build_graphite_press_sends_player_build():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([8, 0], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;graphite-press;0")

def test_step_build_silicon_smelter_sends_player_build():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([9, 1], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;silicon-smelter;1")

def test_step_build_combustion_gen_sends_player_build():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([10, 2], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;combustion-generator;2")

def test_step_build_pneumatic_drill_sends_player_build():
    client = make_mock_client(states=[MOCK_STATE, MOCK_STATE, MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    action = np.array([11, 5], dtype=np.int64)
    env.step(action)
    client.send_command.assert_any_call("PLAYER_BUILD;pneumatic-drill;5")
```

**Step 2: Run to verify FAIL**
```bash
rl/venv/bin/python -m pytest rl/tests/test_env.py -k "conveyor or graphite_press or silicon_smelter or combustion_gen or pneumatic_drill" -v
```

**Step 3: Apply changes** to `mindustry_env.py` as described.

**Step 4: Run to verify PASS**
```bash
rl/venv/bin/python -m pytest rl/tests/test_env.py -k "conveyor or graphite_press or silicon_smelter or combustion_gen or pneumatic_drill" -v
```

**Step 5: Commit**
```bash
git add rl/env/mindustry_env.py rl/tests/test_env.py
git commit -m "feat: env dispatches 12 actions via registry, no hardcoded if-chain"
```

---

## Task 4 — Update multi_objective.py

**Files:**
- Modify: `rl/rewards/multi_objective.py`

**What to build:**

Replace the hardcoded action index constants with registry-derived ones:
```python
from rl.env.spaces import ACTION_REGISTRY, _action_idx

ACTION_WAIT   = _action_idx("WAIT")
ACTION_MOVE   = _action_idx("MOVE")
ACTION_BUILD_TURRET = _action_idx("BUILD_TURRET")
ACTION_BUILD_WALL   = _action_idx("BUILD_WALL")
ACTION_BUILD_POWER  = _action_idx("BUILD_POWER")
ACTION_BUILD_DRILL  = _action_idx("BUILD_DRILL")
ACTION_REPAIR       = _action_idx("REPAIR")
```

Update `CURRICULUM_PHASES` — the phase action lists now reference the derived constants above (they stay the same values 0-6, they just aren't hardcoded). No change needed to phase logic itself.

No other changes — reward computations don't check specific action indices beyond WAIT/MOVE.

**Step 1: Run existing reward tests to establish baseline**
```bash
rl/venv/bin/python -m pytest rl/tests/test_reward.py -v
```
All should pass (no changes to reward logic).

**Step 2: Apply import changes** to `multi_objective.py`.

**Step 3: Run reward tests again to verify nothing broke**
```bash
rl/venv/bin/python -m pytest rl/tests/test_reward.py -v
```
Expected: all pass.

**Step 4: Commit**
```bash
git add rl/rewards/multi_objective.py
git commit -m "refactor: multi_objective derives action indices from registry"
```

---

## Task 5 — Update training_callbacks.py

**Files:**
- Modify: `rl/callbacks/training_callbacks.py`

**What to build:**

1. Import `ACTION_NAMES` and `NUM_ACTION_TYPES` from spaces:
```python
from rl.env.spaces import ACTION_NAMES, NUM_ACTION_TYPES
```

2. In `_compute_episode_metrics`, replace:
```python
ACTION_NAMES = ["WAIT", "MOVE", "BUILD_TURRET", ...]  # remove this local definition
action_counts = [0] * 7                               # replace with NUM_ACTION_TYPES
...
if action_idx is not None and 0 <= action_idx < 7:    # replace 7 with NUM_ACTION_TYPES
...
action_dist = {name: 1.0 / 7 for name in ACTION_NAMES}  # replace 7 with NUM_ACTION_TYPES
```

3. In `_compute_policy_metrics`, replace all hardcoded `7` with `NUM_ACTION_TYPES`:
```python
action_type_dist = (
    np.bincount(action_types, minlength=NUM_ACTION_TYPES) / max(len(action_types), 1)
).tolist()
...
return {"action_type_distribution": [1 / NUM_ACTION_TYPES] * NUM_ACTION_TYPES, ...}
```

**Step 1: Run existing callback tests to establish baseline**
```bash
rl/venv/bin/python -m pytest rl/tests/test_callbacks.py -v 2>/dev/null || echo "no callback tests"
```

**Step 2: Apply changes** as described.

**Step 3: Run full test suite to catch regressions**
```bash
rl/venv/bin/python -m pytest rl/tests/ -v 2>&1 | tail -20
```

**Step 4: Commit**
```bash
git add rl/callbacks/training_callbacks.py
git commit -m "refactor: callbacks use ACTION_NAMES/NUM_ACTION_TYPES from registry"
```

---

## Task 6 — Dashboard (pie charts + extended resources panel)

**Files:**
- Modify: `rl/dashboard.py`
- Modify: `rl/tests/test_dashboard.py`

**What to build:**

### 6a — Fix `_ACTION_LABELS` to use registry

Replace the hardcoded list:
```python
from rl.env.spaces import ACTION_NAMES
_ACTION_LABELS = ACTION_NAMES
```

### 6b — Fix `_draw_action_dist_per_episode` (already pie, remove stale comment)

The function was already rewritten to a pie chart. Remove the inline comment `# Filter out zero-value slices to keep pie readable` — make the code speak for itself by renaming the variable or relying on the filter being self-evident.

### 6c — Replace `_draw_action_dist_rolling` bar chart with pie chart

Replace the entire `_draw_action_dist_rolling` function body (keeping the same signature):
```python
def _draw_action_dist_rolling(ax, metrics: list) -> None:
    ax.cla()
    _style_ax(ax, "Action Distribution (Rolling 100ep Avg)")
    if not metrics or len(metrics) < 2:
        ax.text(0.5, 0.5, "Aguardando dados…", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return

    window_metrics = metrics[-100:] if len(metrics) >= 100 else metrics
    action_counts: dict = {}
    for m in window_metrics:
        for action, pct in m.get("episode_metrics", {}).get("action_dist", {}).items():
            action_counts[action] = action_counts.get(action, 0) + pct

    if not action_counts:
        ax.text(0.5, 0.5, "Aguardando dados…", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return

    n = len(window_metrics)
    labels = list(action_counts.keys())
    values = [v / n for v in action_counts.values()]
    colors = [_PALETTE["blue"], _PALETTE["teal"], _PALETTE["red"], _PALETTE["yellow"],
              _PALETTE["peach"], _PALETTE["green"], _PALETTE["purple"],
              _PALETTE["blue"], _PALETTE["teal"], _PALETTE["red"], _PALETTE["yellow"],
              _PALETTE["peach"]]

    nonzero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not nonzero:
        ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return
    nz_labels, nz_values, nz_colors = zip(*nonzero)

    wedges, _, autotexts = ax.pie(
        nz_values,
        labels=None,
        colors=nz_colors,
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        startangle=90,
        wedgeprops={"edgecolor": _PALETTE["bg_fig"], "linewidth": 0.8},
        textprops={"fontsize": 7, "color": _PALETTE["text"]},
    )
    for at in autotexts:
        at.set_fontsize(6)
        at.set_color(_PALETTE["bg_fig"])

    ax.legend(
        wedges, nz_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(4, len(nz_labels)),
        fontsize=6,
        frameon=False,
        labelcolor=_PALETTE["subtext"],
    )
```

### 6d — Add `_draw_extended_resources` panel

Add a new function that plots silicon/oil/water/metaglass over time from `world.resources`:
```python
def _draw_extended_resources(ax, metrics: list) -> None:
    ax.cla()
    _style_ax(ax, "Extended Resources")
    if not metrics:
        ax.text(0.5, 0.5, "Aguardando dados…", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=_PALETTE["subtext"])
        return

    resource_series: dict[str, list[float]] = {
        "silicon": [], "oil": [], "water": [], "metaglass": []
    }
    for m in metrics:
        world_res = m.get("world", {}).get("resources", {})
        for rname in resource_series:
            resource_series[rname].append(float(world_res.get(rname, 0.0)))

    colors = [_PALETTE["teal"], _PALETTE["peach"], _PALETTE["blue"], _PALETTE["purple"]]
    xs = list(range(len(metrics)))
    any_data = False
    for (rname, series), color in zip(resource_series.items(), colors):
        if any(v > 0 for v in series):
            ax.plot(xs, series, color=color, linewidth=1.2, label=rname)
            any_data = True

    if any_data:
        ax.legend(fontsize=6, frameon=False, labelcolor=_PALETTE["subtext"])
    else:
        ax.text(0.5, 0.5, "Sem silicon/oil/water/metaglass", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=_PALETTE["subtext"])
```

### 6e — Wire new panel into build_figure() and make_updater()

In `build_figure()`, replace `ax_action_dist_roll = fig.add_subplot(gs[5, 2])` with the new panel assignment or add a new row. The simplest approach: replace `ax_action_dist_roll` slot (gs[5,2]) with the new extended resources panel and name it `ax_ext_resources`. This reuses the grid slot cleanly.

Actually: keep both pie charts and add the new panel by replacing an unused area or adding a new row. Looking at the current 7-row grid (rows 0-6), row 6 is `ax_stats`. A clean approach is to add a new row 6 and move stats to row 7, making it an 8-row grid. But that changes the figure height.

**Simpler approach**: place `ax_ext_resources` at `gs[5, 2]` (replacing `ax_action_dist_roll`) and move `ax_action_dist_roll` to an earlier empty slot. But there's no empty slot.

**Best approach**: Add one new row. Change `GridSpec(7, 3, ...)` to `GridSpec(8, 3, ...)` and `height_ratios` from `[2.5, 2.0, 2.0, 1.8, 1.8, 1.8, 0.7]` to `[2.5, 2.0, 2.0, 1.8, 1.8, 1.8, 1.8, 0.7]`. Place `ax_ext_resources` at `gs[6, :]` (full width) and move `ax_stats` to `gs[7, :]`.

In `make_updater` update function, add:
```python
_draw_extended_resources(axes["ext_resources"], metrics_history)
```

**Step 1: Write failing dashboard tests**
```python
def test_draw_action_dist_rolling_renders_pie():
    from rl.dashboard import _draw_action_dist_rolling
    metrics = [
        {"episode_metrics": {"action_dist": {
            "WAIT": 0.1, "MOVE": 0.2, "BUILD_TURRET": 0.15,
            "BUILD_WALL": 0.15, "BUILD_POWER": 0.2, "BUILD_DRILL": 0.15, "REPAIR": 0.05
        }}} for _ in range(10)
    ]
    ax = plt.subplot(1, 1, 1)
    _draw_action_dist_rolling(ax, metrics)
    assert ax.get_title() != ""
    plt.close("all")

def test_draw_extended_resources_renders_with_data():
    from rl.dashboard import _draw_extended_resources
    metrics = [
        {"world": {"resources": {"silicon": 100.0, "oil": 50.0, "water": 200.0, "metaglass": 30.0}}}
        for _ in range(5)
    ]
    ax = plt.subplot(1, 1, 1)
    _draw_extended_resources(ax, metrics)
    assert ax.get_title() != ""
    plt.close("all")

def test_draw_extended_resources_handles_empty():
    from rl.dashboard import _draw_extended_resources
    ax = plt.subplot(1, 1, 1)
    _draw_extended_resources(ax, [])
    assert ax.get_title() != ""
    plt.close("all")
```

**Step 2: Run to verify FAIL**
```bash
rl/venv/bin/python -m pytest rl/tests/test_dashboard.py -k "extended_resources or rolling_renders_pie" -v
```

**Step 3: Apply all 6a-6e changes** to `dashboard.py`.

**Step 4: Run to verify PASS**
```bash
rl/venv/bin/python -m pytest rl/tests/test_dashboard.py -v
```

**Step 5: Commit**
```bash
git add rl/dashboard.py rl/tests/test_dashboard.py
git commit -m "feat: dashboard pie charts + extended resources panel"
```

---

## Task 7 — Fix existing tests broken by the expansion

**Files:**
- Modify: `rl/tests/test_spaces.py`
- Modify: `rl/tests/test_env.py`
- Modify: `rl/tests/test_dashboard.py`

**What breaks and how to fix:**

### test_spaces.py

| Test | Change |
|------|--------|
| `test_obs_space_shape` | `(79,)` → `(83,)` |
| `test_parse_observation_returns_correct_shapes` | `(79,)` → `(83,)` |
| `test_action_mask_shape` | already uses `NUM_ACTION_TYPES + NUM_SLOTS` — no change needed |
| `test_action_mask_no_resources_blocks_build` | mask shape is now 21; indices 7-11 are new blocks — must also assert `mask[7] == False` through `mask[11] == False` |
| `test_action_mask_with_enough_resources` | `rich_state` needs `graphite >= 10` for pneumatic drill (`mask[:12]`) — add graphite to rich_state resources |
| `test_action_mask_partial_resources` | now mask has 21 entries; add assertions for new actions |
| `test_action_space_structure` | uses `NUM_ACTION_TYPES` via import — auto-correct once constant changes |

### test_env.py

| Test | Change |
|------|--------|
| `test_reset_returns_valid_obs` | `(79,)` → `(83,)` |
| `test_step_returns_five_tuple` | `(79,)` → `(83,)` |
| `test_action_masks_returns_correct_shape` | `(16,)` → `(21,)` |
| `test_action_masks_before_reset_returns_all_true` | `(16,)` → `(21,)` |

### test_dashboard.py

| Test | Change |
|------|--------|
| `test_draw_action_dist_per_episode_renders_with_valid_metrics` | Add new action names: `"BUILD_CONVEYOR": 0.02, "BUILD_GRAPHITE_PRESS": 0.02, ...` etc. so distribution sums to 1.0 |
| `test_draw_action_dist_rolling_renders_with_valid_metrics` | Same as above |

**Step 1: Run full suite to see all failures**
```bash
rl/venv/bin/python -m pytest rl/tests/ -v 2>&1 | grep -E "FAILED|ERROR"
```

**Step 2: Fix each failing test** per the table above.

**Step 3: Run full suite to verify all pass**
```bash
rl/venv/bin/python -m pytest rl/tests/ -v
```
Expected: all tests pass.

**Step 4: Commit**
```bash
git add rl/tests/
git commit -m "test: update all tests for 12 actions, 83-dim obs, 21-dim mask"
```

---

## Task 8 — Final smoke check

**Step 1: Run full test suite**
```bash
rl/venv/bin/python -m pytest rl/tests/ -v --tb=short
```
Expected: 150+ tests, all green.

**Step 2: Verify action count summary**
```bash
rl/venv/bin/python -c "from rl.env.spaces import ACTION_REGISTRY, NUM_ACTION_TYPES, OBS_FEATURES_DIM, NUM_SLOTS; print(f'Actions: {NUM_ACTION_TYPES}, Obs: {OBS_FEATURES_DIM}, Mask: {NUM_ACTION_TYPES + NUM_SLOTS}'); [print(f'  {i}: {a.name} ({a.block or \"no-block\"})') for i, a in enumerate(ACTION_REGISTRY)]"
```
Expected output:
```
Actions: 12, Obs: 83, Mask: 21
  0: WAIT (no-block)
  1: MOVE (no-block)
  2: BUILD_TURRET (duo)
  ...
  11: BUILD_PNEUMATIC_DRILL (pneumatic-drill)
```

**Step 3: Final commit**
```bash
git add -A
git commit -m "feat: option-d complete — 12 actions, 83-dim obs, registry-driven masks, dashboard pie+resources"
```
