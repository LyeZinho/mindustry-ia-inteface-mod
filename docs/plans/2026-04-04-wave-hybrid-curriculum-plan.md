# Wave-Hybrid Curriculum, Lookahead Fix & Placement Signal — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the agent build drills+conveyors from step 1, unlock defense by wave 3, fix the lookahead simulator's pessimism, and sharpen placement spatial signals.

**Architecture:** Three independent modules: curriculum logic in `multi_objective.py`, lookahead scoring in `lookahead.py`, placement scoring in `placement.py`. The env wires wave number to the curriculum. All changes are backward-compatible (new parameters default to zero/off).

**Tech Stack:** Python 3, pytest, `PYTHONPATH=. venv/bin/python -m pytest rl/tests/ -q`

---

### Task 1: Restructure CURRICULUM_PHASES and add wave parameter to apply_curriculum_action_mask

**Files:**
- Modify: `rl/rewards/multi_objective.py` — `CURRICULUM_PHASES`, `_REWARD_WEIGHT_PHASES`, `apply_curriculum_action_mask`
- Test: `rl/tests/test_reward.py`

**Step 1: Write the failing tests**

Add to `rl/tests/test_reward.py`:

```python
def test_curriculum_phase0_includes_conveyor():
    from rl.rewards.multi_objective import apply_curriculum_action_mask
    from rl.env.spaces import ACTION_BUILD_CONVEYOR
    mask = apply_curriculum_action_mask(timestep=0, wave=0)
    assert mask[ACTION_BUILD_CONVEYOR] is True


def test_curriculum_wave3_unlocks_turret_and_wall():
    from rl.rewards.multi_objective import apply_curriculum_action_mask
    from rl.env.spaces import ACTION_BUILD_TURRET, ACTION_BUILD_WALL
    mask = apply_curriculum_action_mask(timestep=0, wave=3)
    assert mask[ACTION_BUILD_TURRET] is True
    assert mask[ACTION_BUILD_WALL] is True


def test_curriculum_wave2_does_not_unlock_turret():
    from rl.rewards.multi_objective import apply_curriculum_action_mask
    from rl.env.spaces import ACTION_BUILD_TURRET
    mask = apply_curriculum_action_mask(timestep=0, wave=2)
    assert mask[ACTION_BUILD_TURRET] is False


def test_curriculum_timestep30k_unlocks_turret():
    from rl.rewards.multi_objective import apply_curriculum_action_mask
    from rl.env.spaces import ACTION_BUILD_TURRET
    mask = apply_curriculum_action_mask(timestep=30_000, wave=0)
    assert mask[ACTION_BUILD_TURRET] is True


def test_curriculum_wave5_unlocks_power():
    from rl.rewards.multi_objective import apply_curriculum_action_mask
    from rl.env.spaces import ACTION_BUILD_POWER
    mask = apply_curriculum_action_mask(timestep=0, wave=5)
    assert mask[ACTION_BUILD_POWER] is True


def test_curriculum_wave8_unlocks_all():
    from rl.rewards.multi_objective import apply_curriculum_action_mask
    from rl.env.spaces import NUM_ACTION_TYPES
    mask = apply_curriculum_action_mask(timestep=0, wave=8)
    assert sum(mask) == NUM_ACTION_TYPES
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/test_reward.py::test_curriculum_phase0_includes_conveyor rl/tests/test_reward.py::test_curriculum_wave3_unlocks_turret_and_wall -v
```
Expected: FAIL — conveyor not in phase 0, wave parameter doesn't exist yet

**Step 3: Implement**

In `rl/rewards/multi_objective.py`, replace `CURRICULUM_PHASES` and `apply_curriculum_action_mask`:

```python
CURRICULUM_PHASES = [
    ("bootstrap", (0, 0), [
        ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL, ACTION_BUILD_CONVEYOR, ACTION_DELETE,
    ]),
    ("drill_defend", (30_000, 3), [
        ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL, ACTION_BUILD_CONVEYOR,
        ACTION_BUILD_WALL, ACTION_BUILD_TURRET, ACTION_DELETE,
    ]),
    ("advanced", (100_000, 5), [
        ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL, ACTION_BUILD_CONVEYOR,
        ACTION_BUILD_WALL, ACTION_BUILD_TURRET, ACTION_BUILD_POWER,
        ACTION_REPAIR, ACTION_BUILD_PNEUMATIC_DRILL, ACTION_BUILD_COMBUSTION_GEN, ACTION_DELETE,
    ]),
    ("full", (300_000, 8), list(range(13))),
]
```

Each tuple `(ts_threshold, wave_threshold)` means: unlock when `timestep >= ts_threshold OR wave >= wave_threshold`.

Replace `_REWARD_WEIGHT_PHASES` thresholds to match:

```python
_REWARD_WEIGHT_PHASES = [
    ("bootstrap",    (0,         30_000),         {"survival": 0.3, "economy": 1.0, "defense": 0.1, "build": 0.2}),
    ("drill_defend", (30_000,    100_000),         {"survival": 0.5, "economy": 0.8, "defense": 0.3, "build": 0.4}),
    ("advanced",     (100_000,   300_000),         {"survival": 0.8, "economy": 0.6, "defense": 0.8, "build": 0.5}),
    ("full",         (300_000,   float("inf")),    {"survival": 0.8, "economy": 0.8, "defense": 0.8, "build": 0.6}),
]
```

Replace `apply_curriculum_action_mask`:

```python
def apply_curriculum_action_mask(
    timestep: int,
    wave: int = 0,
) -> list[bool]:
    from rl.env.spaces import NUM_ACTION_TYPES

    if not CURRICULUM_ENABLED:
        return [True] * NUM_ACTION_TYPES

    current_phase_actions = CURRICULUM_PHASES[0][2]

    for _phase_name, (ts_threshold, wave_threshold), actions in CURRICULUM_PHASES:
        if timestep >= ts_threshold or wave >= wave_threshold:
            current_phase_actions = actions

    mask = [False] * NUM_ACTION_TYPES
    for action_idx in current_phase_actions:
        if 0 <= action_idx < NUM_ACTION_TYPES:
            mask[action_idx] = True
    return mask
```

Also update the module docstring section to reflect new phase names/thresholds:
```
4-phase curriculum for 13 actions:
  Phase 0 bootstrap  (always):           WAIT, MOVE, DRILL, CONVEYOR, DELETE
  Phase 1 drill_defend (wave≥3 OR 30k):  + WALL, TURRET
  Phase 2 advanced   (wave≥5 OR 100k):  + POWER, REPAIR, PNEUMATIC_DRILL, COMBUSTION_GEN
  Phase 3 full       (wave≥8 OR 300k):  all 13
```

**Step 4: Run new tests**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/test_reward.py::test_curriculum_phase0_includes_conveyor rl/tests/test_reward.py::test_curriculum_wave3_unlocks_turret_and_wall rl/tests/test_reward.py::test_curriculum_wave2_does_not_unlock_turret rl/tests/test_reward.py::test_curriculum_timestep30k_unlocks_turret rl/tests/test_reward.py::test_curriculum_wave5_unlocks_power rl/tests/test_reward.py::test_curriculum_wave8_unlocks_all -v
```
Expected: 6 PASS

**Step 5: Run existing curriculum tests to catch regressions**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/test_reward.py -v -k curriculum
```
Expected: all pass. If `test_curriculum_phase0_includes_delete` or related tests fail because they call `apply_curriculum_action_mask(timestep=...)` without `wave=`, they still pass since `wave=0` is the default.

**Step 6: Run full suite**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/ -q
```
Expected: all pass

**Step 7: Commit**

```bash
git add rl/rewards/multi_objective.py rl/tests/test_reward.py
git -c user.email="agent@omo" -c user.name="Agent" commit -m "feat(curriculum): wave-hybrid phases — conveyor in p0, wall/turret in p1, OR-gated thresholds"
```

---

### Task 2: Pass wave from game state to apply_curriculum_action_mask in env

**Files:**
- Modify: `rl/env/mindustry_env.py` — `action_masks` method (~line 234)
- Test: `rl/tests/test_env.py`

**Step 1: Write the failing test**

Add to `rl/tests/test_env.py`:

```python
def test_action_masks_wave3_unlocks_turret(monkeypatch):
    """action_masks passes wave from prev_state to curriculum gating."""
    import numpy as np
    from rl.env.mindustry_env import MindustryEnv
    from rl.env.spaces import ACTION_BUILD_TURRET, NUM_ACTION_TYPES, NUM_SLOTS

    env = MindustryEnv.__new__(MindustryEnv)
    env._global_timestep = 0
    env._prev_state = {
        "wave": 3,
        "player": {"x": 10, "y": 10, "alive": True},
        "resources": {"copper": 500},
        "buildings": [], "grid": [], "blockedTiles": [], "oreGrid": [],
    }

    import rl.rewards.multi_objective as mo
    orig = mo.CURRICULUM_ENABLED
    mo.CURRICULUM_ENABLED = True
    try:
        mask = env.action_masks()
    finally:
        mo.CURRICULUM_ENABLED = orig

    assert mask[ACTION_BUILD_TURRET] is True
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/test_env.py::test_action_masks_wave3_unlocks_turret -v
```
Expected: FAIL — mask[ACTION_BUILD_TURRET] is False (wave not passed, defaults to 0)

**Step 3: Implement**

In `rl/env/mindustry_env.py`, change the `action_masks` method from:

```python
        from rl.rewards.multi_objective import apply_curriculum_action_mask, CURRICULUM_ENABLED
        if CURRICULUM_ENABLED:
            curriculum_mask = apply_curriculum_action_mask(self._global_timestep)
```

to:

```python
        from rl.rewards.multi_objective import apply_curriculum_action_mask, CURRICULUM_ENABLED
        if CURRICULUM_ENABLED:
            curr_wave = int(self._prev_state.get("wave", 0))
            curriculum_mask = apply_curriculum_action_mask(self._global_timestep, wave=curr_wave)
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/test_env.py::test_action_masks_wave3_unlocks_turret -v
```
Expected: PASS

**Step 5: Run full suite**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/ -q
```
Expected: all pass

**Step 6: Commit**

```bash
git add rl/env/mindustry_env.py rl/tests/test_env.py
git -c user.email="agent@omo" -c user.name="Agent" commit -m "feat(env): pass wave to curriculum mask — wave-based phase unlock wired end-to-end"
```

---

### Task 3: Lookahead — add existing building income to simulation

**Files:**
- Modify: `rl/optimization/lookahead.py`
- Test: `rl/tests/test_optimization_lookahead.py`

**Step 1: Write the failing test**

Add to `rl/tests/test_optimization_lookahead.py`:

```python
def test_existing_drills_increase_lookahead_score():
    """An agent with placed drills should see higher lookahead score for build actions
    (more resources available in future steps) than one without drills."""
    from rl.optimization.lookahead import compute_lookahead_scores
    from rl.env.spaces import ACTION_BUILD_DRILL

    base_state = {
        "player": {"x": 10, "y": 10, "alive": True},
        "resources": {"copper": 50},
        "buildings": [],
        "power": {"produced": 0.0, "consumed": 0.0},
        "oreGrid": [],
    }
    state_with_drills = {
        **base_state,
        "buildings": [
            {"block": "mechanical-drill", "team": "sharded", "x": 11, "y": 11},
            {"block": "mechanical-drill", "team": "sharded", "x": 12, "y": 11},
        ],
    }

    scores_no_drills = compute_lookahead_scores(base_state)
    scores_with_drills = compute_lookahead_scores(state_with_drills)

    assert scores_with_drills[ACTION_BUILD_DRILL] > scores_no_drills[ACTION_BUILD_DRILL]
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/test_optimization_lookahead.py::test_existing_drills_increase_lookahead_score -v
```
Expected: FAIL — scores identical regardless of placed drills

**Step 3: Implement**

In `rl/optimization/lookahead.py`, replace `compute_lookahead_scores`:

```python
def compute_lookahead_scores(state: dict) -> np.ndarray:
    scores = np.zeros(NUM_ACTION_TYPES, dtype=np.float32)

    player = state.get("player", {})
    if not player.get("alive", False):
        return scores

    existing_income: dict = {}
    for building in state.get("buildings", []):
        block = building.get("block", "")
        for res, val in BUILDING_INCOME.get(block, {}).items():
            if res != "power_delta":
                existing_income[res] = existing_income.get(res, 0.0) + val

    for action_idx, action_def in enumerate(ACTION_REGISTRY):
        sim_state = copy.deepcopy(state)
        total_score = 0.0

        if action_def.block is None:
            ally_buildings = [
                b for b in state.get("buildings", [])
                if b.get("team", "") == "sharded" and "core" not in b.get("block", "")
            ]
            if ally_buildings:
                total_score = 0.5
            scores[action_idx] = math.log(1 + total_score) / math.log(1 + 10.0)
            continue

        for step in range(3):
            resources = sim_state.get("resources", {})
            for res, val in existing_income.items():
                resources[res] = resources.get(res, 0.0) + val

            block_name = action_def.block
            cost = BUILD_COSTS.get(block_name, {})

            can_afford = all(
                resources.get(res_name, 0) >= res_cost
                for res_name, res_cost in cost.items()
            )

            if can_afford:
                total_score += 1.0

                for res_name, res_cost in cost.items():
                    resources[res_name] = resources.get(res_name, 0) - res_cost

                income = BUILDING_INCOME.get(block_name, {})
                for income_res, income_val in income.items():
                    if income_res == "power_delta":
                        power = sim_state.get("power", {})
                        power["produced"] = power.get("produced", 0.0) + income_val
                        total_score += 0.1 * income_val / 60.0
                    else:
                        resources[income_res] = resources.get(income_res, 0) + income_val
                        total_score += 0.1 * income_val

        power = sim_state.get("power", {})
        if power.get("produced", 0.0) > power.get("consumed", 0.0):
            total_score += 0.2

        scores[action_idx] = math.log(1 + total_score) / math.log(1 + 10.0)

    return scores
```

**Step 4: Run new test**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/test_optimization_lookahead.py::test_existing_drills_increase_lookahead_score -v
```
Expected: PASS

**Step 5: Run full suite**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/ -q
```
Expected: all pass

**Step 6: Commit**

```bash
git add rl/optimization/lookahead.py rl/tests/test_optimization_lookahead.py
git -c user.email="agent@omo" -c user.name="Agent" commit -m "fix(lookahead): add existing building income to simulation + score DELETE when buildings present"
```

---

### Task 4: Placement — blocked=-0.5 and K=2.0

**Files:**
- Modify: `rl/optimization/placement.py`
- Test: `rl/tests/test_optimization_worker.py` (or placement test if it exists)

**Step 1: Check if a placement test file exists**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/ -q --collect-only 2>&1 | grep placement
```

If none exists, add the tests to `rl/tests/test_optimization_worker.py`.

**Step 2: Write the failing tests**

Add to the appropriate test file:

```python
def test_placement_blocked_tile_is_negative():
    from rl.optimization.placement import compute_placement_scores

    state = {
        "player": {"x": 10, "y": 10, "alive": True},
        "core": {"x": 5, "y": 5},
        "enemies": [],
        "blockedTiles": [[10, 10]],
        "buildings": [],
        "oreGrid": [],
    }
    scores = compute_placement_scores(state)
    assert scores[4] < 0.0


def test_placement_ore_slot_higher_than_empty_slot():
    from rl.optimization.placement import compute_placement_scores
    from rl.env.spaces import SLOT_DX, SLOT_DY

    px, py = 10, 10
    ore_slot = 0
    ore_x = px + SLOT_DX[ore_slot]
    ore_y = py + SLOT_DY[ore_slot]

    state = {
        "player": {"x": px, "y": py, "alive": True},
        "core": {"x": 5, "y": 5},
        "enemies": [],
        "blockedTiles": [],
        "buildings": [],
        "oreGrid": [[ore_x, ore_y, 1]],
    }
    scores = compute_placement_scores(state)
    assert scores[ore_slot] > scores[1]


def test_placement_scores_range():
    from rl.optimization.placement import compute_placement_scores

    state = {
        "player": {"x": 10, "y": 10, "alive": True},
        "core": {"x": 5, "y": 5},
        "enemies": [],
        "blockedTiles": [[10, 10]],
        "buildings": [],
        "oreGrid": [[11, 11, 1]],
    }
    scores = compute_placement_scores(state)
    assert all(-1.0 <= s <= 1.0 for s in scores)
```

**Step 3: Run tests to verify they fail**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/test_optimization_worker.py::test_placement_blocked_tile_is_negative -v
```
Expected: FAIL — blocked tile returns 0.0, not negative

**Step 4: Implement**

In `rl/optimization/placement.py`, change two lines:

Line with blocked tile handling — change from:
```python
        if (slot_x, slot_y) in blocked_tiles:
            scores[slot] = 0.0
            continue
```
to:
```python
        if (slot_x, slot_y) in blocked_tiles:
            scores[slot] = -0.5
            continue
```

Line with tanh — change from:
```python
        scores[slot] = math.tanh(raw_score / 3.0)
```
to:
```python
        scores[slot] = math.tanh(raw_score / 2.0)
```

**Step 5: Run new tests**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/test_optimization_worker.py::test_placement_blocked_tile_is_negative rl/tests/test_optimization_worker.py::test_placement_ore_slot_higher_than_empty_slot rl/tests/test_optimization_worker.py::test_placement_scores_range -v
```
Expected: 3 PASS

**Step 6: Run full suite**

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/ -q
```
Expected: all pass

**Step 7: Commit**

```bash
git add rl/optimization/placement.py rl/tests/test_optimization_worker.py
git -c user.email="agent@omo" -c user.name="Agent" commit -m "fix(placement): blocked=-0.5, K 3.0→2.0 — sharper ore vs empty vs blocked spatial signal"
```

---

## Final Verification

```bash
PYTHONPATH=. venv/bin/python -m pytest rl/tests/ -q
```
Expected: all tests pass, no regressions.

Spot-check the curriculum behavior manually:

```python
from rl.rewards.multi_objective import apply_curriculum_action_mask
from rl.env.spaces import ACTION_BUILD_CONVEYOR, ACTION_BUILD_TURRET, ACTION_BUILD_WALL

assert apply_curriculum_action_mask(0, wave=0)[ACTION_BUILD_CONVEYOR] is True   # phase 0
assert apply_curriculum_action_mask(0, wave=2)[ACTION_BUILD_TURRET] is False     # not yet
assert apply_curriculum_action_mask(0, wave=3)[ACTION_BUILD_TURRET] is True      # wave gate
assert apply_curriculum_action_mask(30_000, wave=0)[ACTION_BUILD_WALL] is True   # ts gate
```
