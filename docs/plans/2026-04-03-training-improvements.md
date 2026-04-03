# Training Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 6 highest-impact training issues causing avg episode length of 13 steps and avg reward of -0.985, so the RL agent can actually learn to play Mindustry.

**Architecture:** Quick config/logic changes (tasks 1-4) plus a structural switch from A2C to MaskablePPO with invalid action masking (task 5). All changes are backward-compatible with existing env/test infrastructure.

**Tech Stack:** Python 3.11+, stable-baselines3, sb3-contrib (new dep), gymnasium, Mindustry mod (JavaScript)

---

### Task 1: Slow down waves — `scripts/main.js`

**Files:**
- Modify: `scripts/main.js` (line ~907, `waveSpacing` setting)

**Step 1: Change waveSpacing from 7200 to 14400**

In `handleResetCommand`, find:
```javascript
Vars.state.rules.waveSpacing = 7200;
Log.info("[Mimi Gateway] waveSpacing set to 7200 (2x faster)");
```

Replace with:
```javascript
Vars.state.rules.waveSpacing = 14400;
Log.info("[Mimi Gateway] waveSpacing set to 14400 (normal speed)");
```

**Step 2: Verify no other references to waveSpacing**

Search for `waveSpacing` in the codebase to confirm this is the only place it's set.

**Step 3: Commit**

```bash
git add scripts/main.js
git commit -m "feat(training): slow down wave speed for early learning (14400 waveSpacing)"
```

---

### Task 2: Increase survival reward weight — `rl/rewards/multi_objective.py`

**Files:**
- Modify: `rl/rewards/multi_objective.py`
- Test: `rl/tests/test_reward.py`

**Step 1: Update the reward weights and time penalty**

Change the reward computation from:
```python
reward = (
    0.35 * core_hp_delta
    + 0.20 * wave_survived_bonus
    + 0.15 * (resources_delta / 500.0)
    + 0.10 * drill_bonus
    + 0.08 * power_balance_bonus
    + 0.07 * build_efficiency_bonus
    + 0.05 * player_alive_bonus
    - 0.0005
)
```

To:
```python
reward = (
    0.30 * core_hp_delta
    + 0.20 * wave_survived_bonus
    + 0.10 * (resources_delta / 500.0)
    + 0.08 * drill_bonus
    + 0.07 * power_balance_bonus
    + 0.05 * build_efficiency_bonus
    + 0.20 * player_alive_bonus
    - 0.002
)
```

Key changes:
- `player_alive_bonus`: 0.05 → 0.20 (4× stronger survival signal)
- `time_penalty`: 0.0005 → 0.002 (4× stronger urgency)
- `core_hp_delta`: 0.35 → 0.30 (slight reduction to make room)
- `resources_delta`: 0.15 → 0.10 (slight reduction)
- `drill_bonus`: 0.10 → 0.08 (slight reduction)
- `build_efficiency_bonus`: 0.07 → 0.05

Also update the module docstring to match.

**Step 2: Update tests**

In `test_reward.py`:
- `test_time_penalty_halved` → rename to `test_time_penalty` and update expected value from -0.0005 to -0.002.
- `test_reward_player_alive_bonus` → may need adjustment since weight changed from 0.05 to 0.20.
- `test_drill_bonus_when_copper_increases_significantly` → adjust threshold since drill_bonus weight changed from 0.10 to 0.08.

**Step 3: Run tests**

```bash
python -m pytest rl/tests/test_reward.py -v
```

**Step 4: Commit**

```bash
git add rl/rewards/multi_objective.py rl/tests/test_reward.py
git commit -m "feat(training): increase survival reward weight and time penalty"
```

---

### Task 3: Fix block encoding — `rl/env/spaces.py`

**Files:**
- Modify: `rl/env/spaces.py` (replace `_encode_block` function)
- Test: `rl/tests/test_spaces.py`

**Step 1: Add deterministic block lookup table**

Replace the hash-based `_encode_block`:
```python
def _encode_block(block: str) -> float:
    return (abs(hash(block)) % 50) / 50.0
```

With a deterministic lookup:
```python
# Deterministic block encoding — no hash collisions
BLOCK_IDS: dict[str, int] = {
    "air": 0,
    "copper-wall": 1,
    "copper-wall-large": 2,
    "duo": 3,
    "scatter": 4,
    "hail": 5,
    "lancer": 6,
    "wave": 7,
    "swarmer": 8,
    "mechanical-drill": 9,
    "pneumatic-drill": 10,
    "conveyor": 11,
    "titanium-conveyor": 12,
    "router": 13,
    "junction": 14,
    "overflow-gate": 15,
    "sorter": 16,
    "solar-panel": 17,
    "solar-panel-large": 18,
    "battery": 19,
    "battery-large": 20,
    "power-node": 21,
    "power-node-large": 22,
    "thermal-generator": 23,
    "core-shard": 24,
    "vault": 25,
    "container": 26,
    "mender": 27,
    "mend-projector": 28,
    "overdrive-projector": 29,
    "force-projector": 30,
}
_NUM_KNOWN_BLOCKS = len(BLOCK_IDS) + 1  # +1 for "unknown"


def _encode_block(block: str) -> float:
    idx = BLOCK_IDS.get(block, _NUM_KNOWN_BLOCKS - 1)
    return idx / _NUM_KNOWN_BLOCKS
```

**Step 2: Add test for deterministic encoding**

In `test_spaces.py`, add:
```python
from rl.env.spaces import _encode_block, BLOCK_IDS

def test_encode_block_deterministic():
    """Known blocks map to unique, stable floats."""
    seen = set()
    for name in BLOCK_IDS:
        val = _encode_block(name)
        assert 0.0 <= val < 1.0
        assert val not in seen, f"collision for {name}"
        seen.add(val)

def test_encode_block_unknown_maps_to_last():
    """Unknown blocks map to the 'unknown' slot."""
    val = _encode_block("nonexistent-block-xyz")
    assert 0.0 < val < 1.0
```

**Step 3: Run tests**

```bash
python -m pytest rl/tests/test_spaces.py -v
```

**Step 4: Commit**

```bash
git add rl/env/spaces.py rl/tests/test_spaces.py
git commit -m "feat(training): deterministic block encoding lookup table (no hash collisions)"
```

---

### Task 4: Tune hyperparameters — `rl/train.py`

**Files:**
- Modify: `rl/train.py` (default arg values)
- Test: `rl/tests/test_train.py`

**Step 1: Update default hyperparameters in parse_args**

Change these defaults:
```python
p.add_argument("--lr", type=float, default=7e-4)
p.add_argument("--n-steps", type=int, default=128, dest="n_steps")
```

To:
```python
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--n-steps", type=int, default=32, dest="n_steps")
```

**Step 2: Update the model construction to use new gamma and ent_coef**

This will be done as part of Task 5 (MaskablePPO switch), where gamma=0.95 and ent_coef=0.05 will be set in the new model constructor. No separate change needed here if Task 5 follows.

**Step 3: Commit**

```bash
git add rl/train.py
git commit -m "feat(training): tune hyperparams (lr=3e-4, n_steps=32)"
```

---

### Task 5: Action masking with MaskablePPO — `rl/env/mindustry_env.py` + `rl/train.py`

This is the largest change. Three sub-parts:

#### 5a: Add `sb3-contrib` dependency

**Files:**
- Modify: `rl/requirements.txt`

Add:
```
sb3-contrib>=2.0.0
```

Install:
```bash
source rl/venv/bin/activate && pip install sb3-contrib>=2.0.0
```

#### 5b: Add `action_masks()` method to MindustryEnv

**Files:**
- Modify: `rl/env/mindustry_env.py`
- Modify: `rl/env/spaces.py` (add helper for computing valid actions)

In `spaces.py`, add a helper function:
```python
def compute_action_mask(state: Dict[str, Any]) -> np.ndarray:
    """
    Return a 1D boolean mask of length NUM_ACTION_TYPES + NUM_SLOTS (7+9=16).
    
    First 7 entries: which action_types are valid
    Next 9 entries: which slots/directions are valid for the current action
    
    For simplicity, we mask at the action_type level:
    - WAIT (0): always valid
    - MOVE (1): always valid (directions that go out of bounds still move, just clamped)
    - BUILD_TURRET (2): valid if copper >= 0 (we have resources to try)
    - BUILD_WALL (3): valid if copper >= 0
    - BUILD_POWER (4): valid if copper >= 0 and lead >= 0
    - BUILD_DRILL (5): valid if copper >= 0
    - REPAIR (6): valid if any building exists in range
    
    All 9 slots are always valid (the server handles invalid placement gracefully).
    """
    mask = np.ones(NUM_ACTION_TYPES + NUM_SLOTS, dtype=np.bool_)
    
    # Player must be alive for any action except WAIT
    player = state.get("player", {})
    if not player.get("alive", False):
        mask[1:NUM_ACTION_TYPES] = False
        return mask
    
    resources = state.get("resources", {})
    copper = float(resources.get("copper", 0))
    lead = float(resources.get("lead", 0))
    
    # BUILD_TURRET (duo): needs copper (6 copper)
    if copper < 6:
        mask[2] = False
    
    # BUILD_WALL (copper-wall): needs copper (6 copper)
    if copper < 6:
        mask[3] = False
    
    # BUILD_POWER (solar-panel): needs lead (14 lead) + copper
    if lead < 14 or copper < 10:
        mask[4] = False
    
    # BUILD_DRILL (mechanical-drill): needs copper (12 copper)
    if copper < 12:
        mask[5] = False
    
    # REPAIR: valid only if there are buildings nearby
    buildings = state.get("buildings", [])
    if len(buildings) == 0:
        mask[6] = False
    
    return mask
```

In `mindustry_env.py`, add the method:
```python
def action_masks(self) -> np.ndarray:
    """Return action mask for MaskablePPO. Shape: (16,) = 7 action_types + 9 slots."""
    from rl.env.spaces import compute_action_mask
    if self._prev_state is None:
        return np.ones(16, dtype=np.bool_)
    return compute_action_mask(self._prev_state)
```

#### 5c: Switch train.py from A2C to MaskablePPO

**Files:**
- Modify: `rl/train.py`

Replace the A2C import and model construction:

```python
# Old:
from stable_baselines3 import A2C
# ...
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

# New:
from sb3_contrib import MaskablePPO
# ...
model = MaskablePPO(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=args.lr,
    n_steps=args.n_steps,
    gamma=0.95,
    gae_lambda=0.95,
    ent_coef=0.05,
    verbose=1,
    tensorboard_log=args.logs_dir,
)
```

Also update the model save name from `mindustry_a2c` to `mindustry_ppo` in callbacks and final save.

#### 5d: Tests for action masking

**Files:**
- Test: `rl/tests/test_spaces.py` (test compute_action_mask)
- Test: `rl/tests/test_env.py` (test action_masks method)
- Test: `rl/tests/test_train.py` (update import references)

Add to `test_spaces.py`:
```python
from rl.env.spaces import compute_action_mask

def test_action_mask_shape():
    mask = compute_action_mask(MINIMAL_STATE)
    assert mask.shape == (16,)
    assert mask.dtype == np.bool_

def test_action_mask_wait_always_valid():
    mask = compute_action_mask(MINIMAL_STATE)
    assert mask[0] == True  # WAIT always valid

def test_action_mask_dead_player_blocks_actions():
    state = {**MINIMAL_STATE, "player": {"alive": False, "hp": 0.0}}
    mask = compute_action_mask(state)
    assert mask[0] == True  # WAIT still valid
    assert not any(mask[1:7])  # all other actions blocked

def test_action_mask_no_resources_blocks_build():
    state = {**MINIMAL_STATE, "resources": {"copper": 0, "lead": 0}}
    mask = compute_action_mask(state)
    assert mask[0] == True  # WAIT
    assert mask[1] == True  # MOVE
    assert mask[2] == False  # BUILD_TURRET (no copper)
    assert mask[3] == False  # BUILD_WALL (no copper)
    assert mask[4] == False  # BUILD_POWER (no lead)
    assert mask[5] == False  # BUILD_DRILL (no copper)
```

Add to `test_env.py`:
```python
def test_action_masks_returns_correct_shape():
    client = make_mock_client(states=[MOCK_STATE])
    env = MindustryEnv(client=client)
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (16,)
    assert mask.dtype == np.bool_
```

**Step: Run all tests**

```bash
python -m pytest rl/tests/ -v
```

**Step: Commit**

```bash
git add rl/requirements.txt rl/env/spaces.py rl/env/mindustry_env.py rl/train.py rl/tests/
git commit -m "feat(training): switch to MaskablePPO with invalid action masking"
```

---

### Task 6: Final verification

**Step 1: Run full test suite**
```bash
python -m pytest rl/tests/ -v
```

**Step 2: Check LSP diagnostics on all changed files**

**Step 3: Verify imports work**
```bash
source rl/venv/bin/activate && python -c "from sb3_contrib import MaskablePPO; print('OK')"
```
