# Reward Shaping & Inactivity Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate inaction inertia (excessive WAIT/MOVE) and incentivize mining (BUILD_DRILL) by improving reward shaping and adding intelligent inactivity detection.

**Architecture:** 
- Enhance `compute_reward()` to track drilling actions separately from passive copper gain
- Add inactivity detection: (A) repeated same-action cycles, (B) resource bleeding from bad constructions  
- Framework for curriculum learning (disabled by default, comments only)
- All changes in `rl/rewards/multi_objective.py`; no environment changes needed

**Tech Stack:** Python 3.8+, existing NumPy/Gymnasium stack

---

## Task 1: Analyze Current Reward Structure & State History

**Files:**
- Read: `rl/rewards/multi_objective.py`
- Read: `rl/env/spaces.py` (to understand state structure)
- Read: `rl/env/mindustry_env.py` (to see how state flows to reward)

**Step 1: Understand state fields in compute_reward()**

Review what `prev_state` and `curr_state` contain:
- `buildings`: list of {"block": "...", "x": ..., "y": ..., ...}
- `resources`: {"copper": N, "lead": N, ...}
- `core.hp`: float [0, 1]
- Any action tracking? (Check if there's an "lastAction" field)

**Step 2: Document state assumptions**

Create inline comment block in `multi_objective.py` describing:
- How to detect "new drill": filter buildings for `block == "drill"` and compare counts
- How to detect "resource bleeding": track if `curr_buildings > prev_buildings` but `resources_delta < -10`
- How to track action repetition: need state to carry `last_action` or compute from env step history

Run: `grep -n "def compute_reward" rl/rewards/multi_objective.py`
Expected: Shows function signature and line number

**Step 3: Commit analysis notes**

```bash
git add -A
git commit -m "docs: analyze reward state structure for inactivity detection"
```

---

## Task 2: Extend compute_reward() Signature to Track Action History

**Files:**
- Modify: `rl/rewards/multi_objective.py` (extend function signature)
- Modify: `rl/env/mindustry_env.py` (line ~140-160, pass action history to reward)

**Step 1: Add optional parameters to compute_reward()**

Current:
```python
def compute_reward(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    done: bool,
) -> float:
```

New (add at end of signature):
```python
def compute_reward(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    done: bool,
    action_taken: Optional[int] = None,
    action_history: Optional[list[int]] = None,
) -> float:
```

**Step 2: Document new parameters**

Add docstring:
```python
"""
Compute multi-objective reward.

Args:
    prev_state: Game state at t-1
    curr_state: Game state at t
    done: Episode terminal flag
    action_taken: Current action type (0-6, see spaces.py ACTION enum)
    action_history: List of last N actions (default: None, skip inactivity checks)

Returns:
    Scalar reward for this transition
"""
```

**Step 3: Update mindustry_env.py call site**

Find where `compute_reward()` is called (around line 140-160):
```bash
grep -n "compute_reward" rl/env/mindustry_env.py
```

Update call:
```python
reward = compute_reward(
    self._prev_state,
    curr_state,
    done,
    action_taken=action_idx,
    action_history=self._action_history,
)
```

Add to MindustryEnv.__init__():
```python
self._action_history: list[int] = []  # Track last N actions for inactivity detection
```

Add to MindustryEnv.step():
```python
self._action_history.append(action_idx)
if len(self._action_history) > 10:  # Keep last 10 actions
    self._action_history.pop(0)
```

**Step 4: Run tests to ensure no regression**

```bash
cd /home/pedro/repo/mindustry-ia-inteface-mod
python -m pytest rl/tests/test_reward.py -v
```

Expected: All tests pass (existing tests should work with optional params)

**Step 5: Commit**

```bash
git add rl/rewards/multi_objective.py rl/env/mindustry_env.py
git commit -m "feat: extend compute_reward signature for action history tracking"
```

---

## Task 3: Implement BUILD_DRILL Detection & Reward

**Files:**
- Modify: `rl/rewards/multi_objective.py` (add drill detection helper)

**Step 1: Write helper to detect new drills**

Add function after `compute_reward()` definition:
```python
def _detect_new_drills(prev_state: Dict[str, Any], curr_state: Dict[str, Any]) -> int:
    """
    Count newly constructed drills by comparing building lists.
    
    Returns:
        Number of new drill structures (drills in curr that weren't in prev)
    """
    prev_drills = {
        (b.get("x"), b.get("y"))
        for b in prev_state.get("buildings", [])
        if b.get("block") == "drill"
    }
    
    curr_drills = {
        (b.get("x"), b.get("y"))
        for b in curr_state.get("buildings", [])
        if b.get("block") == "drill"
    }
    
    return len(curr_drills - prev_drills)
```

**Step 2: Integrate into compute_reward()**

Inside `compute_reward()`, after `copper_delta` calculation (around line 44):
```python
# Detect new drills built this step
new_drills = _detect_new_drills(prev_state, curr_state)
drill_construction_bonus = min(1.0, new_drills * 0.15)  # +0.15 per new drill
```

**Step 3: Update reward calculation**

Replace current drill_bonus weight:
```python
# OLD:
# 0.08 * drill_bonus

# NEW: Combine mining reward (passive copper) + construction reward
reward = (
    0.30 * core_hp_delta
    + 0.20 * wave_survived_bonus
    + 0.10 * (resources_delta / 500.0)
    + 0.05 * drill_bonus              # Passive: copper collected
    + 0.15 * drill_construction_bonus  # Active: drill constructed
    + 0.07 * power_balance_bonus
    + 0.05 * build_efficiency_bonus
    + 0.20 * player_alive_bonus
    + 0.05 * manual_mining_reward
    + 1.00 * delivery_bonus
    - 0.002
)
```

**Step 4: Test the drill detection**

Create test file `rl/tests/test_drill_detection.py`:
```python
from rl.rewards.multi_objective import _detect_new_drills

def test_detect_single_new_drill():
    prev = {"buildings": [{"block": "wall", "x": 5, "y": 5}]}
    curr = {
        "buildings": [
            {"block": "wall", "x": 5, "y": 5},
            {"block": "drill", "x": 10, "y": 10},
        ]
    }
    assert _detect_new_drills(prev, curr) == 1

def test_detect_multiple_new_drills():
    prev = {"buildings": []}
    curr = {
        "buildings": [
            {"block": "drill", "x": 10, "y": 10},
            {"block": "drill", "x": 11, "y": 11},
            {"block": "wall", "x": 5, "y": 5},
        ]
    }
    assert _detect_new_drills(prev, curr) == 2

def test_drill_not_counted_if_already_present():
    prev = {"buildings": [{"block": "drill", "x": 10, "y": 10}]}
    curr = {"buildings": [{"block": "drill", "x": 10, "y": 10}]}
    assert _detect_new_drills(prev, curr) == 0
```

Run:
```bash
python -m pytest rl/tests/test_drill_detection.py -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add rl/rewards/multi_objective.py rl/tests/test_drill_detection.py
git commit -m "feat: add drill construction detection and +0.15 bonus"
```

---

## Task 4: Implement Inactivity Detection - Scenario A (Action Repetition)

**Files:**
- Modify: `rl/rewards/multi_objective.py` (add action repetition helper)

**Step 1: Write action repetition detector**

Add after `_detect_new_drills()`:
```python
def _detect_action_repetition_penalty(
    action_history: Optional[list[int]],
    resources_delta: float,
    min_history_len: int = 3,
    no_progress_threshold: float = 0.0,
) -> float:
    """
    Penalize excessive repetition of low-value actions (WAIT, MOVE) without resource gain.
    
    Args:
        action_history: Last N actions (0=WAIT, 1=MOVE, 2-6=BUILD/REPAIR)
        resources_delta: Total resources gained this step
        min_history_len: Minimum history to trigger penalty (default 3)
        no_progress_threshold: If resources_delta <= this, consider it "no progress"
    
    Returns:
        Penalty (negative value) or 0 if no violation
    """
    if action_history is None or len(action_history) < min_history_len:
        return 0.0
    
    # Check if last N actions are all WAIT (0) or MOVE (1)
    recent_actions = action_history[-min_history_len:]
    all_passive = all(a in (0, 1) for a in recent_actions)
    
    if all_passive and resources_delta <= no_progress_threshold:
        # Multiple WAIT/MOVE without collecting resources
        return -0.05  # Penalty per repetitive cycle
    
    return 0.0
```

**Step 2: Test action repetition detector**

Add to `rl/tests/test_reward.py`:
```python
def test_no_penalty_if_actions_diverse():
    from rl.rewards.multi_objective import _detect_action_repetition_penalty
    
    history = [0, 1, 2, 0, 1]  # Mixed WAIT, MOVE, BUILD
    penalty = _detect_action_repetition_penalty(history, resources_delta=0.0)
    assert penalty == 0.0

def test_penalty_if_repeated_wait_no_progress():
    from rl.rewards.multi_objective import _detect_action_repetition_penalty
    
    history = [0, 0, 0, 0, 0]  # All WAIT
    penalty = _detect_action_repetition_penalty(history, resources_delta=0.0)
    assert penalty == -0.05

def test_penalty_if_repeated_move_no_progress():
    from rl.rewards.multi_objective import _detect_action_repetition_penalty
    
    history = [1, 1, 1]  # All MOVE
    penalty = _detect_action_repetition_penalty(history, resources_delta=0.0)
    assert penalty == -0.05

def test_no_penalty_if_resources_gained_despite_repetition():
    from rl.rewards.multi_objective import _detect_action_repetition_penalty
    
    history = [1, 1, 1]  # MOVE MOVE MOVE
    penalty = _detect_action_repetition_penalty(history, resources_delta=50.0)  # But gained 50 copper
    assert penalty == 0.0
```

Run:
```bash
python -m pytest rl/tests/test_reward.py::test_no_penalty_if_actions_diverse -v
python -m pytest rl/tests/test_reward.py::test_penalty_if_repeated_wait_no_progress -v
python -m pytest rl/tests/test_reward.py::test_penalty_if_repeated_move_no_progress -v
python -m pytest rl/tests/test_reward.py::test_no_penalty_if_resources_gained_despite_repetition -v
```

Expected: All 4 tests PASS

**Step 3: Integrate into compute_reward()**

Inside `compute_reward()`, after all bonus calculations:
```python
# Inactivity Detection: Scenario A
inactivity_penalty_a = _detect_action_repetition_penalty(
    action_history=action_history,
    resources_delta=resources_delta,
)
```

**Step 4: Add to reward accumulation**

Update reward calculation:
```python
reward = (
    0.30 * core_hp_delta
    + 0.20 * wave_survived_bonus
    + 0.10 * (resources_delta / 500.0)
    + 0.05 * drill_bonus
    + 0.15 * drill_construction_bonus
    + 0.07 * power_balance_bonus
    + 0.05 * build_efficiency_bonus
    + 0.20 * player_alive_bonus
    + 0.05 * manual_mining_reward
    + 1.00 * delivery_bonus
    - 0.002
    + inactivity_penalty_a  # NEW
)
```

**Step 5: Run all reward tests**

```bash
python -m pytest rl/tests/test_reward.py -v
```

Expected: All tests PASS (including old ones)

**Step 6: Commit**

```bash
git add rl/rewards/multi_objective.py rl/tests/test_reward.py
git commit -m "feat: add inactivity penalty for repetitive WAIT/MOVE without progress"
```

---

## Task 5: Implement Inactivity Detection - Scenario B (Resource Bleeding)

**Files:**
- Modify: `rl/rewards/multi_objective.py` (add resource bleeding detector)

**Step 1: Write resource bleeding detector**

Add after `_detect_action_repetition_penalty()`:
```python
def _detect_resource_bleeding_penalty(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    bleeding_threshold: float = -10.0,
) -> float:
    """
    Penalize building structures that cost more resources than collected.
    
    If buildings increased BUT total resources decreased significantly,
    the agent is making expensive, unproductive builds.
    
    Args:
        prev_state: Previous game state
        curr_state: Current game state
        bleeding_threshold: If resources_delta < this, apply penalty (default -10)
    
    Returns:
        Penalty (negative) or 0.0
    """
    prev_buildings = len(prev_state.get("buildings", []))
    curr_buildings = len(curr_state.get("buildings", []))
    new_buildings = curr_buildings - prev_buildings
    
    def _total_resources(state: Dict[str, Any]) -> float:
        return sum(float(v) for v in state.get("resources", {}).values())
    
    resources_delta = _total_resources(curr_state) - _total_resources(prev_state)
    
    # Penalize if built something but hemorrhaging resources
    if new_buildings > 0 and resources_delta < bleeding_threshold:
        return -0.10
    
    return 0.0
```

**Step 2: Test resource bleeding detector**

Add to `rl/tests/test_reward.py`:
```python
def test_no_penalty_if_built_and_gained_resources():
    from rl.rewards.multi_objective import _detect_resource_bleeding_penalty
    
    prev = {"buildings": [{"block": "wall", "x": 0, "y": 0}], "resources": {"copper": 100}}
    curr = {"buildings": [{"block": "wall", "x": 0, "y": 0}, {"block": "drill", "x": 5, "y": 5}], "resources": {"copper": 150}}
    
    penalty = _detect_resource_bleeding_penalty(prev, curr)
    assert penalty == 0.0

def test_penalty_if_built_but_resources_dropped():
    from rl.rewards.multi_objective import _detect_resource_bleeding_penalty
    
    prev = {"buildings": [{"block": "wall", "x": 0, "y": 0}], "resources": {"copper": 100}}
    curr = {"buildings": [{"block": "wall", "x": 0, "y": 0}, {"block": "turret", "x": 5, "y": 5}], "resources": {"copper": 80}}
    
    penalty = _detect_resource_bleeding_penalty(prev, curr, bleeding_threshold=-10.0)
    assert penalty == -0.10

def test_no_penalty_if_resources_dropped_but_no_new_buildings():
    from rl.rewards.multi_objective import _detect_resource_bleeding_penalty
    
    prev = {"buildings": [{"block": "wall", "x": 0, "y": 0}], "resources": {"copper": 100}}
    curr = {"buildings": [{"block": "wall", "x": 0, "y": 0}], "resources": {"copper": 80}}
    
    penalty = _detect_resource_bleeding_penalty(prev, curr)
    assert penalty == 0.0
```

Run:
```bash
python -m pytest rl/tests/test_reward.py::test_no_penalty_if_built_and_gained_resources -v
python -m pytest rl/tests/test_reward.py::test_penalty_if_built_but_resources_dropped -v
python -m pytest rl/tests/test_reward.py::test_no_penalty_if_resources_dropped_but_no_new_buildings -v
```

Expected: All 3 tests PASS

**Step 3: Integrate into compute_reward()**

Inside `compute_reward()`, after scenario A penalty:
```python
# Inactivity Detection: Scenario B - Resource Bleeding
inactivity_penalty_b = _detect_resource_bleeding_penalty(prev_state, curr_state)
```

**Step 4: Add to reward accumulation**

```python
reward = (
    0.30 * core_hp_delta
    + 0.20 * wave_survived_bonus
    + 0.10 * (resources_delta / 500.0)
    + 0.05 * drill_bonus
    + 0.15 * drill_construction_bonus
    + 0.07 * power_balance_bonus
    + 0.05 * build_efficiency_bonus
    + 0.20 * player_alive_bonus
    + 0.05 * manual_mining_reward
    + 1.00 * delivery_bonus
    - 0.002
    + inactivity_penalty_a
    + inactivity_penalty_b  # NEW
)
```

**Step 5: Run all reward tests**

```bash
python -m pytest rl/tests/test_reward.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add rl/rewards/multi_objective.py rl/tests/test_reward.py
git commit -m "feat: add inactivity penalty for resource bleeding from expensive builds"
```

---

## Task 6: Add Curriculum Learning Framework (Disabled)

**Files:**
- Modify: `rl/rewards/multi_objective.py` (add comments + framework)

**Step 1: Add curriculum constants at top of file**

After imports, add:
```python
# ============================================================================
# Curriculum Learning Framework (DISABLED BY DEFAULT)
# ============================================================================
# Set CURRICULUM_ENABLED = True to restrict agent to specific actions
# during early training phases. Currently unused; prepared for future.

CURRICULUM_ENABLED = False

# Action indices (from spaces.py)
ACTION_WAIT = 0
ACTION_MOVE = 1
ACTION_BUILD_TURRET = 2
ACTION_BUILD_WALL = 3
ACTION_BUILD_POWER = 4
ACTION_BUILD_DRILL = 5
ACTION_REPAIR = 6

# Phases: (phase_name, timestep_range, allowed_actions)
CURRICULUM_PHASES = [
    # Phase 0: Learn mining only (MOVE + BUILD_DRILL)
    ("mining_only", (0, 50000), [ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL]),
    # Phase 1: Add power generation
    ("power_gen", (50000, 150000), [ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL, ACTION_BUILD_POWER]),
    # Phase 2: Full actions
    ("full", (150000, float('inf')), list(range(7))),
]
```

**Step 2: Add curriculum function (stub)**

Add after inactivity detector functions:
```python
def apply_curriculum_action_mask(
    timestep: int,
) -> list[bool]:
    """
    Return action mask based on curriculum phase.
    
    Args:
        timestep: Current training timestep
    
    Returns:
        List of 7 bools (one per action type): True=allowed, False=masked
    
    Note:
        Currently unused. Set CURRICULUM_ENABLED=True in train.py to activate.
    """
    if not CURRICULUM_ENABLED:
        return [True] * 7
    
    allowed_actions = [ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL, ACTION_BUILD_POWER, ACTION_BUILD_TURRET, ACTION_BUILD_WALL, ACTION_REPAIR]
    
    for phase_name, (start, end), actions in CURRICULUM_PHASES:
        if start <= timestep < end:
            allowed_actions = actions
            break
    
    mask = [False] * 7
    for action_idx in allowed_actions:
        mask[action_idx] = True
    
    return mask
```

**Step 3: Add comment block documenting future usage**

```python
"""
TO ENABLE CURRICULUM LEARNING:

1. In rl/train.py, set CURRICULUM_ENABLED = True (in imports)
2. Modify MindustryEnv.step() to apply mask:
   
   action_mask = apply_curriculum_action_mask(self.env.num_timesteps)
   obs, reward, done, trunc, info = env.step(action, action_mask)

3. Test convergence on early phases before enabling full action space
"""
```

**Step 4: Test that curriculum function works**

Add to `rl/tests/test_reward.py`:
```python
def test_curriculum_disabled_allows_all_actions():
    from rl.rewards.multi_objective import apply_curriculum_action_mask, CURRICULUM_ENABLED
    
    mask = apply_curriculum_action_mask(timestep=0)
    assert mask == [True] * 7  # All allowed when disabled

def test_curriculum_framework_exists():
    from rl.rewards.multi_objective import CURRICULUM_PHASES, CURRICULUM_ENABLED
    
    assert CURRICULUM_ENABLED is False
    assert len(CURRICULUM_PHASES) > 0
    assert CURRICULUM_PHASES[0][0] == "mining_only"
```

Run:
```bash
python -m pytest rl/tests/test_reward.py::test_curriculum_disabled_allows_all_actions -v
python -m pytest rl/tests/test_reward.py::test_curriculum_framework_exists -v
```

Expected: Both PASS

**Step 5: Commit**

```bash
git add rl/rewards/multi_objective.py rl/tests/test_reward.py
git commit -m "feat: add curriculum learning framework (disabled by default)"
```

---

## Task 7: Final Integration Test & Documentation

**Files:**
- Read: `rl/rewards/multi_objective.py` (entire file)
- Modify: `rl/rewards/multi_objective.py` (add module docstring update)

**Step 1: Update module docstring**

Replace the top docstring with:
```python
"""
Multi-objective reward function for the Mindustry RL player agent.

REWARD COMPOSITION:
  0.30 * core_hp_delta              (defend core from damage)
  0.20 * wave_survived_bonus        (survive wave transitions)
  0.10 * resources_delta / 500      (collect resources passively)
  0.05 * drill_bonus                (passive copper via mining)
  0.15 * drill_construction_bonus   (ACTIVE: build new drills)
  0.07 * power_balance_bonus        (maintain power generation)
  0.05 * build_efficiency_bonus     (construct efficiently)
  0.20 * player_alive_bonus         (keep player alive)
  0.05 * manual_mining_reward       (manual inventory collection)
  1.00 * delivery_bonus             (deliver to core)
  0.002 * time_penalty              (discourage stalling)
  
INACTIVITY PENALTIES (NEW):
  - Scenario A: -0.05 if 3+ consecutive WAIT/MOVE with no resource progress
  - Scenario B: -0.10 if new buildings constructed but resources drop >10

TERMINAL PENALTIES:
  core destroyed        → -0.4
  player dead, core ok  → -0.5
  action failed         → -0.15

CURRICULUM LEARNING (disabled):
  Framework ready for phased action masking. Set CURRICULUM_ENABLED=True to activate.
"""
```

**Step 2: Run full reward test suite**

```bash
python -m pytest rl/tests/test_reward.py -v --tb=short
```

Expected: All tests PASS (old + new)

**Step 3: Manual integration check**

Create a simple test that runs compute_reward() with all new parameters:

```python
def test_full_compute_reward_integration():
    from rl.rewards.multi_objective import compute_reward
    
    # Minimal state for testing
    prev_state = {
        "core": {"hp": 1.0},
        "wave": 1,
        "resources": {"copper": 100, "lead": 50},
        "power": {"produced": 100, "consumed": 50},
        "buildings": [{"block": "wall", "x": 0, "y": 0}],
        "player": {"alive": True},
        "inventory": {},
        "actionFailed": False,
    }
    
    curr_state = {
        "core": {"hp": 1.0},
        "wave": 1,
        "resources": {"copper": 150, "lead": 50},
        "power": {"produced": 100, "consumed": 50},
        "buildings": [
            {"block": "wall", "x": 0, "y": 0},
            {"block": "drill", "x": 5, "y": 5},
        ],
        "player": {"alive": True},
        "inventory": {},
        "actionFailed": False,
    }
    
    # Test with action history (MOVE + BUILD_DRILL)
    reward = compute_reward(
        prev_state=prev_state,
        curr_state=curr_state,
        done=False,
        action_taken=5,  # BUILD_DRILL
        action_history=[1, 1, 5],  # MOVE, MOVE, BUILD_DRILL
    )
    
    # Should be positive (drill bonus + copper collection)
    assert reward > 0, f"Expected positive reward, got {reward}"
```

Add to `rl/tests/test_reward.py` and run:
```bash
python -m pytest rl/tests/test_reward.py::test_full_compute_reward_integration -v
```

Expected: PASS

**Step 4: Commit final documentation**

```bash
git add rl/rewards/multi_objective.py
git commit -m "docs: update reward function documentation with new features"
```

**Step 5: Summary & Next Steps**

Run final verification:
```bash
python -m pytest rl/tests/ -k reward -v
```

Expected: All reward-related tests PASS

Print summary:
```bash
echo "=== Reward Shaping Enhancements Complete ==="
echo "1. BUILD_DRILL instantaneous bonus: +0.15"
echo "2. Action repetition penalty: -0.05 (Scenario A)"
echo "3. Resource bleeding penalty: -0.10 (Scenario B)"
echo "4. Curriculum learning framework: Ready (disabled)"
echo ""
echo "Next: Run rl.train to validate convergence improvements"
```

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `rl/rewards/multi_objective.py` | +150 lines: drill detection, inactivity penalties, curriculum framework |
| `rl/env/mindustry_env.py` | +5 lines: action_history tracking in __init__ and step() |
| `rl/tests/test_reward.py` | +50 lines: new test cases for all penalty types |

## Testing Checklist

- [ ] All new unit tests PASS
- [ ] Backward compatibility with existing reward tests
- [ ] `python -m rl.train` launches without errors
- [ ] Dashboard shows reward changes after 100+ episodes
- [ ] Action distribution shows BUILD_DRILL increasing
- [ ] Episode length stabilizes (less dying early)
