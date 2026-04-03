# Dashboard Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add real-time tracking of drill construction rate, inactivity penalty frequency, and action distribution to the RL training dashboard.

**Architecture:** Metrics originate in `MindustryEnv.step()` (tracking drill deltas, penalty triggers, actions), aggregate in `LiveMetricsCallback._compute_metrics()` (episode summaries), persist to `live_metrics.json`, and render in `dashboard.py` as 6 new time-series and distribution plots.

**Tech Stack:** Python, NumPy, Gymnasium, Stable Baselines3, Rich (for dashboard), JSON (metrics storage)

---

## Task 1: Extend MindustryEnv.step() to Track Metrics

**Files:**
- Modify: `rl/env/mindustry_env.py` (lines ~160-180)
- Test: `rl/tests/test_env.py` (new test)

**Step 1: Understand current step() structure**

Run: `grep -n "def step" rl/env/mindustry_env.py`

Expected output shows `step()` definition around line 140-180. Review the full method to understand:
- How `compute_reward()` is called
- How `info` dict is populated
- How previous state is tracked
- How done flag is computed

**Step 2: Write failing test for drill tracking**

Create test in `rl/tests/test_env.py`:

```python
def test_step_tracks_drills_built():
    """Verify step() adds drills_built_this_step to info dict"""
    env = MindustryEnv()
    obs, info = env.reset()
    
    # Mock a state change with 2 new drills
    env.prev_state = {
        "buildings": [{"block": "duo", "x": 10, "y": 20}]
    }
    env.current_state = {
        "buildings": [
            {"block": "duo", "x": 10, "y": 20},
            {"block": "mechanical-drill", "x": 15, "y": 25},
            {"block": "mechanical-drill", "x": 16, "y": 26}
        ]
    }
    
    obs, reward, done, truncated, info = env.step(action=0)
    
    assert "drills_built_this_step" in info
    assert info["drills_built_this_step"] == 2
```

Run: `pytest rl/tests/test_env.py::test_step_tracks_drills_built -v`

Expected: FAIL with KeyError or AssertionError (metric not tracked yet)

**Step 3: Add drill tracking to step()**

In `rl/env/mindustry_env.py`, after the `compute_reward()` call (~line 165), add:

```python
# Track drills built this step
new_drills = self._detect_new_drills(self.prev_state, curr_state)
info["drills_built_this_step"] = new_drills

# Track penalty triggers
info["penalty_a_triggered"] = 1 if inactivity_penalty_a != 0 else 0
info["penalty_b_triggered"] = 1 if inactivity_penalty_b != 0 else 0

# Track action taken
info["action_taken_index"] = action

# Track step count
info["step_count"] = self.step_count
```

Note: Verify that `_detect_new_drills()` already exists (it was added in Task 3 of the previous reward shaping implementation). If not, implement it:

```python
def _detect_new_drills(self, prev_state, curr_state):
    """Count new mechanical-drill blocks between states"""
    prev_drills = {
        (b["x"], b["y"]) for b in prev_state.get("buildings", [])
        if b.get("block") == "mechanical-drill"
    }
    curr_drills = {
        (b["x"], b["y"]) for b in curr_state.get("buildings", [])
        if b.get("block") == "mechanical-drill"
    }
    return len(curr_drills - prev_drills)
```

**Step 4: Run test to verify pass**

Run: `pytest rl/tests/test_env.py::test_step_tracks_drills_built -v`

Expected: PASS

**Step 5: Add tests for penalty and action tracking**

Add to `rl/tests/test_env.py`:

```python
def test_step_tracks_penalties():
    """Verify step() tracks penalty triggers"""
    env = MindustryEnv()
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action=0)
    
    assert "penalty_a_triggered" in info
    assert "penalty_b_triggered" in info
    assert info["penalty_a_triggered"] in [0, 1]
    assert info["penalty_b_triggered"] in [0, 1]

def test_step_tracks_action():
    """Verify step() tracks action taken"""
    env = MindustryEnv()
    obs, info = env.reset()
    
    for action in range(7):
        obs, reward, done, truncated, info = env.step(action=action)
        assert info["action_taken_index"] == action
```

Run: `pytest rl/tests/test_env.py::test_step_tracks_penalties rl/tests/test_env.py::test_step_tracks_action -v`

Expected: PASS

**Step 6: Commit**

```bash
git add rl/env/mindustry_env.py rl/tests/test_env.py
git commit -m "feat: add drill, penalty, and action tracking to env.step() info dict"
```

---

## Task 2: Extend LiveMetricsCallback to Aggregate Metrics

**Files:**
- Modify: `rl/callbacks/training_callbacks.py` (lines ~80-150)
- Test: `rl/tests/test_training_callbacks.py` (new test)

**Step 1: Understand LiveMetricsCallback structure**

Run: `grep -n "class LiveMetricsCallback" rl/callbacks/training_callbacks.py`

Review the class to understand:
- How `_compute_metrics()` is called
- How episode infos are aggregated
- How rollout buffer is accessed
- How metrics are written to JSON

**Step 2: Write failing test for drill aggregation**

Create test in `rl/tests/test_training_callbacks.py`:

```python
def test_live_metrics_callback_aggregates_drills():
    """Verify callback computes drills_built_total and frequency"""
    callback = LiveMetricsCallback()
    
    # Mock episode infos (3 steps: 2 drills, 0 drills, 1 drill)
    episode_infos = [
        {"drills_built_this_step": 2, "penalty_a_triggered": 0, "penalty_b_triggered": 0, "action_taken_index": 5},
        {"drills_built_this_step": 0, "penalty_a_triggered": 1, "penalty_b_triggered": 0, "action_taken_index": 0},
        {"drills_built_this_step": 1, "penalty_a_triggered": 0, "penalty_b_triggered": 0, "action_taken_index": 5},
    ]
    
    metrics = callback._compute_metrics(episode_infos)
    
    assert metrics["drills_built_total"] == 3
    assert metrics["drill_build_frequency_pct"] == pytest.approx(100.0)  # 3/3 steps
    assert metrics["penalty_a_count"] == 1
    assert metrics["penalty_frequency_pct"] == pytest.approx(33.33, abs=0.1)
```

Run: `pytest rl/tests/test_training_callbacks.py::test_live_metrics_callback_aggregates_drills -v`

Expected: FAIL (method or fields don't exist yet)

**Step 3: Extend _compute_metrics() in callback**

In `rl/callbacks/training_callbacks.py`, find `_compute_metrics()` and add metric aggregation:

```python
def _compute_metrics(self, episode_infos):
    """Compute aggregated metrics from step-level infos"""
    
    # Initialize accumulators
    total_drills = 0
    total_penalty_a = 0
    total_penalty_b = 0
    action_counts = {i: 0 for i in range(7)}
    
    # Aggregate across all steps
    for info in episode_infos:
        total_drills += info.get("drills_built_this_step", 0)
        total_penalty_a += info.get("penalty_a_triggered", 0)
        total_penalty_b += info.get("penalty_b_triggered", 0)
        action = info.get("action_taken_index", 0)
        action_counts[action] += 1
    
    num_steps = len(episode_infos)
    
    # Compute frequencies (handle division by zero)
    drill_frequency_pct = (total_drills / num_steps * 100) if num_steps > 0 else 0.0
    penalty_frequency_pct = ((total_penalty_a + total_penalty_b) / num_steps * 100) if num_steps > 0 else 0.0
    
    # Normalize action distribution
    total_actions = sum(action_counts.values())
    action_dist = {
        i: (action_counts[i] / total_actions if total_actions > 0 else 0.0)
        for i in range(7)
    }
    
    # Map action indices to names
    action_names = ["WAIT", "MOVE", "BUILD_TURRET", "BUILD_WALL", "BUILD_POWER", "BUILD_DRILL", "REPAIR"]
    action_dist_named = {action_names[i]: action_dist[i] for i in range(7)}
    
    return {
        "drills_built_total": total_drills,
        "drill_build_frequency_pct": drill_frequency_pct,
        "penalty_a_count": total_penalty_a,
        "penalty_b_count": total_penalty_b,
        "penalty_frequency_pct": penalty_frequency_pct,
        "action_dist": action_dist_named
    }
```

**Step 4: Integrate into existing _compute_metrics() flow**

Find where `_compute_metrics()` is called and ensure it merges the new metrics with existing ones:

```python
# In the main metrics computation
metrics = {
    # ... existing metrics (rewards, lengths, etc.)
}

# Add new metrics
new_metrics = self._compute_metrics(episode_infos)
metrics.update(new_metrics)

return metrics
```

**Step 5: Run test to verify pass**

Run: `pytest rl/tests/test_training_callbacks.py::test_live_metrics_callback_aggregates_drills -v`

Expected: PASS

**Step 6: Add tests for penalties and action distribution**

Add to `rl/tests/test_training_callbacks.py`:

```python
def test_live_metrics_callback_action_distribution():
    """Verify callback normalizes action distribution"""
    callback = LiveMetricsCallback()
    
    episode_infos = [
        {"drills_built_this_step": 0, "penalty_a_triggered": 0, "penalty_b_triggered": 0, "action_taken_index": 0},  # WAIT
        {"drills_built_this_step": 0, "penalty_a_triggered": 0, "penalty_b_triggered": 0, "action_taken_index": 1},  # MOVE
        {"drills_built_this_step": 0, "penalty_a_triggered": 0, "penalty_b_triggered": 0, "action_taken_index": 5},  # BUILD_DRILL
    ]
    
    metrics = callback._compute_metrics(episode_infos)
    
    assert metrics["action_dist"]["WAIT"] == pytest.approx(1/3)
    assert metrics["action_dist"]["MOVE"] == pytest.approx(1/3)
    assert metrics["action_dist"]["BUILD_DRILL"] == pytest.approx(1/3)
    assert sum(metrics["action_dist"].values()) == pytest.approx(1.0)
```

Run: `pytest rl/tests/test_training_callbacks.py -v`

Expected: All new tests PASS

**Step 7: Commit**

```bash
git add rl/callbacks/training_callbacks.py rl/tests/test_training_callbacks.py
git commit -m "feat: extend LiveMetricsCallback to aggregate drill, penalty, and action metrics"
```

---

## Task 3: Add 6 Drawing Functions to Dashboard

**Files:**
- Modify: `rl/dashboard.py` (lines ~200-300)
- Test: `rl/tests/test_dashboard.py` (new tests)

**Step 1: Understand dashboard structure**

Run: `grep -n "def _draw_" rl/dashboard.py | head -20`

Review the output to understand:
- How drawing functions are structured
- How they access metrics from JSON
- How they use Rich library for rendering
- How they're integrated into `make_updater()`

**Step 2: Write failing test for drill rate plot**

Create test in `rl/tests/test_dashboard.py`:

```python
def test_dashboard_can_render_drill_rate_total():
    """Verify dashboard can render drill rate total plot"""
    from rl.dashboard import Dashboard
    
    # Create dashboard with mock metrics
    dashboard = Dashboard()
    
    # Mock live_metrics.json with drill data
    mock_metrics = [
        {"episode": 1, "drills_built_total": 2, "drill_build_frequency_pct": 1.5},
        {"episode": 2, "drills_built_total": 3, "drill_build_frequency_pct": 2.2},
        {"episode": 3, "drills_built_total": 5, "drill_build_frequency_pct": 3.8},
    ]
    
    # Should not raise exception
    plot = dashboard._draw_drill_rate_total(mock_metrics)
    assert plot is not None
```

Run: `pytest rl/tests/test_dashboard.py::test_dashboard_can_render_drill_rate_total -v`

Expected: FAIL (function doesn't exist)

**Step 3: Implement 6 drawing functions**

In `rl/dashboard.py`, add after existing `_draw_*` functions:

```python
def _draw_drill_rate_total(self, metrics):
    """Line plot: Total drills built per episode"""
    if not metrics:
        return Panel("No drill data", title="Drills Built (Total)")
    
    episodes = [m.get("episode", i) for i, m in enumerate(metrics)]
    drills = [m.get("drills_built_total", 0) for m in metrics]
    
    plot = Plot(
        title="Drills Built (Total)",
        x_label="Episode",
        y_label="Count",
        width=40,
        height=10,
    )
    
    for x, y in zip(episodes, drills):
        plot.plot(x, y, "o")
    
    return plot

def _draw_drill_rate_frequency(self, metrics):
    """Line plot: Drill build frequency (% of steps)"""
    if not metrics:
        return Panel("No drill frequency data", title="Drill Build Frequency")
    
    episodes = [m.get("episode", i) for i, m in enumerate(metrics)]
    frequencies = [m.get("drill_build_frequency_pct", 0) for m in metrics]
    
    plot = Plot(
        title="Drill Build Frequency (%)",
        x_label="Episode",
        y_label="Frequency %",
        width=40,
        height=10,
    )
    
    for x, y in zip(episodes, frequencies):
        plot.plot(x, y, "o")
    
    return plot

def _draw_penalty_counts(self, metrics):
    """Grouped bar chart: penalty_a vs penalty_b counts"""
    if not metrics:
        return Panel("No penalty data", title="Penalty Counts")
    
    penalty_a_counts = [m.get("penalty_a_count", 0) for m in metrics[-20:]]  # Last 20 episodes
    penalty_b_counts = [m.get("penalty_b_count", 0) for m in metrics[-20:]]
    
    table = Table(title="Penalty Counts (Last 20 Episodes)")
    table.add_column("Penalty A", style="red")
    table.add_column("Penalty B", style="yellow")
    
    for pa, pb in zip(penalty_a_counts, penalty_b_counts):
        table.add_row(str(int(pa)), str(int(pb)))
    
    return table

def _draw_penalty_frequency(self, metrics):
    """Line plot: % of steps with any penalty triggered"""
    if not metrics:
        return Panel("No penalty frequency data", title="Penalty Frequency")
    
    episodes = [m.get("episode", i) for i, m in enumerate(metrics)]
    frequencies = [m.get("penalty_frequency_pct", 0) for m in metrics]
    
    plot = Plot(
        title="Penalty Frequency (%)",
        x_label="Episode",
        y_label="Frequency %",
        width=40,
        height=10,
    )
    
    for x, y in zip(episodes, frequencies):
        plot.plot(x, y, "x")
    
    return plot

def _draw_action_dist_per_episode(self, metrics):
    """Stacked bar: Action distribution for latest episode"""
    if not metrics:
        return Panel("No action data", title="Action Distribution (Latest)")
    
    latest = metrics[-1]
    action_dist = latest.get("action_dist", {})
    
    table = Table(title="Action Distribution (Latest Episode)")
    table.add_column("Action", style="cyan")
    table.add_column("Frequency %", style="green")
    
    for action_name, freq in action_dist.items():
        pct = freq * 100
        table.add_row(action_name, f"{pct:.1f}%")
    
    return table

def _draw_action_dist_rolling(self, metrics):
    """Stacked area chart: rolling 100-episode action trend"""
    if not metrics:
        return Panel("No action trend data", title="Action Distribution (Rolling)")
    
    # Aggregate action distribution over last 100 episodes
    rolling_window = metrics[-100:]
    action_totals = {i: 0 for i in range(7)}
    action_names = ["WAIT", "MOVE", "BUILD_TURRET", "BUILD_WALL", "BUILD_POWER", "BUILD_DRILL", "REPAIR"]
    
    for metric in rolling_window:
        action_dist = metric.get("action_dist", {})
        for i, name in enumerate(action_names):
            action_totals[name] = action_totals.get(name, 0) + action_dist.get(name, 0)
    
    # Normalize
    total = sum(action_totals.values())
    action_avg = {name: (action_totals[name] / total * 100) if total > 0 else 0 for name in action_names}
    
    table = Table(title="Action Distribution (Rolling 100-Episode Average)")
    table.add_column("Action", style="cyan")
    table.add_column("Avg Frequency %", style="green")
    
    for action_name in action_names:
        pct = action_avg[action_name]
        table.add_row(action_name, f"{pct:.1f}%")
    
    return table
```

**Step 4: Integrate into make_updater()**

Find the `make_updater()` function and add calls to the 6 new drawing functions:

```python
def make_updater(self):
    """Create updater function for dashboard refresh"""
    
    def update(self):
        # ... existing code ...
        
        # New metric panels
        drill_rate_total = self._draw_drill_rate_total(metrics)
        drill_rate_frequency = self._draw_drill_rate_frequency(metrics)
        penalty_counts = self._draw_penalty_counts(metrics)
        penalty_frequency = self._draw_penalty_frequency(metrics)
        action_dist_latest = self._draw_action_dist_per_episode(metrics)
        action_dist_rolling = self._draw_action_dist_rolling(metrics)
        
        # Arrange in 2 rows of 3
        layout = Layout()
        layout.split_column(
            Layout(name="row1"),
            Layout(name="row2"),
        )
        layout["row1"].split_row(
            Layout(drill_rate_total, name="drill_total"),
            Layout(drill_rate_frequency, name="drill_freq"),
            Layout(penalty_counts, name="penalties"),
        )
        layout["row2"].split_row(
            Layout(penalty_frequency, name="penalty_freq"),
            Layout(action_dist_latest, name="action_latest"),
            Layout(action_dist_rolling, name="action_rolling"),
        )
        
        return layout
    
    return update
```

**Step 5: Run test to verify pass**

Run: `pytest rl/tests/test_dashboard.py::test_dashboard_can_render_drill_rate_total -v`

Expected: PASS

**Step 6: Add regression tests**

Add to `rl/tests/test_dashboard.py`:

```python
def test_dashboard_backwards_compatible_without_new_metrics():
    """Verify dashboard handles missing new metrics gracefully"""
    from rl.dashboard import Dashboard
    
    dashboard = Dashboard()
    
    # Old metrics format (without drill/penalty/action data)
    old_metrics = [
        {"episode": 1, "reward": 10.5, "length": 100},
        {"episode": 2, "reward": 12.3, "length": 120},
    ]
    
    # Should not crash
    drill_plot = dashboard._draw_drill_rate_total(old_metrics)
    assert drill_plot is not None
    
    penalty_plot = dashboard._draw_penalty_frequency(old_metrics)
    assert penalty_plot is not None
```

Run: `pytest rl/tests/test_dashboard.py -v`

Expected: All tests PASS

**Step 7: Commit**

```bash
git add rl/dashboard.py rl/tests/test_dashboard.py
git commit -m "feat: add 6 new dashboard plots for drill rate, penalties, and action distribution"
```

---

## Task 4: Integration Test & Verification

**Files:**
- Test: `rl/tests/test_integration_metrics.py` (new)
- Run: `python -m rl.train --timesteps 1000 --n-envs 1` (short training run)

**Step 1: Write end-to-end integration test**

Create `rl/tests/test_integration_metrics.py`:

```python
import json
import tempfile
import os
from rl.env.mindustry_env import MindustryEnv
from rl.callbacks.training_callbacks import LiveMetricsCallback

def test_metrics_pipeline_end_to_end():
    """Verify metrics flow from env → callback → JSON → dashboard"""
    
    # Create temp directory for metrics
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_file = os.path.join(tmpdir, "live_metrics.json")
        
        # Simulate an episode
        env = MindustryEnv()
        callback = LiveMetricsCallback()
        
        obs, info = env.reset()
        
        episode_infos = []
        for step in range(10):
            action = 5 if step % 3 == 0 else 0  # Build drills every 3 steps
            obs, reward, done, truncated, info = env.step(action)
            episode_infos.append(info)
            
            if done or truncated:
                break
        
        # Compute metrics
        metrics = callback._compute_metrics(episode_infos)
        
        # Verify all fields present
        assert "drills_built_total" in metrics
        assert "drill_build_frequency_pct" in metrics
        assert "penalty_a_count" in metrics
        assert "penalty_b_count" in metrics
        assert "penalty_frequency_pct" in metrics
        assert "action_dist" in metrics
        
        # Verify action_dist is normalized
        action_dist_sum = sum(metrics["action_dist"].values())
        assert action_dist_sum == pytest.approx(1.0, abs=0.01)
        
        # Write to JSON and read back
        with open(metrics_file, "w") as f:
            json.dump([metrics], f)
        
        with open(metrics_file, "r") as f:
            loaded_metrics = json.load(f)
        
        # Verify round-trip integrity
        assert loaded_metrics[0]["drills_built_total"] == metrics["drills_built_total"]
        assert loaded_metrics[0]["action_dist"] == metrics["action_dist"]
```

Run: `pytest rl/tests/test_integration_metrics.py -v`

Expected: PASS

**Step 2: Run short training to verify metrics appear in live_metrics.json**

Run: `python -m rl.train --timesteps 1000 --n-envs 1 2>&1 | head -50`

Expected output: Training starts, callbacks are invoked

Run: `cat rl/server_data/live_metrics.json | python -m json.tool | grep -E "drills_built|penalty|action_dist" | head -20`

Expected: Metrics fields present in JSON output

**Step 3: Verify dashboard renders new plots**

Run: `python -m rl.dashboard &`

Wait 5 seconds, then:

Run: `pkill -f "python -m rl.dashboard"`

Expected: Dashboard starts without errors, displays new plots

**Step 4: Run full test suite**

Run: `pytest rl/tests/test_env.py rl/tests/test_training_callbacks.py rl/tests/test_dashboard.py rl/tests/test_integration_metrics.py -v --tb=short`

Expected: All tests PASS (50+)

**Step 5: Check for regressions**

Run: `pytest rl/tests/ -v -k "not integration" 2>&1 | tail -20`

Expected: No failures in existing tests, all new tests PASS

**Step 6: Commit**

```bash
git add rl/tests/test_integration_metrics.py
git commit -m "test: add end-to-end integration test for metrics pipeline"
```

---

## Summary

**Total commits:** 4 (one per task)

**New files:** `rl/tests/test_integration_metrics.py`

**Modified files:** `rl/env/mindustry_env.py`, `rl/callbacks/training_callbacks.py`, `rl/dashboard.py`, `rl/tests/test_env.py`, `rl/tests/test_training_callbacks.py`, `rl/tests/test_dashboard.py`

**Total LOC added:** ~325 (15 + 30 + 200 + 80)

**Backward compatible:** Yes (existing metrics unaffected, graceful handling of missing fields)

**Ready for:** Training validation with `python -m rl.train --timesteps 500000` and parameter tuning based on metrics

---

## Execution Options

Plan complete and saved to `docs/plans/2026-04-03-dashboard-metrics-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task (Tasks 1-4), review code between tasks, fast iteration with inline feedback

**2. Parallel Session (separate)** — Open new session with executing-plans skill, batch execution with checkpoints, full async workflow

**Which approach would you prefer?**
