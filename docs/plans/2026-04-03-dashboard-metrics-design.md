# Dashboard Metrics Extension Design
**Date:** 2026-04-03  
**Status:** Approved  
**Scope:** Add 3 new metric families (6 total plots) to monitor drill construction, inactivity penalties, and action distribution

---

## Overview

Extend the RL training dashboard to instrument reward function tuning with real-time visibility into:
1. **Drill Construction Rate** — Total drills built + frequency (% of steps)
2. **Inactivity Penalty Frequency** — Counts + % of steps with penalties triggered
3. **Action Distribution** — Per-episode histogram + rolling 100-episode trend

---

## Architecture & Data Flow

**Origin:** Metrics originate in `MindustryEnv.step()` where state changes are observable.

**Collection Pipeline:**
```
MindustryEnv.step() 
  → info dict (drill deltas, penalty triggers, action counts)
  → LiveMetricsCallback._compute_metrics()
  → Aggregation (episode totals, frequencies, distributions)
  → live_metrics.json (persisted per episode)
  → dashboard.py (read, compute rolling windows, render)
```

**Data Structure (added to live_metrics.json):**
```json
{
  "episode": 42,
  "drills_built_total": 5,
  "drill_build_frequency_pct": 2.3,
  "penalty_a_count": 12,
  "penalty_b_count": 3,
  "penalty_frequency_pct": 8.5,
  "action_dist": {
    "WAIT": 0.30,
    "MOVE": 0.25,
    "BUILD_TURRET": 0.05,
    "BUILD_WALL": 0.03,
    "BUILD_POWER": 0.08,
    "BUILD_DRILL": 0.15,
    "REPAIR": 0.14
  }
}
```

---

## Implementation Details

### 1. MindustryEnv.step() Changes

**Metric Tracking (in info dict):**
- `drills_built_this_step` — Count of new mechanical-drill blocks (set diff on building positions)
- `penalty_a_triggered` — 1 if inactivity_penalty_a != 0, else 0
- `penalty_b_triggered` — 1 if inactivity_penalty_b != 0, else 0
- `action_taken_index` — Current action (0-6)
- `step_count` — Total steps in episode (for normalization)

**Implementation:**
```python
# In step() after compute_reward()
new_drills = self._detect_new_drills(prev_state, curr_state)
info["drills_built_this_step"] = new_drills
info["penalty_a_triggered"] = 1 if inactivity_penalty_a != 0 else 0
info["penalty_b_triggered"] = 1 if inactivity_penalty_b != 0 else 0
info["action_taken_index"] = action
info["step_count"] = self.step_count
```

### 2. LiveMetricsCallback._compute_metrics() Changes

**Aggregation logic (per episode):**
```python
# Accumulate across all steps
total_drills = sum(step_info["drills_built_this_step"])
total_penalty_a = sum(step_info["penalty_a_triggered"])
total_penalty_b = sum(step_info["penalty_b_triggered"])
num_steps = episode_length

# Compute frequencies and distributions
drill_frequency_pct = (total_drills / num_steps) * 100
penalty_frequency_pct = ((total_penalty_a + total_penalty_b) / num_steps) * 100
action_dist = normalize_action_counts(action_counter_dict)

# Store in episode summary
episode_metrics = {
    "drills_built_total": total_drills,
    "drill_build_frequency_pct": drill_frequency_pct,
    "penalty_a_count": total_penalty_a,
    "penalty_b_count": total_penalty_b,
    "penalty_frequency_pct": penalty_frequency_pct,
    "action_dist": action_dist
}
```

### 3. Dashboard.py Changes

**6 new drawing functions:**

| Function | Type | Data Source |
|----------|------|-------------|
| `_draw_drill_rate_total()` | Line plot | drills_built_total per episode |
| `_draw_drill_rate_frequency()` | Line plot | drill_build_frequency_pct per episode |
| `_draw_penalty_counts()` | Grouped bar | penalty_a_count vs penalty_b_count |
| `_draw_penalty_frequency()` | Line plot | penalty_frequency_pct per episode |
| `_draw_action_dist_per_episode()` | Stacked bar | action_dist (latest episode) |
| `_draw_action_dist_rolling()` | Stacked area | rolling 100-episode action trend |

**Integration:**
- Add 6 calls to `make_updater()` function
- Arrange in 2 rows of 3 plots for clarity
- Share x-axis with episode number for consistency

---

## Error Handling

**Division by zero:**
- If `num_steps` = 0 (shouldn't happen) → frequencies default to 0

**Missing metrics:**
- If JSON missing drill/penalty fields → dashboard treats as 0 (graceful degradation)
- If JSON missing action_dist → use empty dict (renders blank bar)

**State comparison edge cases:**
- No drills built → drills_built_total = 0
- No actions taken (impossible) → action_dist = uniform distribution
- All penalties triggered → penalty_frequency_pct can approach 100%

---

## Testing

**Unit Tests (in test_dashboard.py):**
1. `test_drill_rate_calculation` — Verify drills_built and frequency math
2. `test_penalty_count_aggregation` — Verify penalty_a + penalty_b counts
3. `test_action_dist_normalization` — Verify action distribution sums to 1.0
4. `test_rolling_window_aggregation` — Verify 100-episode rolling window computation

**Integration Tests:**
1. Run 5 dummy episodes with synthetic metrics
2. Verify all 6 metrics appear in live_metrics.json
3. Verify dashboard renders without errors

**Regression Tests:**
1. Existing reward/length plots unaffected
2. Old live_metrics.json (without new fields) doesn't crash dashboard

---

## Files Modified

| File | Changes | LOC Impact |
|------|---------|-----------|
| `rl/env/mindustry_env.py` | Add metric tracking to info dict in step() | +15 lines |
| `rl/callbacks/training_callbacks.py` | Extend _compute_metrics() aggregation | +30 lines |
| `rl/dashboard.py` | Add 6 new drawing functions + integration | +200 lines |
| `rl/tests/test_dashboard.py` | Add 4 unit + 2 integration tests | +80 lines |

**Total:** ~325 lines added, 0 lines removed (backward compatible)

---

## Backward Compatibility

- Existing metrics (rewards, episode lengths, masks) unchanged
- Old live_metrics.json files won't break (new fields simply absent)
- Dashboard gracefully handles missing metrics (defaults to 0/empty)
- No configuration flags or environment changes needed

---

## Success Criteria

✅ All 6 metrics display correctly during training  
✅ Drill construction bonus visible in drill_rate_frequency  
✅ Inactivity penalties visible in penalty_frequency trend  
✅ Action distribution shows increasing BUILD_DRILL over time  
✅ No regressions in existing dashboard functionality  
✅ All tests passing (unit + integration + regression)

---

## Next Steps

1. **Implementation:** Invoke writing-plans to create detailed implementation plan
2. **Execution:** Implement changes in 4 separate PRs (env → callback → dashboard → tests)
3. **Validation:** Run training with new metrics, verify signal quality
4. **Tuning:** Use metrics to inform parameter adjustments (penalties, thresholds)
