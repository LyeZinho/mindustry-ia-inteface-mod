# Wave-Hybrid Curriculum, Lookahead Fix & Placement Signal Design

## Problem

The agent spams drills exclusively because:
1. Phase 0 curriculum (0–100k steps) only permits WAIT/MOVE/DRILL/DELETE — no conveyors, no defense
2. Lookahead simulator shows "sea of red" because existing drills don't contribute income in simulation, and DELETE scores 0
3. Placement features cluster near zero when most surrounding tiles are blocked (agent has already built drills everywhere), giving the network no gradient signal

Observed metrics confirming this:
- 46.9% masked actions (curriculum too narrow)
- 54.3% build fail rate (agent retries occupies tiles)
- Lookahead heatmap almost entirely red except step ~42
- σ = 1.628 (agent has converged to stable but suboptimal strategy)

## Design

### 1 — Wave-Hybrid Curriculum

`CURRICULUM_PHASES` entries change from `(ts_start, ts_end)` to `(ts_threshold, wave_threshold)`. A phase is unlocked when `timestep >= ts_threshold OR wave >= wave_threshold`. The agent advances to the highest phase whose condition is met.

| Phase | Timestep gate | Wave gate | Actions |
|---|---|---|---|
| `bootstrap` | 0 (always) | 0 (always) | WAIT, MOVE, DRILL, **CONVEYOR**, DELETE |
| `drill_defend` | 30 000 | 3 | + WALL, TURRET |
| `advanced` | 100 000 | 5 | + POWER, REPAIR, PNEUMATIC_DRILL, COMBUSTION_GEN |
| `full` | 300 000 | 8 | all 13 |

Key changes vs current:
- CONVEYOR promoted to phase 0 (mine → connect from step 1)
- WALL + TURRET promoted from phase 2 to phase 1 (protect drills once income flows)
- Phase thresholds compressed 3-10× (current: 100k/300k/600k → new: 30k/100k/300k)
- Wave number as an OR-gate means a fast game can unlock phases before training timestep

`apply_curriculum_action_mask(timestep, wave=0)` gains a `wave` parameter. `action_masks()` in `mindustry_env.py` extracts `wave = int(self._prev_state.get("wave", 0))` and passes it.

`_REWARD_WEIGHT_PHASES` thresholds updated to match new step thresholds (30k/100k/300k).

### 2 — Lookahead Pessimism Fix

Two issues in `lookahead.py`:

**Existing building income missing**: The 3-step simulation starts from current resources but never advances them with income from already-placed buildings. An agent with 3 copper drills looks the same as one with 0 drills.

Fix: compute `existing_income` from `state["buildings"]` using `BUILDING_INCOME`, then add this income to `sim_state["resources"]` at each of the 3 simulated steps before testing affordability.

**DELETE always scores 0**: `block=None → pass → total_score=0`. The network treats DELETE as worthless.

Fix: when `action_def.block is None`, count ally non-core buildings in the state. If any exist, set `total_score = 0.5` (repositioning has value). Score becomes `log(1.5)/log(11) ≈ 0.17` — a small positive signal that grows the action's lookahead feature above zero.

### 3 — Placement Signal (Tanh K + Blocked)

`placement.py` currently:
- Blocked tiles → `0.0` (no signal)
- `tanh(raw_score / 3.0)` — K=3.0

With K=3.0, an unblocked tile with no ore scores ~0.39 and a blocked tile scores 0.0. When most surroundings are blocked, the network sees near-zero scores everywhere with no clear gradient.

Fix:
- Blocked tiles → `-0.5` (explicit "avoid" signal)
- K: `3.0 → 2.0` — unblocked empty slot: 0.39→0.56, copper ore slot: 0.79→0.93; spread between ore and no-ore increases

Downstream: `feat[92:101] = np.clip(placement, -1.0, 1.0)` in `spaces.py` already handles the extended range — no change needed there.
