# Delete Action, Penalty Tuning & Curriculum Fixes — Design

## Summary

Three coordinated changes to unblock the agent from a high-rejection / fear-of-building local optimum:

1. **DELETE action** — new action type that demolishes any allied building at a relative slot (core excluded).
2. **Penalty softening** — reduce `actionFailed` penalty (−0.15 → −0.05) and loosen the resource-bleeding threshold (−10 → −50) so exploration is not punished into extinction.
3. **Curriculum update** — include DELETE in phases 0 and 1 so the agent can recover from bad drill placements from the very first episode.

---

## Decision Log

| Decision | Options considered | Choice | Reason |
|---|---|---|---|
| DELETE action representation | New action type (A), overload REPAIR (B), atomic REPOSITION (C) | **A** | Clean semantics, consistent with BUILD slot system, easy to mask |
| Penalty reduction | Cut to −0.05 (A), remove entirely (B) | **A** | Keeps signal, removes terror |
| Resource bleeding threshold | −50 (A), remove check (B) | **A** | A single 12-copper drill build was triggering the penalty; −50 is calibrated to a realistic "wasteful build" scenario |
| Curriculum gating | DELETE in phase 0+1 (A), phase 2+ only (B) | **A** | Agent needs to fix mistakes from step 1 |

---

## Architecture

### Action Space Change

Current: `MultiDiscrete([12, 9])` — 12 types × 9 slots  
After:   `MultiDiscrete([13, 9])` — 13 types × 9 slots

New entry in `ACTION_REGISTRY` (index 12):
```python
ActionDef("DELETE", None, lambda r: True)
```

The `block` field is `None` (same as WAIT/MOVE/REPAIR) — no resource requirement.

### Masking Logic

`compute_action_mask` in `spaces.py`:
- DELETE is valid only for slots that contain an **allied building** that is **not the core**.
- Core detection: `building["block"]` contains `"core"` (matches `"core-sharded"`, `"core-foundation"`, etc.).
- Slot is unmasked only if `(target_x, target_y) in ally_buildings_set` and the block at that position is not a core variant.

### Mod Command

`DELETE` already exists in `main.js` (`handlePlayerDeleteCommand`):
```
DELETE;x;y
```
`mindustry_env.py` `_execute_action` will send `DELETE;x;y` using the same `player_x + SLOT_DX[arg]` coordinate calculation used for BUILD.

### Penalty Changes

In `multi_objective.py`:
- `actionFailed` deduction: `−0.15 → −0.05` (line inside `compute_reward`)
- `bleeding_threshold` default: `−10.0 → −50.0` (parameter default in `_detect_resource_bleeding_penalty`)

### Curriculum Phases

Current phase 0: `[ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL]`  
New phase 0:     `[ACTION_WAIT, ACTION_MOVE, ACTION_BUILD_DRILL, ACTION_DELETE]`

Current phase 1: `[…, ACTION_BUILD_CONVEYOR]`  
New phase 1:     `[…, ACTION_BUILD_CONVEYOR, ACTION_DELETE]`

Phases 2 and 3 already include all actions via their full list; DELETE is added by extending the index range from `12` to `13`.

---

## Files Touched

| File | Change |
|---|---|
| `rl/env/spaces.py` | Add `ACTION_DELETE` to registry; update `NUM_ACTION_TYPES`; mask DELETE slots |
| `rl/env/mindustry_env.py` | Handle `ACTION_DELETE` in `_execute_action` |
| `rl/rewards/multi_objective.py` | Lower `actionFailed` penalty; raise bleeding threshold; add DELETE to curriculum phases |
| `scripts/main.js` | No change needed — `DELETE;x;y` command already implemented |
| `mimi-gateway-v1.0.7.zip` | No rebuild needed |

---

## Testing

- `test_spaces.py`: mask allows DELETE only on occupied allied non-core slots; DELETE masked when slot is empty or is core
- `test_env.py`: `_execute_action` sends correct `DELETE;x;y` command for each slot
- `test_reward.py`: `actionFailed` penalty is now −0.05; bleeding threshold fires at −50 not −10
- `test_reward.py`: curriculum phase 0 allows DELETE; phase 1 allows DELETE
- All 196 existing tests must continue to pass

---

## Non-Goals

- No DELETE reward/bonus (demolishing is neutral; the reward comes from the subsequent successful BUILD)
- No undo/confirmation mechanics
- No mod-side changes
