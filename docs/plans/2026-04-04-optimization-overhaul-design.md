# Mindustry RL Agent — Optimization Overhaul Design

**Date:** 2026-04-04  
**Status:** Approved  
**Scope:** Full redesign of observation space, neural network, reward system, model saving, dashboard, and Mindustry mod helper functions.

---

## 1. Goals

1. Give the agent pre-computed optimization signals (placement scoring, build order, defense coverage, power network, lookahead) as observation features — not agent-driven, to avoid poisoning training data.
2. Populate the spatial CNN grid with meaningful data (currently always zeros).
3. Redesign the neural network with a multi-head critic for decomposed reward learning.
4. Overhaul the reward system into 4 decomposed components with curriculum-weighted heads.
5. Add principled model saving with metadata validation on load.
6. Rebuild the dashboard with a 3D reward surface visualization.
7. Add spatial query commands and enriched state JSON to the Mindustry mod.

All existing checkpoints are discarded. Full retrain from scratch.

---

## 2. Observation Space

### 2.1 Grid (CNN input): `(8, 31, 31)`

| Channel | Content | Encoding |
|---------|---------|----------|
| 0 | Block type | `block_id / max_id` |
| 1 | HP | `0.0–1.0` |
| 2 | Team | `0=neutral, 0.5=ally, 1.0=enemy` |
| 3 | Ore type | `ore_id / max_ore_id` |
| 4 | **Threat score** | `0.0–1.0` — distance-weighted enemy density per tile |
| 5 | **Power coverage** | `0.0–1.0` — tile within power node range |
| 6 | **Build score** | `0.0–1.0` — placement desirability (ore proximity + path to core) |
| 7 | Rotation | `rotation / 3.0` |

Channels 4–6 are **pre-computed Python-side heatmaps**, updated every 5 steps and cached between updates to avoid step latency impact. Computation runs in a background worker thread.

### 2.2 Features (`~121 dims`)

Existing 92 dims are preserved, with these additions:

| Dims | Content | Normalization |
|------|---------|---------------|
| 9 | Placement score per build slot (0-8) | `tanh(score / K)` → [-1, 1] |
| 1 | Power deficit: `max(0, consumed - produced) / capacity` | raw [0, 1] |
| 1 | Defense gap score: fraction of core perimeter without turret coverage | raw [0, 1] |
| 1 | Wave threat index: enemy count × avg hp, normalized | `log(1+x)/log(1+max)` |
| 5 | Build order priority per action type (5 key build actions) | `log(1+x)/log(1+max)` |
| 12 | Lookahead score per action type (3-step forward simulation) | `log(1+x)/log(1+max)` |

**Total: ~121 dims**

### 2.3 Normalization Strategy (hybrid, per your analysis)

- **Python-side first**: Apply `log(1 + score) / log(1 + max_score)` to all lookahead and optimization scores before entering features. Reduces variance and handles exponential growth in late-game.
- **Tanh saturation** on `placement_score_per_slot` (9 dims): `tanh(raw / K)` where K = expected good-placement score. Bounds to [-1, 1] regardless of scale.
- **VecNormalize on top**: Handles remaining drift across features. Runs on the already-log-scaled values.

This prevents VecNormalize from "flattening" deterministic simulation outputs.

---

## 3. Neural Network Architecture

### 3.1 CNN Branch (grid: 8×31×31)

```
Conv2d(8→32, 3×3, stride=1, ReLU)
Conv2d(32→64, 3×3, stride=2, ReLU)
Conv2d(64→128, 3×3, stride=2, ReLU)
Flatten → Linear(~4608, 512, ReLU)
LayerNorm(512)
```

### 3.2 MLP Branch (features: ~121)

```
Linear(121, 256, ReLU)
Linear(256, 256, ReLU)
LayerNorm(256)
```

LayerNorm on both branches before concatenation prevents gradient dominance from one branch.

### 3.3 Fusion

```
Concat(512, 256) → 768
Linear(768, 512, ReLU)
Linear(512, 256, ReLU)
```

Global gradient clipping at fusion: `max_norm=0.5`.

### 3.4 Policy Head (actor)

```
Linear(256, 21)  → MultiDiscrete logits [12 action_types, 9 slots]
```

With action masking from MaskablePPO.

### 3.5 Multi-Head Critic

```
head_survival: Linear(256, 1)  — core_hp_delta + player_alive
head_economy:  Linear(256, 1)  — resources + drill + delivery
head_defense:  Linear(256, 1)  — wave_survived + power_balance
head_build:    Linear(256, 1)  — build_efficiency + new_buildings
```

A **learnable weight vector** (4 scalars, initialized uniform) combines the 4 heads into a single value estimate for PPO:

```
V(s) = w_survival * V_survival + w_economy * V_economy + w_defense * V_defense + w_build * V_build
```

The weight vector is updated via gradient descent alongside the rest of the network. Initial values are set by curriculum phase (see Section 5).

### 3.6 Training Config

```python
policy_kwargs = {
    "net_arch": [],  # replaced by custom feature extractor
    "features_extractor_class": MindustryFeatureExtractor,
}
MaskablePPO(
    learning_rate=3e-4 → 1e-5 (linear schedule),
    n_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.05,
    max_grad_norm=0.5,   # global gradient clipping
)
```

---

## 4. Reward System

### 4.1 Decomposed Reward Components

| Component | Signals | Notes |
|-----------|---------|-------|
| `R_survival` | core_hp_delta × 0.30, player_alive_bonus × 0.20, terminal penalties | Core defense priority |
| `R_economy` | resources_delta, drill_bonus, delivery_bonus, drill_on_ore_bonus, conveyor_connectivity | Mining income |
| `R_defense` | wave_survived_bonus × 0.20, power_balance_bonus × 0.07 | Tactical resilience |
| `R_build` | build_efficiency_bonus × 0.05, graphite/silicon production (future) | Infrastructure |

Each component is normalized to approximately [−1, 1] by scaling.

### 4.2 Combined Reward

```
R_total = w_s * R_survival + w_e * R_economy + w_d * R_defense + w_b * R_build
```

Weights `(w_s, w_e, w_d, w_b)` are **fixed per curriculum phase** for training stability (not the learnable head weights — those are separate):

| Phase | Steps | w_survival | w_economy | w_defense | w_build |
|-------|-------|------------|-----------|-----------|---------|
| mining_only | 0–100k | 0.3 | 1.0 | 0.1 | 0.2 |
| drill_connect | 100k–300k | 0.5 | 0.8 | 0.3 | 0.4 |
| defense_power | 300k–600k | 0.8 | 0.6 | 0.8 | 0.5 |
| full | 600k+ | 0.8 | 0.8 | 0.8 | 0.6 |

### 4.3 Removed

- `inactivity_penalty_a` (WAIT/MOVE repetition) — replaced by lookahead scores structurally penalizing low-value actions
- `inactivity_penalty_b` (resource bleeding) — subsumed by `R_economy` going negative on resource loss

### 4.4 Terminal Penalties (unchanged)

- Core destroyed: −0.4
- Player dead, core ok: −0.5
- Action failed: −0.15

---

## 5. Curriculum Learning

### 5.1 Action Curriculum (4 phases, unchanged structure)

| Phase | Steps | Available Actions |
|-------|-------|-------------------|
| mining_only | 0–100k | WAIT, MOVE, BUILD_DRILL |
| drill_connect | 100k–300k | + BUILD_CONVEYOR |
| defense_power | 300k–600k | + BUILD_POWER, BUILD_WALL, BUILD_TURRET, REPAIR, BUILD_GRAPHITE_PRESS, BUILD_COMBUSTION_GEN |
| full | 600k+ | All 12 actions |

### 5.2 Reward Weight Curriculum

See Section 4.2. Weights shift with phases to prevent mixed signals during early training.

### 5.3 Initial Critic Head Weights (match reward curriculum)

| Phase | w_survival | w_economy | w_defense | w_build |
|-------|------------|-----------|-----------|---------|
| mining_only | 0.3 | 1.0 | 0.1 | 0.2 |
| defense_power | 0.8 | 0.6 | 0.8 | 0.5 |
| full | 0.8 | 0.8 | 0.8 | 0.6 |

Head weights are initialized from the curriculum phase weights at the start of training, then allowed to learn freely via gradient descent.

---

## 6. Optimization Sub-Routines (Python-side, pre-computed)

All sub-routines run in a **background thread** (Python `threading.Thread`), computing every 5 steps. Results are cached and read by the main step loop. Thread-safe handoff via `threading.Lock`.

### 6.1 Placement Scoring

For each slot (0-8) around the player:
- Ore proximity score: inverse distance to nearest ore, weighted by ore type value
- Path-to-core score: Manhattan distance penalty if slot is too far from core
- Threat avoidance: inverse distance to nearest enemy
- Obstruction check: zero if tile is blocked

Formula: `placement_score[slot] = (ore_w * ore_prox + core_w * core_dist + threat_w * threat_dist) / normalization`

### 6.2 Build Order Priority

For each buildable action type, score based on:
- Current resource levels vs cost
- Whether the action's building would improve network topology (drills near ore, conveyors connecting drills, turrets near core perimeter)
- Current curriculum phase (prunes low-priority actions)

### 6.3 Defense Coverage

- Trace a circle around the core at radius `core_size + 3`
- For each point on the perimeter, check if any turret has range covering it
- `defense_gap = uncovered_points / total_perimeter_points`

### 6.4 Power Network Solver

- Sum `produced` vs `consumed` from state
- Identify tiles within range of no power node → `power_uncovered_tiles`
- `power_deficit = max(0, consumed - produced) / max(capacity, 1)`

### 6.5 Lookahead / 3-Step Forward Simulation

For each of the 12 action types (using the best-scoring slot for build actions):
1. Simulate state after taking action: update resource deltas, building list, expected ore output
2. Compute expected R_economy + R_defense after 3 steps (deterministic, no stochasticity)
3. Normalize via `log(1 + score) / log(1 + max_score)`

This is a **pure Python simulation** (no game interaction), using current state as starting point.

---

## 7. Mindustry Mod Changes (`scripts/main.js`)

### 7.1 Enriched State JSON (added fields, sent every tick)

```json
{
  "oreGrid": [[x, y, ore_id], ...],
  "powerNodes": [{"x": x, "y": y, "range": r}, ...],
  "blockedTiles": [[x, y], ...]
}
```

- `oreGrid`: all ore overlay tiles within `gridRadius`
- `powerNodes`: all power-producing nodes with their effective range
- `blockedTiles`: tiles with indestructible blocks (environmental blocks, spawn areas) — prevents the agent from attempting builds that the game will always reject

### 7.2 Query Commands (new, on-demand)

| Command | Response | Purpose |
|---------|----------|---------|
| `SCAN_ORES` | JSON array of `[x, y, ore_id]` for all ores in full map | Full ore map for planning |
| `GET_POWER_NETWORK` | JSON array of power nodes with coverage circles | Power solver input |
| `GET_THREAT_MAP` | JSON array of `[x, y, threat_level]` for enemy positions + attack radius | Threat heatmap input |
| `CHECK_BLOCKED;x;y` | `1` or `0` | Is this tile indestructible/occupied? |
| `GET_BUILD_CANDIDATES;block_name` | JSON array of `[x, y, score]` top-10 valid positions | Server-side placement candidates |

---

## 8. Model Saving & Loading

### 8.1 Checkpoint Structure (per 10k steps)

```
rl/models/
  mindustry_ppo_{step}.zip        — policy + value head weights
  vecnormalize_{step}.pkl         — normalization running stats
  metadata_{step}.json            — fingerprint for validation
```

### 8.2 `metadata_{step}.json` Format

```json
{
  "obs_dims": {"grid": [8, 31, 31], "features": 121},
  "action_space": [12, 9],
  "curriculum_phase": "defense_power",
  "reward_weights": {"survival": 0.8, "economy": 0.6, "defense": 0.8, "build": 0.5},
  "timesteps": 350000,
  "created_at": "2026-04-04T15:30:00"
}
```

### 8.3 Load-time Validation

On load (in `play.py` and `train.py` resume path):
1. Parse `metadata.json`
2. Compare `obs_dims` and `action_space` against current env config
3. If mismatch → raise `RuntimeError` with clear message (fail loudly, never silently corrupt)

---

## 9. Dashboard Redesign

### 9.1 New Layout (replaces existing `dashboard.py`)

Panel layout (16×20 figure, 6×4 grid):

| Row | Col 0 | Col 1 | Col 2 |
|-----|-------|-------|-------|
| 0 | Reward per episode | Episode length | Reward distribution histogram |
| 1 | **3D reward surface** (economy_head vs defense_head → Z=combined) | Critic head values (4 lines) | Action mask ratio |
| 2 | Power grid (produced vs consumed) | Step latency | Buildings/units count |
| 3 | Resource throughput | Stability index (σ) | Lookahead scores heatmap (12 actions × time) |
| 4 | Optimization signals (placement score per slot) | Threat map thumbnail | Defense gap + power deficit |
| 5 | Stats bar | Stats bar | Stats bar |

### 9.2 3D Reward Surface (`ax_3d`)

- **X axis**: `V_economy` value head (rolling mean, last 50 episodes)
- **Y axis**: `V_defense` value head (rolling mean, last 50 episodes)
- **Z axis / color**: Combined reward `R_total` (rolling mean)
- Updates every 10 rollouts (expensive to render)
- Uses `matplotlib` `Axes3D` with `plot_surface` over a 20×20 meshgrid interpolated from collected (economy, defense, reward) triples

### 9.3 Lookahead Scores Heatmap

- 12 actions × last 50 rollouts = heatmap showing which actions the lookahead scores rate highly over time
- Reveals if agent is ignoring high-scoring actions (policy bias) or following them (healthy)

### 9.4 CNN Branch Activation Monitor

- Track mean absolute activation norm of CNN branch output (post LayerNorm) vs MLP branch
- Alert (yellow/red color) if ratio exceeds 3:1 (one branch dominating)

---

## 10. File-Level Change Map

| File | Change Type |
|------|-------------|
| `scripts/main.js` | Add `oreGrid`, `powerNodes`, `blockedTiles` to state JSON; add 5 query commands |
| `rl/env/spaces.py` | Grid channels 4→8; features 92→~121; new normalization helpers |
| `rl/env/mindustry_env.py` | Integrate optimization worker; pass enriched obs |
| `rl/rewards/multi_objective.py` | Decompose into 4 components; curriculum weights |
| `rl/train.py` | Custom feature extractor; multi-head critic; metadata save |
| `play.py` | Load metadata; validate on load |
| `rl/dashboard.py` | Full rewrite: 3D surface, lookahead heatmap, CNN activation monitor |
| `rl/callbacks/training_callbacks.py` | Log critic head values; log optimization signal stats |
| **NEW** `rl/optimization/` | Placement scorer, build planner, defense coverage, power solver, lookahead |
| **NEW** `rl/models/custom_policy.py` | Custom CNN+MLP feature extractor; multi-head critic |

---

## 11. Out of Scope (YAGNI)

- Genetic algorithm / Newton-Raphson optimization inside the agent (agent is pre-computed, not algorithmic)
- Multi-agent (multiple player units)
- Unit command actions beyond WAIT/MOVE
- Saving lookahead score caches between training runs
- Online model updates during inference
