# RL System Design — Mindustry AI Agent

**Date**: 2026-04-02  
**Status**: Approved  
**Scope**: Training pipeline for a Reinforcement Learning agent to play Mindustry using the Mimi Gateway mod

---

## Goal

Train an RL agent capable of full Mindustry strategy: building defenses, collecting resources, managing units, and surviving waves — using A2C via Stable Baselines3 and a Gymnasium environment that connects to the game via the Mimi Gateway TCP mod.

---

## Framework

| Component | Choice | Reason |
|---|---|---|
| RL Library | Stable Baselines3 (SB3) | A2C built-in, callbacks, TensorBoard, battle-tested |
| Env Interface | Gymnasium | Industry standard, SB3 native support |
| Algorithm | A2C | Good for continuous control, single-env, faster than PPO to set up |
| Neural Net | MultiInputPolicy | Supports CNN + MLP combined input natively |

---

## Project Structure

```
rl/
├── env/
│   ├── __init__.py
│   ├── mindustry_env.py       # Main Gymnasium Env
│   ├── mimi_client.py         # TCP client wrapper for Mimi Gateway mod
│   └── spaces.py              # Observation and action space definitions
├── rewards/
│   ├── __init__.py
│   └── multi_objective.py     # Reward function
├── callbacks/
│   ├── __init__.py
│   └── training_callbacks.py  # Checkpointing, logging, early stopping
├── models/                    # Saved model checkpoints (.zip)
├── logs/                      # TensorBoard training logs
├── train.py                   # Training entry point
├── evaluate.py                # Evaluate a saved model
└── requirements.txt           # Python dependencies
```

---

## Observation Space

Two tensors combined via `spaces.Dict`, processed by `MultiInputPolicy`:

### Tensor 1: `grid` — Shape `(4, 31, 31)` → CNN encoder

| Channel | Description | Range |
|---|---|---|
| 0 | Block type (int-encoded, normalized) | 0–1 |
| 1 | Tile HP | 0–1 |
| 2 | Team (0=neutral, 0.5=ally, 1=enemy) | 0–1 |
| 3 | Rotation (0–3, normalized) | 0–1 |

Source: `state["grid"]` from Mimi Gateway (31×31 radius 15 snapshot)

### Tensor 2: `features` — Shape `(43,)` → MLP encoder

| Group | Features | Size |
|---|---|---|
| Core | hp | 1 |
| Resources | copper, lead, graphite, titanium, thorium (normalized /1000) | 5 |
| Power | produced, consumed, stored, capacity (normalized) | 4 |
| Wave | wave number (norm), ticks_to_next_wave (norm) | 2 |
| Enemies (top 5) | hp, x, y, type_encoded per enemy | 20 |
| Friendly units (top 3) | hp, x, y per unit | 9 |
| Core position | x, y (normalized by map size) | 2 |
| **Total** | | **43** |

Missing enemies/units are zero-padded.

```python
obs_space = spaces.Dict({
    "grid":     spaces.Box(0.0, 1.0, shape=(4, 31, 31), dtype=np.float32),
    "features": spaces.Box(-np.inf, np.inf, shape=(43,), dtype=np.float32),
})
```

---

## Action Space

Hybrid: discrete action type + integer tile coordinates.

| Index | Action | Uses x, y | Notes |
|---|---|---|---|
| 0 | WAIT | No | No-op |
| 1 | BUILD_TURRET (duo) | Yes | Builds at (x, y) in grid |
| 2 | BUILD_WALL | Yes | |
| 3 | BUILD_SOLAR | Yes | |
| 4 | REPAIR | Yes | Repairs building at (x, y) |
| 5 | MOVE_UNIT | Yes | Moves first available unit to (x, y) |
| 6 | ATTACK | Yes | Commands first unit to attack (x, y) |
| 7 | SPAWN_UNIT (poly) | Yes | Factory at (x, y) spawns poly |

```python
action_space = spaces.Dict({
    "action_type": spaces.Discrete(8),
    "x":           spaces.Box(0, 30, shape=(1,), dtype=np.int32),
    "y":           spaces.Box(0, 30, shape=(1,), dtype=np.int32),
})
```

Coordinates are relative to the 31×31 grid snapshot (0 = leftmost tile in view).

---

## Reward Function

```python
reward = (
    + 0.50 * core_hp_delta            # Main signal: penalize losing core HP
    + 0.20 * wave_survived_bonus       # +1.0 upon completing each wave
    + 0.15 * (resources_delta / 500)   # Encourage resource accumulation (normalized)
    + 0.15 * friendly_units_ratio      # Maintain units alive
    - 0.001                            # Time penalty (encourages efficiency)
)

# Terminal penalties / bonuses
if core_destroyed:
    reward -= 1.0
    done = True
```

**Episode ends when:**
- Core HP reaches 0 (loss)
- Max steps reached (configurable, default 5000)

---

## Training Loop

```
train.py
  └─ creates MindustryEnv (connects to Mimi Gateway on localhost:9000)
  └─ wraps with SB3 A2C(policy="MultiInputPolicy")
  └─ trains with callbacks:
       - CheckpointCallback  → saves model every N steps to models/
       - EvalCallback        → evaluates on separate env every N steps
       - TensorboardCallback → logs reward, loss, entropy to logs/
```

**Hyperparameters (starting point):**

| Param | Value |
|---|---|
| learning_rate | 7e-4 |
| n_steps | 128 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| ent_coef | 0.01 |
| total_timesteps | 1_000_000 |

---

## Dependencies (`requirements.txt`)

```
stable-baselines3[extra]>=2.0.0
gymnasium>=0.29.0
torch>=2.0.0
numpy>=1.24.0
tensorboard>=2.13.0
```

---

## Scalability Notes

- **Add new actions**: extend `action_space["action_type"]` Discrete(N) and add handler in `mindustry_env.py`
- **Add new obs features**: extend `features` vector in `spaces.py`  
- **Add grid CNN**: already in design from day 1  
- **Multi-env training**: wrap with `SubprocVecEnv` (requires multiple Mindustry instances)
- **Upgrade algorithm**: swap A2C → PPO with one line change (same interface)
