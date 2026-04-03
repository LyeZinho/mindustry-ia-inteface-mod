# Design: Multi-Agent Training, Game Speed, Spectator Mode, Reward Shaping

**Date:** 2026-04-03  
**Status:** Approved  
**Scope:** 4 new features on top of the existing single-agent RL pipeline

---

## 1. Multi-Agent Parallel Training

### Goal
Run 4 parallel Mindustry server instances, each with one AI agent (poly unit), all feeding experience into a single shared A2C model via SB3's `SubprocVecEnv`. This is what "shared knowledge" means — standard SB3 parallel training.

### Architecture

```
train.py (--n-envs 4)
  └── SubprocVecEnv([env_0, env_1, env_2, env_3])
        ├── MindustryEnv(tcp_port=9000)  ─── Mindustry server instance_0 (TCP 9000, game 6567)
        ├── MindustryEnv(tcp_port=9001)  ─── Mindustry server instance_1 (TCP 9001, game 6568)
        ├── MindustryEnv(tcp_port=9002)  ─── Mindustry server instance_2 (TCP 9002, game 6569)
        └── MindustryEnv(tcp_port=9003)  ─── Mindustry server instance_3 (TCP 9003, game 6570)
```

All 4 envs step in parallel. SB3 batches their experience and updates the single model. No exotic inter-agent communication.

### Changes

#### `rl/server/manager.py`
- Add `start_n_servers(n, base_tcp_port=9000, base_game_port=6567, base_data_dir="rl/server_data", jar_path="server-release.jar", mod_zip="mimi-gateway-v1.0.4.zip")` function.
- For each `i` in `range(n)`:
  - Creates data dir `{base_data_dir}/instance_{i}/config/mods/` and installs mod zip.
  - Writes `{base_data_dir}/instance_{i}/mimi_port.txt` containing `str(base_tcp_port + i)`.
  - Instantiates `MindustryServer(jar_path, data_dir=f"{base_data_dir}/instance_{i}", port=base_tcp_port+i)`.
- Starts all N servers concurrently (threads), waits for all to be ready.
- Returns list of `MindustryServer` instances.
- `MindustryServer` gets optional `game_port: int` parameter; if set, passes `-Dmindustry.port={game_port}` as a JVM arg.

#### `scripts/main.js`
- At startup (before `startSocketServer()`), read `Vars.dataDirectory.child("mimi_port.txt")`:
  - If file exists and is parseable, set `config.port` to that value.
  - Fallback to `9000` on any error.

#### `rl/env/mindustry_env.py`
- `MindustryEnv.__init__` gains `tcp_port: int = 9000` parameter.
- Existing `port` parameter renamed to `tcp_port` (or `tcp_port` added alongside existing `port` for backward compat — check current signature).

#### `rl/train.py`
- Add `--n-envs N` argument (default `4`).
- When `n_envs == 1`: existing single-env path (backward compat, uses `Monitor`).
- When `n_envs > 1`: call `start_n_servers(n_envs, ...)`, build `SubprocVecEnv` with lambda factories, wrap with `VecMonitor`.
- `_install_mod` call removed from top-level; each server instance installs its own mod copy inside `start_n_servers`.
- Graceful shutdown: stop all servers on exit/SIGTERM.

#### Tests
- `rl/tests/test_server_manager.py`: add tests for `start_n_servers` (mock `MindustryServer.start`).
- `rl/tests/test_env.py` / `test_spaces.py`: ensure `tcp_port` param doesn't break existing tests.

---

## 2. Spectator Mode for Human Joiners

### Goal
Any human player who connects to a training server instance is automatically put into spectator mode (cannot build, interact, or interfere with the AI's game).

### Changes

#### `scripts/main.js`
- Register `EventType.PlayerJoin` hook at mod startup (alongside existing `EventType.Tap` hooks):

```javascript
Events.on(EventType.PlayerJoin.class, event => {
    let p = event.player;
    p.team(Team.derelict);
    Call.sendMessage("[yellow][Mimi AI] Você entrou como espectador. Aproveite o treinamento!");
});
```

- The AI's `poly` unit is spawned server-side (not as a connecting player), so it is unaffected by this hook.

---

## 3. Game Speed — 2× Wave Timer

### Goal
Halve the time between waves so the agent experiences more waves per wall-clock second.

### Changes

#### `scripts/main.js`
- In `handleResetCommand`, after `Vars.net.host()` / map load, add:

```javascript
Vars.state.rules.waveSpacing = 7200; // half of default ~14400 ticks
```

#### `rl/server/manager.py`
- In `start()`, after `_ready` event fires, send `fps 120` over stdin after a short delay (1 second). This hints the JVM to run faster when the machine can support it:

```python
# after self._ready.wait() succeeds
import threading
def _send_fps():
    time.sleep(1.0)
    self.send_stdin("fps 120")
threading.Thread(target=_send_fps, daemon=True).start()
```

---

## 4. Reward Shaping — Drill Activity Bonus

### Goal
Add a small incentive for the agent to place drills on ore tiles and keep them working. Uses the existing `resources.copper` field as a proxy (copper is the most abundant ore on most maps).

### New reward formula

```
reward = 0.35 * core_hp_delta
       + 0.20 * wave_survived_bonus
       + 0.15 * resources_delta / 500
       + 0.10 * drill_bonus          # NEW: copper increased >= 5 this step
       + 0.08 * power_balance_bonus
       + 0.07 * build_efficiency_bonus
       + 0.05 * player_alive_bonus
       - 0.0005                       # time penalty (halved from 0.001)
```

Terminal penalties unchanged: `-1.0` core destroyed, `-0.5` player dead.

### `drill_bonus` definition
```python
prev_copper = float(prev_state.get("resources", {}).get("copper", 0.0))
curr_copper = float(curr_state.get("resources", {}).get("copper", 0.0))
drill_bonus = 1.0 if (curr_copper - prev_copper) >= 5.0 else 0.0
```

### Changes

#### `rl/rewards/multi_objective.py`
- Update docstring with new weights.
- Add `drill_bonus` computation (copper delta ≥ 5).
- Update the `reward = ...` expression with new weights.

---

## Files Changed (Summary)

| File | Change |
|---|---|
| `scripts/main.js` | Read `mimi_port.txt` for TCP port; PlayerJoin → spectator; waveSpacing=7200 in RESET |
| `rl/server/manager.py` | Add `game_port` JVM arg support; `start_n_servers()` factory; post-ready `fps 120` |
| `rl/env/mindustry_env.py` | Add `tcp_port` parameter |
| `rl/train.py` | Add `--n-envs`, SubprocVecEnv path, multi-server startup/shutdown |
| `rl/rewards/multi_objective.py` | New weights + drill_bonus |
| `rl/tests/test_server_manager.py` | Tests for `start_n_servers` |
| `mimi-gateway-v1.0.4.zip` | Repackage after JS changes |
| `rl/server_data/config/mods/mimi-gateway.zip` | Copy of updated zip |

---

## Repackage command (after JS changes)

```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('mimi-gateway-v1.0.4.zip', 'w', zipfile.ZIP_DEFLATED)
z.write('mod.hjson'); z.write('scripts/main.js')
z.close()
"
cp mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
```

## Test command

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/ -v
```
