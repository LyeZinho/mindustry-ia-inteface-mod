# Multi-Agent + Speed + Spectator + Reward Shaping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 4-agent parallel RL training (SubprocVecEnv), spectator mode for human joiners, 2× faster wave timer, and improved reward shaping with a drill-activity bonus.

**Architecture:** `manager.py` gains `start_n_servers()` to spawn N isolated Mindustry server instances (each with its own data dir and TCP port); `train.py` uses `SubprocVecEnv` when `--n-envs > 1`; `main.js` reads its TCP port from `mimi_port.txt`, enforces spectator mode on `PlayerJoin`, and sets `waveSpacing=7200` on RESET.

**Tech Stack:** Python 3, Stable-Baselines3, Gymnasium, JavaScript (Rhino/Mindustry mod runtime), pytest, threading

---

## Repackage command (run after ANY change to `scripts/main.js`)

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

---

## Task 1: Reward shaping — drill bonus + rebalanced weights

**Files:**
- Modify: `rl/rewards/multi_objective.py`
- Test: `rl/tests/test_multi_objective_reward.py`

### Step 1: Write failing tests

Add these tests to `rl/tests/test_multi_objective_reward.py`:

```python
def test_drill_bonus_when_copper_increases_significantly():
    prev = {"resources": {"copper": 10}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    curr = {"resources": {"copper": 20}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    r = compute_reward(prev, curr, done=False)
    # drill_bonus=1.0 (copper +10 >= 5), weighted at 0.10 → reward should be higher than baseline
    assert r > -0.002  # time penalty only without drill bonus would be ~ -0.0005 + 0.05 player alive

def test_no_drill_bonus_when_copper_increases_less_than_5():
    prev = {"resources": {"copper": 10}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    curr = {"resources": {"copper": 13}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    r_small = compute_reward(prev, curr, done=False)
    prev2 = {"resources": {"copper": 10}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    curr2 = {"resources": {"copper": 16}, "core": {"hp": 0.9}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": True}}
    r_large = compute_reward(prev2, curr2, done=False)
    assert r_large > r_small  # drill bonus fires for +6, not for +3

def test_time_penalty_halved():
    """Time penalty should now be 0.0005 not 0.001."""
    prev = {"resources": {}, "core": {"hp": 0.5}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": False}}
    curr = {"resources": {}, "core": {"hp": 0.5}, "wave": 1, "power": {}, "buildings": [], "player": {"alive": False}}
    r = compute_reward(prev, curr, done=False)
    # No positive signals, no negative terminal → just time penalty (-0.0005)
    assert abs(r - (-0.0005)) < 1e-6
```

**Step 2: Run tests to verify they fail**

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/test_multi_objective_reward.py::test_drill_bonus_when_copper_increases_significantly rl/tests/test_multi_objective_reward.py::test_no_drill_bonus_when_copper_increases_less_than_5 rl/tests/test_multi_objective_reward.py::test_time_penalty_halved -v
```

Expected: FAIL (drill_bonus doesn't exist yet, time penalty still 0.001)

### Step 3: Implement in `rl/rewards/multi_objective.py`

Replace the entire file content with:

```python
"""
Multi-objective reward function for the Mindustry RL player agent.

reward = 0.35 * core_hp_delta
       + 0.20 * wave_survived_bonus
       + 0.15 * resources_delta / 500
       + 0.10 * drill_bonus          (copper increased >= 5 this step)
       + 0.08 * power_balance_bonus
       + 0.07 * build_efficiency_bonus
       + 0.05 * player_alive_bonus
       - 0.0005                       (time penalty)

Terminal penalties:
  core destroyed        → -1.0
  player dead, core ok  → -0.5
"""
from __future__ import annotations

from typing import Any, Dict


def compute_reward(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    done: bool,
) -> float:
    prev_hp = float(prev_state.get("core", {}).get("hp", 0.0))
    curr_hp = float(curr_state.get("core", {}).get("hp", 0.0))
    core_hp_delta = curr_hp - prev_hp

    prev_wave = int(prev_state.get("wave", 0))
    curr_wave = int(curr_state.get("wave", 0))
    wave_survived_bonus = 1.0 if curr_wave > prev_wave else 0.0

    def _total_resources(state: Dict[str, Any]) -> float:
        return sum(float(v) for v in state.get("resources", {}).values())

    resources_delta = _total_resources(curr_state) - _total_resources(prev_state)

    prev_copper = float(prev_state.get("resources", {}).get("copper", 0.0))
    curr_copper = float(curr_state.get("resources", {}).get("copper", 0.0))
    drill_bonus = 1.0 if (curr_copper - prev_copper) >= 5.0 else 0.0

    power = curr_state.get("power", {})
    produced = float(power.get("produced", 0.0))
    consumed = float(power.get("consumed", 0.0))
    if produced > 0:
        power_balance_bonus = max(0.0, min(1.0, (produced - consumed) / produced))
    else:
        power_balance_bonus = 0.0

    prev_buildings = len(prev_state.get("buildings", []))
    curr_buildings = len(curr_state.get("buildings", []))
    new_buildings = max(0, curr_buildings - prev_buildings)
    build_efficiency_bonus = min(1.0, new_buildings * 0.1)

    player_alive = bool(curr_state.get("player", {}).get("alive", False))
    core_destroyed = curr_hp <= 0.0
    player_alive_bonus = 1.0 if (player_alive and not core_destroyed) else 0.0

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

    if done:
        if curr_hp <= 0.0:
            reward -= 1.0
        elif not player_alive:
            reward -= 0.5

    return float(reward)
```

### Step 4: Run all reward tests

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/test_multi_objective_reward.py -v
```

Expected: ALL PASS

### Step 5: Run full test suite

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/ -v
```

Expected: ALL PASS (48+ tests)

### Step 6: Commit

```bash
git add rl/rewards/multi_objective.py rl/tests/test_multi_objective_reward.py
git commit -m "feat(reward): add drill_bonus, rebalance weights, halve time penalty"
```

---

## Task 2: Mod — spectator mode on PlayerJoin

**Files:**
- Modify: `scripts/main.js` (near the `init()` function at line ~900)

### Step 1: Add PlayerJoin event hook

In `scripts/main.js`, find the `// MOD INITIALIZATION` section (around line 900). Just before `startSocketServer();` inside `init()`, add the spectator hook:

```javascript
// Spectator mode: any human player who joins is set to Team.derelict
Events.on(EventType.PlayerJoin.class, event => {
    let p = event.player;
    p.team(Team.derelict);
    Call.sendMessage("[yellow][Mimi AI] Você entrou como espectador. Aproveite o treinamento!");
    Log.info("[Mimi Gateway] Player " + p.name + " set to spectator (Team.derelict)");
});
```

The exact location to insert: after `Log.info("==============================================");` (the second one, just before `startSocketServer()` call).

### Step 2: Repackage mod

```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('mimi-gateway-v1.0.4.zip', 'w', zipfile.ZIP_DEFLATED)
z.write('mod.hjson'); z.write('scripts/main.js')
z.close()
"
cp mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
```

### Step 3: Run full test suite (mod changes don't break Python tests)

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/ -v
```

Expected: ALL PASS

### Step 4: Commit

```bash
git add scripts/main.js mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
git commit -m "feat(mod): set human joiners to spectator mode via PlayerJoin event"
```

---

## Task 3: Mod — read TCP port from mimi_port.txt + 2× game speed

**Files:**
- Modify: `scripts/main.js`

### Step 1: Add port-from-file logic

At the top of `scripts/main.js`, right after the `config` object definition (after line 13, the closing `};`), add:

```javascript
// Read TCP port from mimi_port.txt if present (used for multi-instance training)
(function() {
    try {
        let portFile = Vars.dataDirectory.child("mimi_port.txt");
        if (portFile.exists()) {
            let portStr = portFile.readString().trim();
            let parsed = parseInt(portStr);
            if (!isNaN(parsed) && parsed > 0) {
                config.port = parsed;
                Log.info("[Mimi Gateway] Port loaded from mimi_port.txt: " + config.port);
            }
        }
    } catch(e) {
        Log.info("[Mimi Gateway] mimi_port.txt not found, using default port " + config.port);
    }
})();
```

### Step 2: Add waveSpacing in handleResetCommand

In `handleResetCommand` (around line 856), after `Vars.logic.play();` and before the nested `Core.app.post(()` that spawns the unit, add:

```javascript
// Set wave spacing to half normal (2× faster waves)
Vars.state.rules.waveSpacing = 7200;
Log.info("[Mimi Gateway] waveSpacing set to 7200 (2x faster)");
```

### Step 3: Repackage mod

```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('mimi-gateway-v1.0.4.zip', 'w', zipfile.ZIP_DEFLATED)
z.write('mod.hjson'); z.write('scripts/main.js')
z.close()
"
cp mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
```

### Step 4: Run full test suite

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/ -v
```

Expected: ALL PASS

### Step 5: Commit

```bash
git add scripts/main.js mimi-gateway-v1.0.4.zip rl/server_data/config/mods/mimi-gateway.zip
git commit -m "feat(mod): read TCP port from mimi_port.txt; set waveSpacing=7200 on RESET"
```

---

## Task 4: manager.py — game_port JVM arg + fps 120 after ready + start_n_servers()

**Files:**
- Modify: `rl/server/manager.py`
- Test: `rl/tests/test_server_manager.py`

### Step 1: Write failing tests

Add to `rl/tests/test_server_manager.py`:

```python
from rl.server.manager import MindustryServer, start_n_servers
import shutil


def test_start_n_servers_returns_n_instances(tmp_path):
    """start_n_servers creates N MindustryServer objects."""
    mod_zip = tmp_path / "mimi-gateway-v1.0.4.zip"
    mod_zip.write_bytes(b"fake")

    started = []

    def fake_start(self, timeout=30):
        started.append(self)
        self._proc = MagicMock()
        self._proc.poll.return_value = None
        self._ready.set()

    with patch.object(MindustryServer, "start", fake_start):
        servers = start_n_servers(
            n=3,
            base_tcp_port=9100,
            base_game_port=6700,
            base_data_dir=str(tmp_path / "servers"),
            jar_path="fake.jar",
            mod_zip=str(mod_zip),
        )

    assert len(servers) == 3
    assert len(started) == 3


def test_start_n_servers_writes_port_files(tmp_path):
    """Each instance's data dir gets a mimi_port.txt with the right port."""
    mod_zip = tmp_path / "mimi-gateway-v1.0.4.zip"
    mod_zip.write_bytes(b"fake")

    def fake_start(self, timeout=30):
        self._proc = MagicMock()
        self._proc.poll.return_value = None
        self._ready.set()

    with patch.object(MindustryServer, "start", fake_start):
        start_n_servers(
            n=2,
            base_tcp_port=9200,
            base_game_port=6800,
            base_data_dir=str(tmp_path / "servers"),
            jar_path="fake.jar",
            mod_zip=str(mod_zip),
        )

    port_file_0 = tmp_path / "servers" / "instance_0" / "mimi_port.txt"
    port_file_1 = tmp_path / "servers" / "instance_1" / "mimi_port.txt"
    assert port_file_0.read_text().strip() == "9200"
    assert port_file_1.read_text().strip() == "9201"


def test_game_port_passed_as_jvm_arg(tmp_path):
    """MindustryServer with game_port passes -Dmindustry.port to JVM."""
    server = MindustryServer(
        jar_path="fake.jar",
        data_dir=str(tmp_path),
        port=9000,
        game_port=6570,
    )
    # game_port should appear in _java_args
    assert any("6570" in arg for arg in server._java_args)
```

**Step 2: Run tests to verify they fail**

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/test_server_manager.py::test_start_n_servers_returns_n_instances rl/tests/test_server_manager.py::test_start_n_servers_writes_port_files rl/tests/test_server_manager.py::test_game_port_passed_as_jvm_arg -v
```

Expected: FAIL (ImportError on `start_n_servers`, `game_port` param missing)

### Step 3: Implement in `rl/server/manager.py`

Add `game_port` parameter to `MindustryServer.__init__`:

```python
def __init__(
    self,
    jar_path: str = "server-release.jar",
    data_dir: str = "rl/server_data",
    port: int = 9000,
    game_port: Optional[int] = None,
    java_args: Optional[list[str]] = None,
) -> None:
    self._jar_path = Path(jar_path)
    self._data_dir = Path(data_dir)
    self._port = port
    extra_args: list[str] = []
    if game_port is not None:
        extra_args.append(f"-Dmindustry.port={game_port}")
    self._java_args = (java_args or []) + extra_args
    self._proc: Optional[subprocess.Popen] = None
    self._ready = threading.Event()
    self._failed = threading.Event()
```

In `start()`, after `self._ready.wait()` resolves (i.e., right after the while-loop that waits for ready), add the fps hint:

```python
# Hint JVM to run at higher tick rate
def _send_fps_hint(self_ref=self) -> None:
    time.sleep(1.0)
    self_ref.send_stdin("fps 120")
threading.Thread(target=_send_fps_hint, daemon=True).start()
```

Add `start_n_servers` function at module bottom:

```python
def start_n_servers(
    n: int,
    base_tcp_port: int = 9000,
    base_game_port: int = 6567,
    base_data_dir: str = "rl/server_data",
    jar_path: str = "server-release.jar",
    mod_zip: str = "mimi-gateway-v1.0.4.zip",
) -> list["MindustryServer"]:
    """Start N Mindustry server instances in parallel and return them all ready."""
    import shutil as _shutil
    from pathlib import Path as _Path

    servers: list[MindustryServer] = []
    for i in range(n):
        inst_dir = _Path(base_data_dir) / f"instance_{i}"
        mods_dir = inst_dir / "config" / "mods"
        mods_dir.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(mod_zip, mods_dir / "mimi-gateway.zip")
        (inst_dir / "mimi_port.txt").write_text(str(base_tcp_port + i))
        server = MindustryServer(
            jar_path=jar_path,
            data_dir=str(inst_dir),
            port=base_tcp_port + i,
            game_port=base_game_port + i,
        )
        servers.append(server)

    errors: list[Exception] = []
    threads: list[threading.Thread] = []

    def _start(srv: MindustryServer) -> None:
        try:
            srv.start()
        except Exception as e:
            errors.append(e)

    for srv in servers:
        t = threading.Thread(target=_start, args=(srv,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if errors:
        for srv in servers:
            try:
                srv.stop()
            except Exception:
                pass
        raise RuntimeError(f"Failed to start {len(errors)} server(s): {errors}")

    return servers
```

### Step 4: Run all manager tests

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/test_server_manager.py -v
```

Expected: ALL PASS

### Step 5: Run full test suite

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/ -v
```

Expected: ALL PASS

### Step 6: Commit

```bash
git add rl/server/manager.py rl/tests/test_server_manager.py
git commit -m "feat(manager): add game_port JVM arg, fps 120 hint, start_n_servers() factory"
```

---

## Task 5: MindustryEnv — add tcp_port parameter

**Files:**
- Modify: `rl/env/mindustry_env.py`
- Test: `rl/tests/test_env.py`

### Step 1: Write failing test

Add to `rl/tests/test_env.py`:

```python
def test_env_accepts_tcp_port_parameter():
    """MindustryEnv should accept tcp_port and use it to connect."""
    env = MindustryEnv(tcp_port=9002, client=MagicMock())
    assert env._port == 9002
```

**Step 2: Run test to verify it fails**

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/test_env.py::test_env_accepts_tcp_port_parameter -v
```

Expected: FAIL (no `tcp_port` parameter)

### Step 3: Implement

In `rl/env/mindustry_env.py`, update `__init__` signature:

```python
def __init__(
    self,
    host: str = "localhost",
    port: int = 9000,
    tcp_port: Optional[int] = None,   # alias for port; takes precedence if given
    max_steps: int = 5000,
    client: Optional[MimiClient] = None,
    maps: Optional[list[str]] = None,
) -> None:
    super().__init__()
    self.observation_space = make_obs_space()
    self.action_space = make_action_space()
    self.max_steps = max_steps

    self._host = host
    self._port = tcp_port if tcp_port is not None else port
    self._client: Optional[MimiClient] = client
    self._maps: list[str] = maps if maps is not None else DEFAULT_TRAINING_MAPS
    self._map_index: int = 0

    self._prev_state: Optional[Dict[str, Any]] = None
    self._step_count: int = 0
```

### Step 4: Run env tests

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/test_env.py -v
```

Expected: ALL PASS

### Step 5: Run full test suite

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/ -v
```

Expected: ALL PASS

### Step 6: Commit

```bash
git add rl/env/mindustry_env.py rl/tests/test_env.py
git commit -m "feat(env): add tcp_port parameter as alias for port"
```

---

## Task 6: train.py — --n-envs flag + SubprocVecEnv multi-server path

**Files:**
- Modify: `rl/train.py`
- Test: `rl/tests/test_train.py` (create if not exists, or add to existing)

### Step 1: Write failing test

Create/add to `rl/tests/test_train.py`:

```python
"""Tests for train.py argument parsing and env factory."""
import pytest
from unittest.mock import patch, MagicMock
from rl.train import parse_args, _make_env_factory


def test_parse_args_default_n_envs():
    args = parse_args([])
    assert args.n_envs == 4


def test_parse_args_custom_n_envs():
    args = parse_args(["--n-envs", "2"])
    assert args.n_envs == 2


def test_make_env_factory_returns_callable():
    factory = _make_env_factory(host="localhost", tcp_port=9001, max_steps=100, maps=None)
    assert callable(factory)
```

**Step 2: Run tests to verify they fail**

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/test_train.py -v
```

Expected: FAIL (`n_envs` not in args, `_make_env_factory` doesn't exist)

### Step 3: Implement in `rl/train.py`

Add `--n-envs` argument in `parse_args()`. Update the function to accept a list for testability:

```python
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Mindustry A2C agent")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--n-envs", type=int, default=4, dest="n_envs",
                   help="Number of parallel environments (default: 4)")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--max-steps", type=int, default=5000, dest="max_steps")
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--n-steps", type=int, default=128, dest="n_steps")
    p.add_argument("--models-dir", default="rl/models")
    p.add_argument("--logs-dir", default="rl/logs")
    p.add_argument("--server-jar", default="server-release.jar", dest="server_jar")
    p.add_argument("--maps", default=None)
    p.add_argument("--no-server", action="store_true", dest="no_server")
    p.add_argument("--server-data-dir", default="rl/server_data", dest="server_data_dir")
    p.add_argument("--mod-zip", default="mimi-gateway-v1.0.4.zip", dest="mod_zip")
    return p.parse_args(argv)
```

Add `_make_env_factory` helper:

```python
def _make_env_factory(host: str, tcp_port: int, max_steps: int, maps):
    """Return a zero-arg callable that constructs one MindustryEnv (for SubprocVecEnv)."""
    def _factory():
        from rl.env.mindustry_env import MindustryEnv
        return MindustryEnv(host=host, tcp_port=tcp_port, max_steps=max_steps, maps=maps)
    return _factory
```

Update `main()` to use multi-server path when `n_envs > 1`:

```python
def main() -> None:
    args = parse_args()

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    maps = [m.strip() for m in args.maps.split(",")] if args.maps else None

    servers = []

    def _shutdown():
        for srv in servers:
            try:
                srv.stop()
            except Exception:
                pass

    def _on_sigterm(signum, frame):
        print("\nStopping servers (SIGTERM)...")
        _shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    if not args.no_server:
        from rl.server.manager import start_n_servers
        print(f"Starting {args.n_envs} Mindustry server instance(s)...")
        servers = start_n_servers(
            n=args.n_envs,
            base_tcp_port=args.port,
            base_data_dir=args.server_data_dir,
            jar_path=args.server_jar,
            mod_zip=args.mod_zip,
        )
        for i in range(args.n_envs):
            servers[i].send_stdin("host")
        print(f"All servers ready. Observe at localhost:6567 (instance 0)")

    try:
        if args.n_envs == 1:
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(
                MindustryEnv(host=args.host, tcp_port=args.port, max_steps=args.max_steps, maps=maps),
                filename=f"{args.logs_dir}/monitor",
            )
        else:
            from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
            env_fns = [
                _make_env_factory(
                    host=args.host,
                    tcp_port=args.port + i,
                    max_steps=args.max_steps,
                    maps=maps,
                )
                for i in range(args.n_envs)
            ]
            env = VecMonitor(SubprocVecEnv(env_fns), filename=f"{args.logs_dir}/monitor")

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

        callbacks = make_callbacks(save_path=args.models_dir)

        print(f"Starting A2C training for {args.timesteps:,} timesteps ({args.n_envs} envs)...")
        model.learn(total_timesteps=args.timesteps, callback=callbacks)

        final_path = f"{args.models_dir}/final_model"
        model.save(final_path)
        print(f"Training complete. Model saved to {final_path}.zip")

    finally:
        _shutdown()
```

Also remove the now-unneeded top-level `_install_mod` call in `main()` (handled by `start_n_servers`). Keep the `_install_mod` function itself for the `--no-server` / single-env backward compat path — but only call it when `n_envs == 1` and `not args.no_server`.

### Step 4: Run train tests

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/test_train.py -v
```

Expected: ALL PASS

### Step 5: Run full test suite

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/ -v
```

Expected: ALL PASS

### Step 6: Commit

```bash
git add rl/train.py rl/tests/test_train.py
git commit -m "feat(train): add --n-envs flag, SubprocVecEnv path, start_n_servers integration"
```

---

## Final verification

```bash
source rl/venv/bin/activate && python -m pytest rl/tests/ -v
```

Expected: ALL PASS (all original 48 + new tests added in Tasks 1, 4, 5, 6)

Check for any lint issues:

```bash
source rl/venv/bin/activate && python -m py_compile rl/rewards/multi_objective.py rl/server/manager.py rl/env/mindustry_env.py rl/train.py && echo "Syntax OK"
```

Expected: `Syntax OK`
