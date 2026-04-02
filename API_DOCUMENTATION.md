# Mimi Gateway - Comprehensive API Documentation

## Table of Contents

1. [Protocol Overview](#protocol-overview)
2. [State Updates](#state-updates)
3. [Commands](#commands)
4. [Python Client Examples](#python-client-examples)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)
7. [Performance Guide](#performance-guide)

---

## Protocol Overview

### Connection

**Address**: `localhost:9000` (default)  
**Protocol**: TCP with newline-delimited JSON  
**Encoding**: UTF-8  
**Max Clients**: 1 (per game instance)

### Message Format

**State Updates (Mod → Client)**:
```
<JSON_STATE_OBJECT>\n
```

**Commands (Client → Mod)**:
```
<COMMAND_STRING>\n
```

### Connection Lifecycle

```
Client connects → Server accepts → Server sends state updates → Client sends commands
                                 ↓
                         Reconnect on error (up to 5 attempts)
```

---

## State Updates

### Structure

State updates are JSON objects sent every 10 game ticks (~166ms).

```json
{
  "tick": 1699999999999,
  "time": 1234,
  "wave": 5,
  "waveTime": 300,
  "resources": {
    "copper": 450,
    "lead": 120,
    "graphite": 75,
    "titanium": 50,
    "thorium": 0,
    "scrap": 100,
    "coal": 60
  },
  "power": {
    "produced": 120.5,
    "consumed": 80.2,
    "stored": 500,
    "capacity": 1000
  },
  "core": {
    "hp": 0.95,
    "x": 10,
    "y": 20,
    "size": 3
  },
  "player": {
    "x": 12,
    "y": 22
  },
  "enemies": [
    {
      "id": 1,
      "type": "dagger",
      "hp": 0.8,
      "x": 15,
      "y": 25,
      "command": "attack"
    }
  ],
  "friendlyUnits": [
    {
      "id": 2,
      "type": "poly",
      "hp": 1.0,
      "x": 10,
      "y": 20,
      "command": "idle"
    }
  ],
  "buildings": [
    {
      "block": "duo",
      "team": "sharded",
      "x": 11,
      "y": 21,
      "hp": 1.0,
      "rotation": 1
    }
  ],
  "grid": [
    {
      "x": 0,
      "y": 0,
      "block": "core",
      "floor": "stone",
      "team": "sharded",
      "hp": 0.95,
      "rotation": 0
    }
  ]
}
```

### Field Reference

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `tick` | number | unix timestamp | Current system time in milliseconds |
| `time` | number | >= 0 | Game ticks since game start |
| `wave` | number | >= 1 | Current wave number |
| `waveTime` | number | >= 0 | Ticks remaining until next wave |
| `resources[item]` | number | >= 0 | Amount of item in core storage |
| `power.produced` | float | >= 0 | Power generation rate (units/tick) |
| `power.consumed` | float | >= 0 | Power consumption rate (units/tick) |
| `power.stored` | float | >= 0 | Current power in batteries (units) |
| `power.capacity` | float | > 0 | Maximum power storage (units) |
| `core.hp` | float | 0.0-1.0 | Core health as percentage |
| `core.x`, `core.y` | number | tile coords | Core tile position |
| `core.size` | number | 1-5 | Core size in tiles |
| `player.x`, `player.y` | number | tile coords | Player/camera position |
| `enemies[].id` | number | > 0 | Unique unit ID |
| `enemies[].type` | string | see list | Unit type name |
| `enemies[].hp` | float | 0.0-1.0 | Unit health as percentage |
| `enemies[].x`, `enemies[].y` | float | world coords | Unit position |
| `enemies[].command` | string | see list | Current unit command |

### Example: Processing State

```python
def process_state(state):
    """Extract useful information from state"""
    
    # Check if we're under attack
    enemies = state.get('enemies', [])
    if enemies:
        nearest_enemy = min(enemies, key=lambda e: (
            (e['x'] - state['core']['x'])**2 + 
            (e['y'] - state['core']['y'])**2
        ))
        print(f"Enemy nearby: {nearest_enemy['type']} at ({nearest_enemy['x']}, {nearest_enemy['y']})")
    
    # Check resources
    copper = state['resources'].get('copper', 0)
    lead = state['resources'].get('lead', 0)
    
    # Check power status
    power_balance = state['power']['produced'] - state['power']['consumed']
    power_percent = 100 * state['power']['stored'] / state['power']['capacity']
    
    # Check unit status
    units = state.get('friendlyUnits', [])
    print(f"Friendly units: {len(units)}")
    for unit in units:
        print(f"  - {unit['type']} (HP: {unit['hp']:.0%})")
    
    # Get grid info around core
    grid = state.get('grid', [])
    for tile in grid:
        if tile['block'] != 'air':
            print(f"Block at ({tile['x']}, {tile['y']}): {tile['block']}")
```

---

## Commands

### Command Syntax

All commands use semicolon-separated format:

```
COMMAND;arg1;arg2;arg3
```

### 1. BUILD - Construct a Block

**Syntax**:
```
BUILD;block_name;x;y;rotation
```

**Parameters**:
- `block_name` (string): Name of block to build (e.g., "duo", "scatter", "wall")
- `x` (number): Tile X coordinate
- `y` (number): Tile Y coordinate  
- `rotation` (number, optional): Rotation 0-3 (default: 0)

**Requirements**:
- Sufficient resources for block
- Valid tile location
- Adjacent to existing structure or core

**Example**:
```python
client.send_command("BUILD;duo;15;20;0")      # Build duo turret
client.send_command("BUILD;wall;10;10;0")     # Build wall
client.send_command("BUILD;router;12;12;0")   # Build router
```

**Response**: None (check logs or state updates)

**Error Cases**:
- Insufficient resources: Command ignored
- Invalid location: Command ignored  
- Invalid block name: Logged error

---

### 2. DELETE - Deconstruct a Building

**Syntax**:
```
DELETE;x;y
```

**Parameters**:
- `x` (number): Tile X coordinate
- `y` (number): Tile Y coordinate

**Requirements**:
- Building must exist at location
- Must be team's building

**Example**:
```python
client.send_command("DELETE;15;20")     # Remove structure
```

**Response**: Structure deconstructed

**Error Cases**:
- No building at location: Logged warning
- Foreign building: Command ignored

---

### 3. REPAIR - Repair a Damaged Building

**Syntax**:
```
REPAIR;x;y
```

**Parameters**:
- `x` (number): Tile X coordinate
- `y` (number): Tile Y coordinate

**Requirements**:
- Building must exist
- Building must be damaged (HP < max)

**Example**:
```python
client.send_command("REPAIR;15;20")     # Repair damaged building
```

**Response**: Building health increased by 50% of max

**Error Cases**:
- No building: Logged error
- Building at full health: Command ignored

---

### 4. UPGRADE - Upgrade a Block

**Syntax**:
```
UPGRADE;x;y
```

**Parameters**:
- `x` (number): Tile X coordinate
- `y` (number): Tile Y coordinate

**Requirements**:
- Building must exist
- Block must have upgrade path defined
- Sufficient resources for new block

**Upgrade Paths**:
- duo → scatter
- scatter → hail
- wall → large-wall
- battery → large-battery
- etc.

**Example**:
```python
client.send_command("UPGRADE;15;20")    # Upgrade building
```

**Response**: Block replaced with upgraded version

**Error Cases**:
- No upgrade available: Logged message
- Insufficient resources: Command ignored

---

### 5. UNIT_MOVE - Move Unit to Position

**Syntax**:
```
UNIT_MOVE;unit_id;target_x;target_y
```

**Parameters**:
- `unit_id` (number): Unit ID from state
- `target_x` (number): Target tile X coordinate
- `target_y` (number): Target tile Y coordinate

**Requirements**:
- Unit must exist
- Target must be valid tile

**Example**:
```python
# From state, get unit ID 42
client.send_command("UNIT_MOVE;42;25;30")   # Move to position
```

**Response**: Unit receives move command and heads toward target

**Error Cases**:
- Unit not found: Logged error
- Invalid target: Command ignored

**Notes**:
- Unit will navigate around obstacles
- Movement takes time (not instant)
- Multiple move commands override previous ones

---

### 6. ATTACK - Order Unit to Attack

**Syntax**:
```
ATTACK;unit_id;target_x;target_y
```

**Parameters**:
- `unit_id` (number): Unit ID
- `target_x` (number): Target tile X
- `target_y` (number): Target tile Y

**Requirements**:
- Unit must exist
- Target must have building/enemy at location

**Example**:
```python
# Attack enemy building
client.send_command("ATTACK;42;30;35")
```

**Response**: Unit targets specified location/building

**Error Cases**:
- Unit not found: Logged error
- Invalid target: Command may be ignored

**Notes**:
- Unit must be in range to attack
- Different units have different ranges
- Unit will path toward target

---

### 7. STOP - Stop Unit Movement

**Syntax**:
```
STOP;unit_id         # Stop specific unit
STOP                 # Stop all units
```

**Parameters**:
- `unit_id` (number, optional): Unit ID to stop

**Example**:
```python
client.send_command("STOP;42")           # Stop unit 42
client.send_command("STOP")              # Stop all units
```

**Response**: Unit clears movement/attack command

**Error Cases**:
- Unit not found: Logged warning

---

### 8. FACTORY - Spawn Unit from Factory

**Syntax**:
```
FACTORY;factory_x;factory_y;unit_type
```

**Parameters**:
- `factory_x` (number): Factory building tile X
- `factory_y` (number): Factory building tile Y
- `unit_type` (string): Type of unit to spawn

**Supported Unit Types**:
- `poly` - Poly (basic)
- `mega` - Mega (heavy)
- `glaive` - Glaive (melee)
- `reaper` - Reaper (flying)
- `flare` - Flare (flying)
- `ripple` - Ripple (tank)
- `wraith` - Wraith (flying)
- `titan` - Titan (ground)
- `fortress` - Fortress (heavy)

**Requirements**:
- Factory must exist at location
- Factory must be team's building
- Sufficient power and items in factory

**Example**:
```python
client.send_command("FACTORY;10;12;poly")       # Spawn poly unit
client.send_command("FACTORY;10;12;mega")       # Spawn mega unit
```

**Response**: Unit spawned at factory location

**Error Cases**:
- No factory: Logged error
- Factory insufficient power: Command queued or ignored
- Invalid unit type: Logged error

---

### 9. MSG - Send Chat Message

**Syntax**:
```
MSG;text
```

**Parameters**:
- `text` (string): Message content

**Example**:
```python
client.send_command("MSG;Hello from Mimi!")
client.send_command("MSG;Building defense at sector 7")
```

**Response**: Message appears in game chat

**Notes**:
- Prefixed with `[cyan][Mimi v2]:[]`
- Can include semicolons in message (will be joined back)
- Supports Mindustry color codes if desired

---

## Python Client Examples

### Example 1: Basic Connection and State Monitoring

```python
import socket
import json
import time

class MimiClient:
    def __init__(self, host='localhost', port=9000):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
    
    def receive_state(self):
        """Receive one state update"""
        line = self.socket.recv(4096).decode('utf-8').strip()
        return json.loads(line) if line else None
    
    def send_command(self, cmd):
        """Send a command"""
        self.socket.send(f"{cmd}\n".encode('utf-8'))
    
    def close(self):
        self.socket.close()

# Usage
client = MimiClient()

for i in range(10):
    state = client.receive_state()
    print(f"Wave {state['wave']}, Core HP: {state['core']['hp']:.0%}")
    time.sleep(1)

client.close()
```

### Example 2: Build Turrets Automatically

```python
client = MimiClient()

while True:
    state = client.receive_state()
    
    copper = state['resources'].get('copper', 0)
    core_x = state['core']['x']
    core_y = state['core']['y']
    
    # Build towers if we have copper
    if copper > 50:
        # Build duo at 3 tiles away from core
        client.send_command(f"BUILD;duo;{core_x + 3};{core_y};0")
        print("Built duo turret")
    
    time.sleep(0.5)
```

### Example 3: Defend Against Enemies

```python
client = MimiClient()
unit_ids = set()

while True:
    state = client.receive_state()
    
    enemies = state.get('enemies', [])
    units = state.get('friendlyUnits', [])
    
    # If enemies nearby, tell units to attack
    if enemies and units:
        enemy = enemies[0]  # Attack nearest
        for unit in units:
            client.send_command(f"ATTACK;{unit['id']};{int(enemy['x'])};{int(enemy['y'])}")
    
    # Repair core if damaged
    if state['core']['hp'] < 0.5:
        print("Core damaged! Consider retreating or repairing.")
    
    time.sleep(0.5)
```

### Example 4: Factory Unit Management

```python
client = MimiClient()

while True:
    state = client.receive_state()
    
    units = state.get('friendlyUnits', [])
    
    # Keep unit count at target
    target_units = 5
    
    if len(units) < target_units:
        # Find factory and spawn unit
        for building in state.get('buildings', []):
            if 'factory' in building.get('block', ''):
                client.send_command(f"FACTORY;{building['x']};{building['y']};poly")
                break
    
    # Move units toward enemies
    enemies = state.get('enemies', [])
    if enemies:
        target_enemy = enemies[0]
        for unit in units:
            client.send_command(f"UNIT_MOVE;{unit['id']};{int(target_enemy['x'])};{int(target_enemy['y'])}")
    
    time.sleep(1)
```

### Example 5: Gymnasium RL Environment

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MindustryEnv(gym.Env):
    """OpenAI Gym environment for Mindustry"""
    
    metadata = {'render_modes': ['ansi']}
    
    def __init__(self):
        super().__init__()
        from mimi_client import MimiClient
        self.client = MimiClient()
        self.last_hp = 1.0
        
        # Action space: 10 actions (9 commands + no-op)
        self.action_space = spaces.Discrete(10)
        
        # Observation space: scaled features
        self.observation_space = spaces.Box(
            low=0, high=1000,
            shape=(20,),  # Resources, power, units, enemies, etc.
            dtype=np.float32
        )
    
    def _get_observation(self):
        """Extract features from game state"""
        state = self.client.receive_state()
        
        obs = np.array([
            state['wave'],
            state['core']['hp'],
            state['resources'].get('copper', 0) / 1000,
            state['resources'].get('lead', 0) / 1000,
            state['power']['produced'],
            state['power']['consumed'],
            state['power']['stored'] / state['power']['capacity'],
            len(state.get('friendlyUnits', [])),
            len(state.get('enemies', [])),
            # ... more features
        ], dtype=np.float32)
        
        return obs[:20]  # Pad/truncate to 20
    
    def reset(self, seed=None):
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return reward"""
        
        # Map action to command
        commands = [
            "MSG;RL Action 0",
            "MSG;RL Action 1",
            # ... etc
        ]
        
        if action < len(commands):
            self.client.send_command(commands[action])
        
        # Get new state
        obs = self._get_observation()
        
        # Calculate reward (higher HP is better)
        current_hp = obs[1]
        reward = (current_hp - self.last_hp) * 100  # Penalize damage
        self.last_hp = current_hp
        
        # Episode done if core destroyed
        done = current_hp <= 0
        
        return obs, reward, done, False, {}
    
    def close(self):
        self.client.close()

# Usage
env = MindustryEnv()
obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()  # Random policy
    obs, reward, done, _, info = env.step(action)
    
    if done:
        break

env.close()
```

---

## Advanced Usage

### Async Command Queue

```python
import threading
import queue

class AsyncMimiClient(MimiClient):
    def __init__(self, host='localhost', port=9000):
        super().__init__(host, port)
        self.command_queue = queue.Queue()
        self.running = True
        
        # Start sender thread
        sender_thread = threading.Thread(target=self._send_loop, daemon=True)
        sender_thread.start()
    
    def _send_loop(self):
        """Background thread that sends queued commands"""
        while self.running:
            try:
                cmd = self.command_queue.get(timeout=0.1)
                self.send_command(cmd)
            except queue.Empty:
                continue
    
    def queue_command(self, cmd):
        """Queue command for sending (non-blocking)"""
        self.command_queue.put(cmd)
```

### State Caching

```python
class CachedMimiClient(MimiClient):
    def __init__(self):
        super().__init__()
        self.cached_state = None
        self.state_updated_at = 0
    
    def get_state(self, use_cache=True, max_age=0.1):
        """Get state with optional caching"""
        import time
        
        if use_cache and self.cached_state:
            age = time.time() - self.state_updated_at
            if age < max_age:
                return self.cached_state
        
        self.cached_state = self.receive_state()
        self.state_updated_at = time.time()
        return self.cached_state
```

---

## Troubleshooting

### Common Issues

#### Issue: Connection refused
```
Error: [Errno 111] Connection refused
```

**Solution**:
1. Verify Mindustry is running with mod loaded
2. Check F1 console for `[Mimi Gateway] Servidor iniciado`
3. Verify mod is in correct directory
4. Try `telnet localhost 9000` to test connection

#### Issue: No state received
```
timeout: timed out
```

**Solution**:
1. Ensure game is loaded (not on menu)
2. Check network connectivity
3. Verify no firewall blocking port 9000
4. Check mod debug logs in F1 console

#### Issue: Command not executing
```
Command received but nothing happens in game
```

**Solution**:
1. Check coordinates are valid for map
2. Verify resources available for BUILD
3. Check unit IDs match current units
4. Enable debug mode to see log messages

#### Issue: Connection drops after N commands
```
Socket connection closed unexpectedly
```

**Solution**:
1. Check for stack trace in F1 console
2. Look for out-of-memory errors
3. Try reducing gridRadius if handling too much data
4. Check for rapid command spam (> 100/sec)

### Debug Mode

Enable verbose logging in `scripts/main.js`:

```javascript
const config = {
    port: 9000,
    updateInterval: 10,
    gridRadius: 15,
    debug: true  // ← Set to true
};
```

**Debug Output**:
- Command receipt/processing
- Error stack traces
- Connection state changes
- State snapshot headers

---

## Performance Guide

### Optimization Tips

1. **Reduce Update Frequency**:
   ```javascript
   updateInterval: 20  // Update every 20 ticks (~333ms)
   ```

2. **Reduce Grid Size**:
   ```javascript
   gridRadius: 10  // Smaller radius = less data
   ```

3. **Batch Commands**:
   - Send multiple commands per state update
   - Avoid rate-limiting to 1 command per update

4. **Async Processing**:
   ```python
   # Use threading for long-running logic
   def process_state_async(state):
       threading.Thread(target=analyze_state, args=(state,)).start()
   ```

### Monitoring Performance

```python
import time

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.times = []
        self.window_size = window_size
    
    def record(self, latency):
        self.times.append(latency)
        if len(self.times) > self.window_size:
            self.times.pop(0)
    
    def get_stats(self):
        import statistics
        return {
            'avg': statistics.mean(self.times),
            'min': min(self.times),
            'max': max(self.times),
            'stdev': statistics.stdev(self.times) if len(self.times) > 1 else 0
        }

# Usage
monitor = PerformanceMonitor()

while True:
    start = time.time()
    state = client.receive_state()
    latency = time.time() - start
    
    monitor.record(latency)
    
    if len(monitor.times) % 100 == 0:
        stats = monitor.get_stats()
        print(f"Latency: avg={stats['avg']*1000:.1f}ms, max={stats['max']*1000:.1f}ms")
```

---

## Additional Resources

- **README.md** - Quick start guide
- **IMPLEMENTATION_SUMMARY.md** - Technical architecture
- **TEST_SUITE.md** - Comprehensive test procedures
- **mindustry_reference.md** - Mindustry modding API reference

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-04-02 | Initial release - all 9 commands implemented |

---

## License

Part of the Mimi v2 AI Agent project.
