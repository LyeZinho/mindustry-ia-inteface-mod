# Mimi Gateway - Python Client Quick Start

Get your Python RL agent connected to Mindustry in 5 minutes.

---

## Installation

### Step 1: Install Python (if not already installed)

```bash
# Check if you have Python 3.6+
python --version
python3 --version

# macOS (with Homebrew)
brew install python3

# Linux (Ubuntu/Debian)
sudo apt-get install python3

# Windows: Download from https://www.python.org/downloads/
```

### Step 2: Download Mimi Client

Copy `test_mimi_client.py` from the mod folder to your project:

```bash
cp /path/to/mindustry-mod/test_mimi_client.py ./
```

Or create your own minimal client:

```python
import socket
import json

class MimiClient:
    def __init__(self, host='localhost', port=9000):
        self.socket = socket.socket()
        self.socket.connect((host, port))
    
    def get_state(self):
        """Receive game state"""
        return json.loads(self.socket.recv(4096).decode())
    
    def send(self, cmd):
        """Send command to game"""
        self.socket.send(f"{cmd}\n".encode())
    
    def close(self):
        self.socket.close()
```

---

## 5-Minute Quick Start

### 1. Start Mindustry with Mod Loaded

```bash
# Just start Mindustry normally
# Make sure mod is installed (see DEPLOYMENT_GUIDE.md)
# Start a campaign or sandbox game
# Keep the game window open
```

### 2. Create `hello_mimi.py`

```python
from test_mimi_client import MimiTestClient
import time

# Connect to Mindustry
client = MimiTestClient()
if not client.connect():
    print("Failed to connect")
    exit(1)

# Get game state
state = client.receive_state()
print(f"Wave: {state['wave']}")
print(f"Resources: Copper={state['resources']['copper']}")
print(f"Core HP: {state['core']['hp']:.0%}")

# Send a command
client.send_command("MSG;Hello from Python!")
time.sleep(0.5)

# Get next state
state = client.receive_state()
print(f"Updated: {state['time']} ticks")

client.disconnect()
```

### 3. Run It

```bash
python hello_mimi.py
```

**Expected output:**
```
HH:MM:SS [PASS] Connected to localhost:9000
HH:MM:SS [INFO] Received state #1 (tick=1234)
Wave: 3
Resources: Copper=450
Core HP: 95%
HH:MM:SS [PASS] MSG command sent and acknowledged
Updated: 1235 ticks
```

---

## Common Patterns

### Pattern 1: Monitor Game State

```python
from test_mimi_client import MimiTestClient
import time

client = MimiTestClient()
client.connect()

for i in range(60):  # 60 iterations
    state = client.receive_state()
    if state:
        print(f"[{i}] Wave={state['wave']}, Copper={state['resources']['copper']}")
    time.sleep(0.5)

client.disconnect()
```

### Pattern 2: Build When You Have Resources

```python
client = MimiTestClient()
client.connect()

while True:
    state = client.receive_state()
    
    copper = state['resources'].get('copper', 0)
    core_x = state['core']['x']
    
    # If we have enough copper, build a turret
    if copper > 50:
        client.send_command(f"BUILD;duo;{core_x + 3};{core_x + 3};0")
        print("Built DUO turret")
        time.sleep(1)  # Wait a bit before checking again

client.disconnect()
```

### Pattern 3: Defend Against Enemies

```python
client = MimiTestClient()
client.connect()

while True:
    state = client.receive_state()
    
    enemies = state.get('enemies', [])
    units = state.get('friendlyUnits', [])
    
    if enemies and units:
        # Attack nearest enemy
        enemy = enemies[0]
        for unit in units:
            cmd = f"ATTACK;{unit['id']};{int(enemy['x'])};{int(enemy['y'])}"
            client.send_command(cmd)

client.disconnect()
```

### Pattern 4: Spawn Units from Factory

```python
client = MimiTestClient()
client.connect()

unit_count = 0
target_units = 5

while unit_count < target_units:
    state = client.receive_state()
    
    # Count current units
    current_units = len(state.get('friendlyUnits', []))
    
    if current_units < target_units:
        # Find factory and spawn
        for building in state.get('buildings', []):
            if 'spawn' in building.get('block', '').lower():
                client.send_command(f"FACTORY;{building['x']};{building['y']};poly")
                print(f"Spawning unit {current_units + 1}")
                unit_count += 1
                break
    
    time.sleep(1)

client.disconnect()
```

---

## Command Reference

Quick lookup for sending commands:

| Goal | Command |
|------|---------|
| Build DUO turret at (15,20) | `client.send_command("BUILD;duo;15;20;0")` |
| Move unit 42 to (25,30) | `client.send_command("UNIT_MOVE;42;25;30")` |
| Attack with unit 42 at (30,35) | `client.send_command("ATTACK;42;30;35")` |
| Stop unit 42 | `client.send_command("STOP;42")` |
| Stop all units | `client.send_command("STOP")` |
| Repair building at (15,20) | `client.send_command("REPAIR;15;20")` |
| Delete building at (15,20) | `client.send_command("DELETE;15;20")` |
| Upgrade building at (15,20) | `client.send_command("UPGRADE;15;20")` |
| Send chat message | `client.send_command("MSG;Hello!")` |
| Spawn POLY from factory (10,12) | `client.send_command("FACTORY;10;12;poly")` |

See `API_DOCUMENTATION.md` for complete reference.

---

## State Object Reference

The state object returned by `receive_state()` has this structure:

```python
state = {
    'time': 1234,                    # Game ticks elapsed
    'tick': 1699999999999,           # Unix timestamp
    'wave': 5,                       # Current wave number
    'waveTime': 300,                 # Ticks until next wave
    
    'resources': {
        'copper': 450,
        'lead': 120,
        'graphite': 75,
        'titanium': 50,
        'thorium': 0,
        'scrap': 100,
        'coal': 60
    },
    
    'power': {
        'produced': 120.5,
        'consumed': 80.2,
        'stored': 500,
        'capacity': 1000
    },
    
    'core': {
        'hp': 0.95,           # 0.0-1.0 (0% to 100%)
        'x': 10,              # Tile position X
        'y': 20,              # Tile position Y
        'size': 3             # Building size
    },
    
    'player': {
        'x': 12,
        'y': 22
    },
    
    'friendlyUnits': [
        {'id': 2, 'type': 'poly', 'hp': 1.0, 'x': 10, 'y': 20, 'command': 'idle'},
        {'id': 3, 'type': 'mega', 'hp': 0.8, 'x': 15, 'y': 25, 'command': 'move'}
    ],
    
    'enemies': [
        {'id': 1, 'type': 'dagger', 'hp': 0.8, 'x': 15, 'y': 25, 'command': 'attack'}
    ],
    
    'buildings': [
        {'block': 'duo', 'team': 'sharded', 'x': 11, 'y': 21, 'hp': 1.0, 'rotation': 1},
        {'block': 'wall', 'team': 'sharded', 'x': 12, 'y': 21, 'hp': 0.95, 'rotation': 0}
    ],
    
    'grid': [
        {'x': 0, 'y': 0, 'block': 'core', 'floor': 'stone', 'team': 'sharded', 'hp': 0.95, 'rotation': 0},
        # ... 30x30 grid of tiles around core
    ]
}
```

**Common queries:**

```python
# Check if core is damaged
if state['core']['hp'] < 0.5:
    print("Core in danger!")

# Get unit count
friendly_count = len(state['friendlyUnits'])
enemy_count = len(state['enemies'])

# Get total resources
total_copper = state['resources']['copper']
total_power = state['power']['stored']

# Find nearest enemy
if state['enemies']:
    nearest = min(state['enemies'], 
                  key=lambda e: abs(e['x'] - state['core']['x']) + abs(e['y'] - state['core']['y']))
```

---

## Troubleshooting

### "Connection refused"

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Fix:**
1. Is Mindustry running? (Should have game window open)
2. Is mod loaded? (Press F1, look for `[Mimi Gateway] Servidor iniciado`)
3. Is it port 9000? (Check `scripts/main.js` - line should show `port = 9000`)

```bash
# Test connection manually
nc -zv localhost 9000
# Should output: Connection to localhost port 9000 [tcp/*] succeeded!
```

### "No state received" / Timeout

```
socket.timeout: timed out
```

**Fix:**
1. Is game paused? (Press F5 to unpause)
2. Is game at menu? (Start a campaign/sandbox game)
3. Is mod running? (Check F1 console continuously for messages)

### Command doesn't execute

**Check:**
- Do you have enough resources? (`state['resources']['copper'] > cost`)
- Are coordinates valid? (Within map bounds)
- Is unit ID correct? (Compare with `state['friendlyUnits'][n]['id']`)

```python
# Debug: Print full state
import json
print(json.dumps(state, indent=2))
```

### Connection drops after N commands

**Try:**
- Reduce command frequency (add `time.sleep(0.5)` between commands)
- Reduce grid size (edit `scripts/main.js`: `gridRadius: 10`)
- Increase update interval (edit `scripts/main.js`: `updateInterval: 20`)

---

## Next Steps

### Option A: RL Training

Integrate with Gymnasium:

```python
import gymnasium as gym
from mimi_client import MimiClient

class MindustryEnv(gym.Env):
    def __init__(self):
        self.client = MimiClient()
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(0, 1000, shape=(20,))
    
    def reset(self):
        state = self.client.get_state()
        obs = self._extract_features(state)
        return obs, {}
    
    def step(self, action):
        # Execute action
        self._execute_action(action)
        
        # Get reward
        state = self.client.get_state()
        reward = state['core']['hp']  # Reward staying alive
        done = state['core']['hp'] <= 0
        
        obs = self._extract_features(state)
        return obs, reward, done, False, {}
    
    def _extract_features(self, state):
        # Convert state to observation vector
        pass
    
    def _execute_action(self, action):
        # Convert action to command
        pass
```

See `API_DOCUMENTATION.md` → "Example 5: Gymnasium RL Environment"

### Option B: Strategic Logic

Build complex decision-making:

```python
class MindustryAgent:
    def __init__(self):
        self.client = MimiClient()
    
    def should_build_turret(self, state):
        # Turret positioning logic
        pass
    
    def should_spawn_unit(self, state):
        # Unit spawning strategy
        pass
    
    def process_enemies(self, state):
        # Attack prioritization
        pass
    
    def run(self):
        while True:
            state = self.client.get_state()
            
            if self.should_build_turret(state):
                self.build_turret(state)
            
            if self.should_spawn_unit(state):
                self.spawn_unit(state)
            
            self.process_enemies(state)
```

### Option C: Data Collection

Log game data for analysis:

```python
import csv
import time

client = MimiClient()
client.connect()

with open('game_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'wave', 'copper', 'core_hp', 'units', 'enemies'])
    
    for _ in range(1000):
        state = client.get_state()
        writer.writerow([
            state['time'],
            state['wave'],
            state['resources']['copper'],
            state['core']['hp'],
            len(state['friendlyUnits']),
            len(state['enemies'])
        ])

client.disconnect()
```

---

## Performance Tips

1. **Batch commands**: Send multiple commands per state update
2. **Cache state**: Reuse recent state instead of fetching every iteration
3. **Reduce updates**: Increase `updateInterval` if you don't need frequent updates
4. **Async I/O**: Use threading for long-running logic (see `API_DOCUMENTATION.md`)

---

## Full API Reference

See **API_DOCUMENTATION.md** for:
- Complete command syntax
- All 9 command types
- Advanced patterns
- Performance optimization
- Troubleshooting guide

---

## Testing

Run the full test suite:

```bash
python test_mimi_client.py --verbose
```

Should output:
```
[TEST] Connection to Mimi Gateway
[PASS] Connected to localhost:9000
...
Total: 10/10 tests passed
All tests passed! ✓
```

---

## Support

- **Installation issues**: See DEPLOYMENT_GUIDE.md
- **Command reference**: See API_DOCUMENTATION.md
- **Test failures**: See TEST_VERIFICATION_CHECKLIST.md
- **Implementation details**: See IMPLEMENTATION_SUMMARY.md
