# Mimi Gateway - Mindustry AI Interface Mod

A Mindustry mod that exposes the game's internal API via a TCP socket server, enabling RL agents and AI systems to perceive game state and execute actions in real-time.

**[📖 Guia em Português: GUIA_INSTALACAO_PT.md](./GUIA_INSTALACAO_PT.md)**

## Installation

1. Copy the `mimi-gateway` folder to your Mindustry mods directory:
   - **Windows**: `%appdata%/Mindustry/mods/`
   - **Linux**: `~/.local/share/Mindustry/mods/`
   - **macOS**: `~/Library/Application Support/Mindustry/mods/`

2. Launch Mindustry and verify the mod loads (check `F1` console for "[Mimi Gateway] Servidor iniciado na porta 9000")

3. Connect your Python/AI client to `localhost:9000`

For detailed step-by-step instructions, see **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** (English) or **[GUIA_INSTALACAO_PT.md](./GUIA_INSTALACAO_PT.md)** (Português).

## Architecture

```
Mimi v2 (Python/RL) <--TCP JSON--> Mimi Gateway (Mindustry Mod) <--API--> Game World
```

The mod runs as a daemon thread, maintaining a persistent TCP connection for bidirectional communication.

## Configuration

Edit `scripts/main.js` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `port` | 9000 | TCP server port |
| `updateInterval` | 10 | Ticks between state updates (~6Hz) |
| `gridRadius` | 15 | Tile radius for grid snapshot |
| `debug` | true | Enable verbose logging |

## Protocol: State Updates (Mod → Client)

The mod sends JSON state every `updateInterval` ticks:

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
    "thorium": 0
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

### State Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `tick` | int | Unix timestamp |
| `time` | int | Game ticks since start |
| `wave` | int | Current wave number |
| `waveTime` | int | Ticks until next wave |
| `resources` | object | Item counts in core (all items) |
| `power.produced` | float | Power production rate |
| `power.consumed` | float | Power consumption rate |
| `power.stored` | float | Current power storage |
| `power.capacity` | float | Total power capacity |
| `core.hp` | float | Core health (0.0-1.0) |
| `core.x` | int | Core tile X position |
| `core.y` | int | Core tile Y position |
| `player.x` | int | Player tile X position |
| `player.y` | int | Player tile Y position |
| `enemies[]` | array | Nearby enemy units |
| `friendlyUnits[]` | array | Nearby friendly units |
| `buildings[]` | array | Buildings in range |
| `grid[]` | array | Tile grid snapshot (31x31) |

## Protocol: Commands (Client → Mod)

Send commands as newline-terminated strings:

### BUILD - Construct a block
```
BUILD;block_name;x;y;rotation
```
Example: `BUILD;duo;15;20;0`

### UNIT_MOVE - Move a unit
```
UNIT_MOVE;unit_id;target_x;target_y
```
Example: `UNIT_MOVE;2;25;30`

### FACTORY - Spawn unit from factory
```
FACTORY;factory_x;factory_y;unit_type
```
Example: `FACTORY;10;12;poly`
Unit types: `poly`, `mega`, `glaive`, `reaper`, etc.

### ATTACK - Command unit to attack
```
ATTACK;unit_id;target_x;target_y
```
Example: `ATTACK;2;30;35`

### STOP - Stop unit movement
```
STOP;unit_id
```
Or stop all: `STOP`

### REPAIR - Repair building
```
REPAIR;x;y
```
Example: `REPAIR;15;20`

### DELETE - Deconstruct building
```
DELETE;x;y
```
Example: `DELETE;15;20`

### UPGRADE - Upgrade block
```
UPGRADE;x;y
```
Example: `UPGRADE;10;12`

### MSG - Send chat message
```
MSG;text
```
Example: `MSG;Building defense at sector 7`

## Python Client Example

```python
import socket
import json
import time

class MimiClient:
    def __init__(self, host='localhost', port=9000):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        print("Connected to Mimi Gateway")
    
    def receive_state(self):
        line = self.socket.readline()
        if line:
            return json.loads(line)
        return None
    
    def send_command(self, cmd):
        self.socket.send(f"{cmd}\n".encode())
    
    def build(self, block, x, y, rotation=0):
        self.send_command(f"BUILD;{block};{x};{y};{rotation}")
    
    def move_unit(self, unit_id, x, y):
        self.send_command(f"UNIT_MOVE;{unit_id};{x};{y}")
    
    def spawn_unit(self, factory_x, factory_y, unit_type="poly"):
        self.send_command(f"FACTORY;{factory_x};{factory_y};{unit_type}")
    
    def attack(self, unit_id, x, y):
        self.send_command(f"ATTACK;{unit_id};{x};{y}")
    
    def repair(self, x, y):
        self.send_command(f"REPAIR;{x};{y}")
    
    def delete(self, x, y):
        self.send_command(f"DELETE;{x};{y}")
    
    def message(self, text):
        self.send_command(f"MSG;{text}")

client = MimiClient()

while True:
    state = client.receive_state()
    if state:
        print(f"Wave: {state['wave']}, Core HP: {state['core']['hp']}")
        if state['resources'].get('copper', 0) > 100:
            client.build('duo', state['core']['x'] + 3, state['core']['y'])
    time.sleep(0.1)
```

## RL Training Usage

For headless server training:

1. Run Mindustry with `-server -no-graphics` or use `server.jar`
2. Enable fast-forward: `fps 60` in console
3. Connect your RL environment:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MindustryEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.client = MimiClient()
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(0, 1000, shape=(50,))
    
    def reset(self, seed=None):
        return np.random.rand(50), {}
    
    def step(self, action):
        commands = [("BUILD;duo", 1), ("BUILD;turret", 2), ("UNIT_MOVE", 3)]
        cmd, param = commands[action]
        self.client.send_command(f"{cmd};{param}")
        state = self.client.receive_state()
        reward = state['core']['hp'] if state else 0
        done = state is None or state['core']['hp'] <= 0
        return np.random.rand(50), reward, done, False, {}

gym.register('Mindustry-v0', MindustryEnv)
```

## Available Block Names

Common blocks:
- **Distribution**: `conveyor`, `router`, `junction`, `sorter`, `overflow-gate`
- **Production**: `drill`, `pump`
- **Power**: `battery`, `solar-panel`, `thermal-generator`, `power-node`
- **Defense**: `wall`, `door`, `duo`, `scatter`, `hail`, `lancer`, `swarmer`
- **Storage**: `core-sharded`, `vault`, `container`
- **Units**: `spawn`, `command-center`, `overdrive-projector`

## Troubleshooting

- **No connection on port 9000**: Verify mod is loaded (check F1 console)
- **Commands not working**: Ensure coordinates are valid, player has resources
- **Performance issues**: Reduce `gridRadius` or increase `updateInterval`

## License

Part of the Mimi v2 AI Agent project.