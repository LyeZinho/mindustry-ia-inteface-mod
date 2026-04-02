# Mimi Gateway - Deployment & Setup Guide

Complete step-by-step guide for installing, configuring, and validating the Mimi Gateway mod.

---

## Prerequisites

### System Requirements
- **Mindustry**: v146 or later (check: `Settings → About`)
- **OS**: Windows, Linux, or macOS
- **Python**: 3.6 or later (for test client)
- **Network**: Localhost access to TCP port 9000

### Installation Paths (Mindustry Mods Directory)
| OS | Path |
|----|------|
| Windows | `%APPDATA%\Mindustry\mods\` |
| Linux | `~/.local/share/Mindustry/mods/` |
| macOS | `~/Library/Application Support/Mindustry/mods/` |

---

## Phase 1: Mod Installation

### Step 1.1: Locate Mod Directory
```bash
# Linux/macOS
mkdir -p ~/.local/share/Mindustry/mods/

# Windows (using Command Prompt)
mkdir %APPDATA%\Mindustry\mods\
```

### Step 1.2: Copy Mod Files
Copy the following files to `mods/mimi-gateway/`:

```
~/.local/share/Mindustry/mods/mimi-gateway/
├── mod.hjson
├── scripts/
│   └── main.js
└── icon.png (optional)
```

**Using Git** (recommended):
```bash
cd ~/.local/share/Mindustry/mods/
git clone https://github.com/yourusername/mindustry-ia-interface-mod.git mimi-gateway
```

**Or manually**:
1. Download mod files as ZIP
2. Extract to `mimi-gateway/` folder
3. Verify structure: `ls ~/.local/share/Mindustry/mods/mimi-gateway/`

### Step 1.3: Verify Installation
1. Launch Mindustry
2. Go to **Settings → Mods → Modify**
3. Confirm **"mimi-gateway"** appears in the mod list
4. **Do NOT enable** (it loads automatically)
5. Check console for startup message

---

## Phase 2: Verify Mod Loads

### Step 2.1: Check Console
1. Start a campaign game
2. Press **F1** to open console
3. Look for one of these messages:

**SUCCESS** (mod loaded):
```
[Mimi Gateway] Servidor iniciado na porta 9000
```

**FAILURE** (mod not loaded):
- No message appears
- Or error message in red text

### Step 2.2: Troubleshooting Mod Load Failures

| Issue | Solution |
|-------|----------|
| **Mod not found in list** | Verify folder structure: `mods/mimi-gateway/mod.hjson` exists |
| **"Invalid mod"** error | Check `mod.hjson` syntax (use online JSON validator) |
| **"Java exception"** in console | Syntax error in `scripts/main.js` — check line numbers in error |
| **Port already in use** | Change `port = 9000` in `scripts/main.js` to `port = 9001` |

---

## Phase 3: Test Connection

### Step 3.1: Start Game with Mod
1. Launch Mindustry
2. Start **Campaign** or **Sandbox** mode
3. Wait for game to load (F1 console should show startup message)
4. Keep game window open

### Step 3.2: Run Test Client

#### Option A: Python Test Suite (Recommended)
```bash
# Download test_mimi_client.py to your computer
cd /path/to/mimi-gateway/
chmod +x test_mimi_client.py  # Linux/macOS only

# Run tests
python test_mimi_client.py --verbose
```

**Expected output:**
```
17:42:05 [TEST] Connection to Mimi Gateway
17:42:05 [PASS] Connected to localhost:9000
17:42:05 [INFO] Received state #1 (tick=1234)
17:42:05 [PASS] State validation passed (12 fields)

17:42:05 [TEST] BUILD command execution
17:42:05 [PASS] BUILD command sent and acknowledged
...

======================
Test Results Summary
======================
  ✓ PASS  CONNECTION
  ✓ PASS  BUILD
  ✓ PASS  UNIT_MOVE
  ...

Total: 10/10 tests passed
All tests passed! ✓
```

#### Option B: Manual Connection Test
```bash
# Linux/macOS
nc -zv localhost 9000

# Expected: Connection to localhost port 9000 [tcp/*] succeeded!
```

#### Option C: Python Interactive
```python
import socket
import json

sock = socket.socket()
sock.connect(('localhost', 9000))
state = json.loads(sock.recv(4096).decode())
print(state.keys())  # Should show: dict_keys(['time', 'tick', 'wave', ...])
```

---

## Phase 4: Send Test Commands

### Step 4.1: Basic Python Client
Create `test_commands.py`:

```python
import socket
import json
import time

def send_command(cmd):
    sock = socket.socket()
    sock.connect(('localhost', 9000))
    
    # Receive state
    state = json.loads(sock.recv(4096).decode())
    print(f"Game state: wave={state['wave']}, copper={state['resources'].get('copper')}")
    
    # Send command
    sock.send(f"{cmd}\n".encode())
    print(f"Sent: {cmd}")
    time.sleep(0.5)
    
    # Receive response
    try:
        state = json.loads(sock.recv(4096).decode())
        print(f"Response: tick={state['time']}")
    except:
        print("No response (command may succeed asynchronously)")
    
    sock.close()

# Test commands
send_command("BUILD;duo;15;20;0")  # Build DUO turret
time.sleep(1)
send_command("MSG;Hello from Mimi!")  # Send chat message
```

Run:
```bash
python test_commands.py
```

### Step 4.2: Command Reference

| Command | Effect | When it works |
|---------|--------|---------------|
| `BUILD;duo;15;20;0` | Build DUO turret at (15,20) | When player has copper & core is powered |
| `MSG;text` | Send chat message | Always |
| `STOP` | Stop all units | When friendly units exist |
| `ATTACK;1;25;30` | Attack position (25,30) with unit 1 | When unit 1 exists and enemies visible |

---

## Phase 5: Performance Validation

### Step 5.1: Connection Stability Test

Create `stress_test.py`:

```python
import socket
import json
import time
import threading

def receive_states(duration=30):
    """Receive states for N seconds and measure throughput"""
    sock = socket.socket()
    sock.connect(('localhost', 9000))
    
    count = 0
    start = time.time()
    errors = 0
    
    while time.time() - start < duration:
        try:
            state = json.loads(sock.recv(4096).decode())
            count += 1
        except Exception as e:
            errors += 1
    
    elapsed = time.time() - start
    sock.close()
    
    print(f"States received: {count}")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Throughput: {count/elapsed:.1f} states/sec")
    print(f"Errors: {errors}")

# Run for 30 seconds
receive_states(30)
```

**Expected performance:**
- Throughput: **5-10 states/sec** (normal)
- Errors: **0** (no dropped connections)
- Latency: **<50ms** per state update

If throughput is **<2 states/sec**, check:
1. Is Mindustry running?
2. Is F1 console showing the startup message?
3. Are other processes using port 9000?

---

## Phase 6: Validate All Commands

### Step 6.1: Run Full Test Suite

```bash
# Terminal 1: Start Mindustry with mod loaded (keep running)
cd ~/.local/share/Mindustry/mods/mimi-gateway/

# Terminal 2: Run tests
python test_mimi_client.py --verbose

# Expected output
✓ CONNECTION  - Connects and receives state
✓ BUILD       - Can construct buildings
✓ UNIT_MOVE   - Can move friendly units
✓ ATTACK      - Can command attacks
✓ STOP        - Can halt units
✓ REPAIR      - Can repair buildings
✓ DELETE      - Can deconstruct buildings
✓ UPGRADE     - Can upgrade blocks
✓ MSG         - Can send chat messages
✓ FACTORY     - Can spawn units
```

### Step 6.2: Manual Command Validation

For each command, run and verify game state changed:

```python
import socket, json, time

sock = socket.socket()
sock.connect(('localhost', 9000))

# Get initial state
state1 = json.loads(sock.recv(4096).decode())
core_copper_before = state1['resources']['copper']

# Send build command
sock.send("BUILD;duo;15;20;0\n".encode())
time.sleep(2)

# Check if copper decreased (means building consumed resources)
state2 = json.loads(sock.recv(4096).decode())
copper_after = state2['resources']['copper']

if copper_after < core_copper_before:
    print("✓ BUILD command verified")
else:
    print("✗ BUILD may not have executed")

sock.close()
```

---

## Phase 7: Integration Testing (RL Training)

### Step 7.1: Headless Server Setup

For training without UI:

```bash
# Linux/macOS
java -jar server.jar -server -no-graphics

# Windows
java -jar server.jar -server -no-graphics
```

Then connect Python RL agent:

```python
import socket
import json

class MindustryEnv:
    def __init__(self):
        self.sock = socket.socket()
        self.sock.connect(('localhost', 9000))
    
    def get_state(self):
        return json.loads(self.sock.recv(8192).decode())
    
    def send_command(self, cmd):
        self.sock.send(f"{cmd}\n".encode())

env = MindustryEnv()
state = env.get_state()
print(f"Wave: {state['wave']}, Resources: {state['resources']}")
```

### Step 7.2: Gymnasium Integration

```python
import gymnasium as gym
from mindustry_env import MindustryEnv

gym.register('Mindustry-v0', MindustryEnv)
env = gym.make('Mindustry-v0')
```

---

## Troubleshooting

### Connection Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `Connection refused` | Mod not loaded | Check F1 console for startup message |
| `Connection timeout` | Game crashed or hung | Restart Mindustry |
| `Port 9000 in use` | Another process using port | Change port in `scripts/main.js` |
| `Invalid JSON` | Corruption in state transmission | Restart game and reconnect |

### Command Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| BUILD fails silently | Not enough resources or power | Check `state['resources']` and `state['power']['stored']` |
| UNIT_MOVE doesn't work | Invalid unit ID | Verify unit ID from `state['friendlyUnits']` |
| ATTACK command ignored | No enemy in range | Check `state['enemies']` array |
| Factory not spawning | No factory at coordinates | Verify coordinates from `state['buildings']` |

### Performance Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Low state update frequency | Game too busy | Reduce `gridRadius` in `scripts/main.js` |
| Network lag | Slow connection | Increase `updateInterval` from 10 to 20 |
| CPU usage high | Grid snapshot too large | Reduce `gridRadius` from 15 to 8 |

---

## Validation Checklist

- [ ] Mod folder created at correct path
- [ ] `mod.hjson` exists and is valid JSON
- [ ] `scripts/main.js` exists and has no syntax errors
- [ ] Mindustry loads without crashing (F1 shows startup message)
- [ ] Python test client connects successfully
- [ ] At least 5 states received within 10 seconds
- [ ] At least 1 test command executes without errors
- [ ] No connection timeouts during 30-second stress test
- [ ] Full test suite shows ≥8/10 tests passing
- [ ] No errors or exceptions in Mindustry console (F1)

---

## Support & Next Steps

### If All Tests Pass ✓
Your Mimi Gateway is ready for:
- RL training environments
- AI agent control
- Automated testing
- Custom client development

### If Tests Fail ✗
1. Check the **Troubleshooting** section above
2. Verify **Validation Checklist** items
3. Review mod logs: `~/.mindustry/logs/` 
4. Restart from **Phase 1: Mod Installation**

### Next: RL Training
See `API_DOCUMENTATION.md` for:
- Python client API reference
- Advanced usage patterns
- Gymnasium environment setup
- Performance optimization

---

## Command Syntax Reference

Quick lookup for all 9 commands:

```
BUILD;block;x;y;rotation        # Construct block
UNIT_MOVE;id;x;y                # Move unit to position
ATTACK;id;x;y                   # Attack position with unit
STOP;id                          # Stop unit (or STOP for all)
REPAIR;x;y                       # Repair building
DELETE;x;y                       # Deconstruct building
UPGRADE;x;y                      # Upgrade block
MSG;text                         # Send chat message
FACTORY;x;y;type                # Spawn unit from factory
```

See `README.md` and `API_DOCUMENTATION.md` for full details.
