# Mimi Gateway - Test Verification Checklist

Complete validation matrix for all commands and features. Use this checklist to verify the mod works correctly in your environment.

**Duration**: ~15 minutes  
**Requirements**: Mindustry running, mod loaded, Python test client

---

## Pre-Test Verification

Run this before starting command tests:

- [ ] **Mindustry Running**: Game window visible
- [ ] **Mod Loaded**: F1 console shows `[Mimi Gateway] Servidor iniciado na porta 9000`
- [ ] **Campaign/Sandbox Started**: Actively playing or in pause mode
- [ ] **Python 3.6+**: `python --version` shows v3.6 or later
- [ ] **Test Client Ready**: `test_mimi_client.py` in mod folder
- [ ] **Port Accessible**: `python test_mimi_client.py --verbose` connects without timeout

**If any fail**: See DEPLOYMENT_GUIDE.md → Troubleshooting

---

## Connection Test

### TEST 1.1: Initial Connection

| Step | Action | Expected | Status |
|------|--------|----------|--------|
| 1 | Run: `python test_mimi_client.py --verbose` | "Connected to localhost:9000" ✓ | ☐ |
| 2 | Check output for state reception | State #1 received with ≥10 fields | ☐ |
| 3 | Verify JSON parsing | No JSON decode errors | ☐ |
| 4 | Check game still running | Mindustry window active, not frozen | ☐ |

**PASS CRITERIA**: All steps succeed, no errors  
**Expected Duration**: 1-2 seconds

---

### TEST 1.2: State Structure Validation

| Field | Type | Expected Value | Status |
|-------|------|-----------------|--------|
| `time` | int | Game tick (e.g., 1234) | ☐ |
| `tick` | int | Unix timestamp | ☐ |
| `wave` | int | Wave number ≥0 | ☐ |
| `waveTime` | int | Time until next wave | ☐ |
| `resources` | dict | Keys: copper, lead, etc. | ☐ |
| `power` | dict | Keys: produced, consumed, stored | ☐ |
| `core` | dict | Keys: hp, x, y, size | ☐ |
| `buildings` | list | Array of building objects | ☐ |
| `grid` | list | Array of tile objects (31×31) | ☐ |
| `friendlyUnits` | list | Array of allied units | ☐ |
| `enemies` | list | Array of enemy units (may be empty) | ☐ |

**PASS CRITERIA**: All required fields present and correct type  
**Acceptable**: `enemies` and `friendlyUnits` can be empty arrays in early game

---

## Command Tests

### TEST 2.1: BUILD Command

**Objective**: Construct a turret at specified coordinates

**Setup**:
```python
import json
state = json.loads(sock.recv(4096).decode())
core_x = state['core']['x']  # Get core position
build_x = core_x + 3
build_y = state['core']['y'] + 3
resources_before = state['resources']['copper']
```

| Step | Command | Expected | Check | Status |
|------|---------|----------|-------|--------|
| 1 | `BUILD;duo;{build_x};{build_y};0` | Command sent without error | Console: no error | ☐ |
| 2 | Wait 2 seconds | Mod processes build request | Game state updated | ☐ |
| 3 | Receive next state | New state with tick++  | `state['time']` > before | ☐ |
| 4 | Check resources | Copper decreased by ~30 | `copper_after < copper_before` | ☐ |
| 5 | Verify in-game | DUO turret visible on map | Visual confirmation | ☐ |

**PASS CRITERIA**: 
- ✓ Command sent successfully (no exception)
- ✓ New state received (no timeout)
- ✓ Resources decreased (copper/energy consumed)
- ✓ Building visible in-game (optional but validates)

**FAIL CASES**:
- ❌ `Connection refused` → Mod not loaded
- ❌ `Invalid coordinates` → X/Y out of bounds (too far from core)
- ❌ `Insufficient resources` → Not enough copper in core
- ❌ `No power` → Grid not powered

**Expected Duration**: 3-5 seconds

---

### TEST 2.2: MSG Command

**Objective**: Send chat message

| Step | Command | Expected | Status |
|------|---------|----------|--------|
| 1 | `MSG;Test message from Mimi` | Command sent | ☐ |
| 2 | Check game | Message appears in chat or system log | ☐ |
| 3 | No crash | Game continues running normally | ☐ |

**PASS CRITERIA**: 
- ✓ Message sent without error
- ✓ Game acknowledges (chat or console)
- ✓ No game freeze/crash

**Expected Duration**: 1-2 seconds

---

### TEST 2.3: UNIT_MOVE Command

**Objective**: Move a friendly unit to new coordinates

**Setup**:
```python
state = json.loads(sock.recv(4096).decode())
if not state.get('friendlyUnits'):
    print("SKIP: No friendly units available")
else:
    unit = state['friendlyUnits'][0]
    unit_id = unit['id']
    move_x = unit['x'] + 5
    move_y = unit['y'] + 5
```

| Step | Command | Expected | Status |
|------|---------|----------|--------|
| 1 | `UNIT_MOVE;{unit_id};{move_x};{move_y}` | Command sent | ☐ |
| 2 | Wait 1 second | Unit starts moving | ☐ |
| 3 | Observe unit | Unit moves to target or nearby | Visual check | ☐ |
| 4 | Receive state | New state shows unit at new position | ☐ |

**PASS CRITERIA**:
- ✓ Command sent without error
- ✓ Unit visibly moves (check in-game)
- ✓ New state received with updated coordinates

**SKIP CONDITION**: No friendly units in `state['friendlyUnits']`

**Expected Duration**: 2-4 seconds

---

### TEST 2.4: ATTACK Command

**Objective**: Command unit to attack enemy

**Setup**:
```python
state = json.loads(sock.recv(4096).decode())
friendly = state.get('friendlyUnits', [])
enemies = state.get('enemies', [])
if not friendly or not enemies:
    print("SKIP: Need friendly unit AND enemy")
else:
    unit_id = friendly[0]['id']
    enemy_x = enemies[0]['x']
    enemy_y = enemies[0]['y']
```

| Step | Command | Expected | Status |
|------|---------|----------|--------|
| 1 | `ATTACK;{unit_id};{enemy_x};{enemy_y}` | Command sent | ☐ |
| 2 | Observe unit | Unit targets enemy/position | Visual check | ☐ |
| 3 | Check combat | Turrets or units fire | Visual confirmation | ☐ |

**PASS CRITERIA**:
- ✓ Command sent without error
- ✓ Unit targets location (turrets rotate, units move)
- ✓ Combat occurs (optional but validates)

**SKIP CONDITION**: No enemies detected in `state['enemies']`

**Expected Duration**: 2-5 seconds

---

### TEST 2.5: STOP Command

**Objective**: Halt unit movement

| Step | Command | Expected | Status |
|------|---------|----------|--------|
| 1 | `STOP;{unit_id}` | Command sent | ☐ |
| 2 | Observe unit | Unit stops moving | Visual check | ☐ |
| 3 | No crash | Game continues | ☐ |

**PASS CRITERIA**:
- ✓ Command sent without error
- ✓ Unit stops or idle (visual confirmation)

**ALTERNATIVE**: `STOP` (no unit_id) stops all units

**Expected Duration**: 1-2 seconds

---

### TEST 2.6: REPAIR Command

**Objective**: Repair a damaged building

**Setup**:
```python
# Find a damaged building
damaged = [b for b in state['buildings'] if b['hp'] < 1.0]
if not damaged:
    print("SKIP: No damaged buildings")
else:
    repair_x = damaged[0]['x']
    repair_y = damaged[0]['y']
    hp_before = damaged[0]['hp']
```

| Step | Command | Expected | Status |
|------|---------|----------|--------|
| 1 | `REPAIR;{repair_x};{repair_y}` | Command sent | ☐ |
| 2 | Wait 2 seconds | Building repairs | ☐ |
| 3 | Check state | Building `hp` increased | ☐ |

**PASS CRITERIA**:
- ✓ Command sent without error
- ✓ Building HP increased in next state

**SKIP CONDITION**: All buildings at 100% HP

**Expected Duration**: 3-5 seconds

---

### TEST 2.7: DELETE Command

**Objective**: Deconstruct a building

**Setup**:
```python
# Find a non-critical building (not core)
destructible = [b for b in state['buildings'] if b.get('block') != 'core-sharded']
if not destructible:
    print("SKIP: No destructible buildings")
else:
    delete_x = destructible[0]['x']
    delete_y = destructible[0]['y']
```

| Step | Command | Expected | Status |
|------|---------|----------|--------|
| 1 | `DELETE;{delete_x};{delete_y}` | Command sent | ☐ |
| 2 | Check in-game | Building disappears | Visual check | ☐ |
| 3 | Verify state | Building removed from next state | ☐ |

**PASS CRITERIA**:
- ✓ Command sent without error
- ✓ Building no longer visible
- ✓ Building removed from state['buildings']

**WARNING**: This permanently destroys the building. Use test-only turrets.

**Expected Duration**: 2-4 seconds

---

### TEST 2.8: UPGRADE Command

**Objective**: Upgrade block to next tier

**Setup**:
```python
# Find an upgradeable building (e.g., turrets)
upgradeable = [b for b in state['buildings'] if b.get('block') in ['duo', 'scatter', 'hail']]
if not upgradeable:
    print("SKIP: No upgradeable buildings")
else:
    upgrade_x = upgradeable[0]['x']
    upgrade_y = upgradeable[0]['y']
```

| Step | Command | Expected | Status |
|------|---------|----------|--------|
| 1 | `UPGRADE;{upgrade_x};{upgrade_y}` | Command sent | ☐ |
| 2 | Check in-game | Building changes appearance | Visual check | ☐ |
| 3 | Verify resources | Core items decreased | ☐ |

**PASS CRITERIA**:
- ✓ Command sent without error
- ✓ Building upgraded (visual change)
- ✓ Resources consumed

**SKIP CONDITION**: No upgradeable buildings available

**Expected Duration**: 3-5 seconds

---

### TEST 2.9: FACTORY Command

**Objective**: Spawn unit from factory

**Setup**:
```python
# Find factory/spawn building
factories = [b for b in state['buildings'] if 'spawn' in b.get('block', '').lower()]
if not factories:
    print("SKIP: No factories")
else:
    factory_x = factories[0]['x']
    factory_y = factories[0]['y']
    units_before = len(state['friendlyUnits'])
```

| Step | Command | Expected | Status |
|------|---------|----------|--------|
| 1 | `FACTORY;{factory_x};{factory_y};poly` | Command sent | ☐ |
| 2 | Wait 2 seconds | Factory produces unit | ☐ |
| 3 | Check state | New unit in `friendlyUnits` | ☐ |
| 4 | Verify in-game | Unit visible on map | Visual check | ☐ |

**PASS CRITERIA**:
- ✓ Command sent without error
- ✓ Unit count increased
- ✓ New unit visible and moving

**UNIT TYPES**: `poly`, `mega`, `glaive`, `reaper`, `ferry`, etc.

**SKIP CONDITION**: No factories built

**Expected Duration**: 4-6 seconds

---

## Performance Tests

### TEST 3.1: State Update Frequency

**Objective**: Verify mod sends states at expected rate

```bash
python -c "
import socket, json, time
sock = socket.socket()
sock.connect(('localhost', 9000))
start = time.time()
count = 0
while time.time() - start < 10:
    try:
        state = json.loads(sock.recv(4096).decode())
        count += 1
    except:
        break
elapsed = time.time() - start
sock.close()
print(f'States in {elapsed:.1f}s: {count}')
print(f'Rate: {count/elapsed:.1f} states/sec')
print(f'Expected: 5-10 states/sec')
"
```

| Metric | Expected | Your Result | Status |
|--------|----------|-------------|--------|
| States/10 seconds | 50-100 | ___ | ☐ |
| Frequency | 5-10 Hz | ___ Hz | ☐ |
| Dropped packets | 0 | ___ | ☐ |

**PASS CRITERIA**:
- ✓ ≥50 states in 10 seconds (≥5 Hz)
- ✓ No connection drops
- ✓ No JSON errors

**WARNING**: If <2 states/sec, check:
1. Is Mindustry paused? (Unpause F5)
2. Is game updating? (Check wave timer)
3. Try reducing `gridRadius` in `scripts/main.js`

---

### TEST 3.2: Connection Stability (30 seconds)

**Objective**: Verify sustained connection without drops

```bash
python -c "
import socket, json, time
sock = socket.socket()
sock.connect(('localhost', 9000))
start = time.time()
errors = 0
while time.time() - start < 30:
    try:
        state = json.loads(sock.recv(4096).decode())
    except Exception as e:
        errors += 1
        print(f'Error: {e}')
sock.close()
print(f'Errors in 30s: {errors}')
print(f'Status: {'✓ PASS' if errors == 0 else '✗ FAIL'}')
"
```

| Metric | Expected | Your Result | Status |
|--------|----------|-------------|--------|
| Connection errors | 0 | ___ | ☐ |
| Timeout errors | 0 | ___ | ☐ |
| JSON parse errors | 0 | ___ | ☐ |

**PASS CRITERIA**: 0 errors during 30-second test

**If errors occur**: Restart Mindustry and try again

---

## Full Test Suite Execution

### TEST 4.1: Automated Test Run

```bash
# Run with verbose output
python test_mimi_client.py --verbose

# Or with custom host/port
python test_mimi_client.py --host localhost --port 9000 --verbose
```

**Expected output format**:
```
[TEST] Connection to Mimi Gateway
[PASS] Connected to localhost:9000
[PASS] State validation passed (12 fields)

[TEST] BUILD command execution
[PASS] BUILD command sent and acknowledged

...

======================
Test Results Summary
======================
  ✓ PASS  CONNECTION
  ✓ PASS  BUILD
  ✓ PASS  UNIT_MOVE
  ✓ PASS  ATTACK
  ✓ PASS  STOP
  ✓ PASS  REPAIR
  ✓ PASS  DELETE
  ✓ PASS  UPGRADE
  ✓ PASS  MSG
  ✓ PASS  FACTORY

Total: 10/10 tests passed
All tests passed! ✓
```

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| CONNECTION | PASS | ___ | ☐ |
| BUILD | PASS | ___ | ☐ |
| UNIT_MOVE | PASS | ___ | ☐ |
| ATTACK | PASS | ___ | ☐ |
| STOP | PASS | ___ | ☐ |
| REPAIR | PASS | ___ | ☐ |
| DELETE | PASS | ___ | ☐ |
| UPGRADE | PASS | ___ | ☐ |
| MSG | PASS | ___ | ☐ |
| FACTORY | PASS | ___ | ☐ |

**PASS CRITERIA**: ≥8/10 tests pass

**Acceptable skips**: UNIT_MOVE, ATTACK, REPAIR, FACTORY (if no units/buildings available)

---

## Failure Recovery

### If a command test fails:

1. **Check preconditions**:
   - Is Mindustry still running? (Check window)
   - Is mod still loaded? (F1 console check)
   - Try receiving next state: `python -c "import socket, json; s=socket.socket(); s.connect(('localhost', 9000)); print(json.loads(s.recv(4096).decode()).get('time'))"`

2. **Restart connection**:
   ```bash
   # Close any running clients
   pkill -f test_mimi_client.py
   
   # Wait 2 seconds
   sleep 2
   
   # Restart test
   python test_mimi_client.py --verbose
   ```

3. **Check game state**:
   - Is the game paused? (Press F5 to unpause)
   - Are there enough resources? (Check core inventory)
   - Is power available? (Check power bar)

4. **Consult troubleshooting**:
   - See DEPLOYMENT_GUIDE.md → Troubleshooting
   - Check Mindustry log: `~/.mindustry/logs/`

---

## Sign-Off

| Item | Verified | Date | Initials |
|------|----------|------|----------|
| Pre-test checks | ☐ | ___ | ___ |
| Connection test | ☐ | ___ | ___ |
| State validation | ☐ | ___ | ___ |
| All 9 commands | ☐ | ___ | ___ |
| Performance test | ☐ | ___ | ___ |
| Stability test | ☐ | ___ | ___ |

**Final Status**:
- ☐ All tests PASS - Ready for production
- ☐ Most tests PASS - Ready with limitations
- ☐ Some tests FAIL - Needs troubleshooting
- ☐ Critical failures - See Troubleshooting section

**Notes**:
```
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
```

---

## Next Steps

If all tests pass:
- ✓ Proceed to RL training setup (see API_DOCUMENTATION.md)
- ✓ Begin custom client development
- ✓ Integrate with Gymnasium environment

If tests fail:
- ✗ Review DEPLOYMENT_GUIDE.md troubleshooting
- ✗ Restart Mindustry and retry
- ✗ Check game logs: `~/.mindustry/logs/`
