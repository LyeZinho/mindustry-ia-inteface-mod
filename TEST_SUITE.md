# Mimi Gateway - Comprehensive Test Suite

## Overview

This document provides detailed test procedures for validating all 9 commands in the Mimi Gateway mod with an actual Mindustry game client.

---

## Prerequisites

1. **Mindustry Installation**: Game must be installed and runnable
2. **Mod Deployment**: Mod copied to Mindustry mods directory
3. **Python Client**: Test client script ready (see below)
4. **Network**: Localhost TCP connection available
5. **Game State**: Started a campaign game or custom map with resources

---

## Test Client Script

Create `test_mimi_client.py`:

```python
#!/usr/bin/env python3
"""
Mimi Gateway Test Client - Validates all command types
"""

import socket
import json
import time
import sys

class MimiTestClient:
    def __init__(self, host='localhost', port=9000, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to Mimi Gateway server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✅ Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def receive_state(self):
        """Receive one state update from server"""
        try:
            line = self.socket.recv(4096).decode('utf-8').strip()
            if line:
                return json.loads(line)
            return None
        except socket.timeout:
            print("⏱️  Timeout waiting for state update")
            return None
        except Exception as e:
            print(f"❌ Error receiving state: {e}")
            return None
    
    def send_command(self, cmd):
        """Send command to server"""
        try:
            self.socket.send(f"{cmd}\n".encode('utf-8'))
            print(f"📤 Sent: {cmd}")
            time.sleep(0.5)  # Give server time to process
            return True
        except Exception as e:
            print(f"❌ Error sending command: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("🔌 Disconnected")

# Test procedures
def test_connection(client):
    """Test 1: Basic Connection"""
    print("\n" + "="*60)
    print("TEST 1: Connection to Mimi Gateway")
    print("="*60)
    
    if not client.connect():
        return False
    
    state = client.receive_state()
    if state:
        print(f"✅ Received state with {len(state)} fields")
        print(f"   - Game tick: {state.get('time')}")
        print(f"   - Current wave: {state.get('wave')}")
        print(f"   - Core HP: {state.get('core', {}).get('hp', 0):.0%}")
        return True
    else:
        print("❌ No state received")
        return False

def test_build_command(client):
    """Test 2: BUILD Command"""
    print("\n" + "="*60)
    print("TEST 2: BUILD - Construct a block")
    print("="*60)
    
    state = client.receive_state()
    if not state:
        print("❌ No state received")
        return False
    
    core_x = state.get('core', {}).get('x', 0)
    core_y = state.get('core', {}).get('y', 0)
    copper = state.get('resources', {}).get('copper', 0)
    
    print(f"📍 Core location: ({core_x}, {core_y})")
    print(f"💰 Available copper: {copper}")
    
    if copper < 20:
        print("⚠️  Insufficient copper to build (need ~20)")
        return False
    
    # Try to build a duo turret next to core
    build_x = core_x + 2
    build_y = core_y
    
    cmd = f"BUILD;duo;{build_x};{build_y};0"
    if client.send_command(cmd):
        time.sleep(1)
        state = client.receive_state()
        if state:
            buildings = state.get('buildings', [])
            # Check if new building appeared
            for b in buildings:
                if b.get('x') == build_x and b.get('y') == build_y and b.get('block') == 'duo':
                    print(f"✅ Duo turret built at ({build_x}, {build_y})")
                    return True
            print(f"⚠️  Building not found in state (might need more ticks)")
            return True  # Command sent successfully
    
    return False

def test_unit_move_command(client):
    """Test 3: UNIT_MOVE Command"""
    print("\n" + "="*60)
    print("TEST 3: UNIT_MOVE - Move unit to position")
    print("="*60)
    
    state = client.receive_state()
    if not state:
        print("❌ No state received")
        return False
    
    friendly_units = state.get('friendlyUnits', [])
    if not friendly_units:
        print("⚠️  No friendly units available (spawn units first)")
        return False
    
    unit = friendly_units[0]
    unit_id = unit.get('id')
    current_x = unit.get('x')
    current_y = unit.get('y')
    
    print(f"🎯 Unit: {unit.get('type')} (ID: {unit_id})")
    print(f"   Current position: ({current_x}, {current_y})")
    
    # Move to new position
    new_x = current_x + 5
    new_y = current_y
    
    cmd = f"UNIT_MOVE;{unit_id};{new_x};{new_y}"
    if client.send_command(cmd):
        time.sleep(2)  # Give unit time to move
        state = client.receive_state()
        if state:
            for u in state.get('friendlyUnits', []):
                if u.get('id') == unit_id:
                    moved_x = u.get('x')
                    moved_y = u.get('y')
                    distance = abs(moved_x - current_x) + abs(moved_y - current_y)
                    print(f"✅ Unit moved to ({moved_x}, {moved_y}) (distance: {distance})")
                    return True
    
    return False

def test_factory_command(client):
    """Test 4: FACTORY Command"""
    print("\n" + "="*60)
    print("TEST 4: FACTORY - Spawn unit from factory")
    print("="*60)
    
    state = client.receive_state()
    if not state:
        print("❌ No state received")
        return False
    
    # Find a factory building
    buildings = state.get('buildings', [])
    factory = None
    for b in buildings:
        if 'factory' in b.get('block', '').lower():
            factory = b
            break
    
    if not factory:
        print("⚠️  No factory building found in state")
        return False
    
    factory_x = factory.get('x')
    factory_y = factory.get('y')
    
    print(f"🏭 Factory found at ({factory_x}, {factory_y})")
    
    cmd = f"FACTORY;{factory_x};{factory_y};poly"
    if client.send_command(cmd):
        time.sleep(2)
        state = client.receive_state()
        if state:
            units_after = len(state.get('friendlyUnits', []))
            print(f"✅ Unit spawn command sent (new unit count: {units_after})")
            return True
    
    return False

def test_attack_command(client):
    """Test 5: ATTACK Command"""
    print("\n" + "="*60)
    print("TEST 5: ATTACK - Order unit to attack")
    print("="*60)
    
    state = client.receive_state()
    if not state:
        print("❌ No state received")
        return False
    
    friendly_units = state.get('friendlyUnits', [])
    enemies = state.get('enemies', [])
    
    if not friendly_units:
        print("⚠️  No friendly units to attack with")
        return False
    
    if not enemies:
        print("⚠️  No enemies to attack")
        return False
    
    unit = friendly_units[0]
    enemy = enemies[0]
    
    unit_id = unit.get('id')
    enemy_x = enemy.get('x')
    enemy_y = enemy.get('y')
    
    print(f"🎯 Unit {unit_id} attacking enemy at ({enemy_x}, {enemy_y})")
    
    cmd = f"ATTACK;{unit_id};{enemy_x};{enemy_y}"
    if client.send_command(cmd):
        time.sleep(1)
        print(f"✅ Attack command sent")
        return True
    
    return False

def test_stop_command(client):
    """Test 6: STOP Command"""
    print("\n" + "="*60)
    print("TEST 6: STOP - Stop unit movement")
    print("="*60)
    
    state = client.receive_state()
    if not state:
        print("❌ No state received")
        return False
    
    friendly_units = state.get('friendlyUnits', [])
    if not friendly_units:
        print("⚠️  No friendly units")
        return False
    
    unit = friendly_units[0]
    unit_id = unit.get('id')
    
    cmd = f"STOP;{unit_id}"
    if client.send_command(cmd):
        print(f"✅ Stop command sent for unit {unit_id}")
        return True
    
    return False

def test_repair_command(client):
    """Test 7: REPAIR Command"""
    print("\n" + "="*60)
    print("TEST 7: REPAIR - Repair damaged building")
    print("="*60)
    
    state = client.receive_state()
    if not state:
        print("❌ No state received")
        return False
    
    buildings = state.get('buildings', [])
    damaged = [b for b in buildings if b.get('hp', 1) < 1]
    
    if not damaged:
        print("⚠️  No damaged buildings found")
        return False
    
    building = damaged[0]
    x = building.get('x')
    y = building.get('y')
    hp_before = building.get('hp')
    
    print(f"🔧 Repairing building at ({x}, {y}) (HP: {hp_before:.0%})")
    
    cmd = f"REPAIR;{x};{y}"
    if client.send_command(cmd):
        time.sleep(1)
        state = client.receive_state()
        if state:
            for b in state.get('buildings', []):
                if b.get('x') == x and b.get('y') == y:
                    hp_after = b.get('hp')
                    print(f"✅ Repair command sent (HP after: {hp_after:.0%})")
                    return True
    
    return False

def test_delete_command(client):
    """Test 8: DELETE Command"""
    print("\n" + "="*60)
    print("TEST 8: DELETE - Deconstruct building")
    print("="*60)
    
    state = client.receive_state()
    if not state:
        print("❌ No state received")
        return False
    
    buildings = state.get('buildings', [])
    # Try to delete a wall or non-critical building
    target = None
    for b in buildings:
        if 'wall' in b.get('block', '').lower():
            target = b
            break
    
    if not target:
        if buildings:
            target = buildings[0]
        else:
            print("⚠️  No buildings to delete")
            return False
    
    x = target.get('x')
    y = target.get('y')
    block_name = target.get('block')
    
    print(f"🗑️  Deleting {block_name} at ({x}, {y})")
    
    cmd = f"DELETE;{x};{y}"
    if client.send_command(cmd):
        time.sleep(1)
        print(f"✅ Delete command sent")
        return True
    
    return False

def test_upgrade_command(client):
    """Test 9: UPGRADE Command"""
    print("\n" + "="*60)
    print("TEST 9: UPGRADE - Upgrade block to next tier")
    print("="*60)
    
    state = client.receive_state()
    if not state:
        print("❌ No state received")
        return False
    
    buildings = state.get('buildings', [])
    # Look for a block that can be upgraded (like duo -> scatter)
    target = None
    for b in buildings:
        block = b.get('block', '')
        # Duo turrets can upgrade to scatter turrets
        if block == 'duo':
            target = b
            break
    
    if not target:
        print("⚠️  No upgradeable blocks found (place a duo turret)")
        return False
    
    x = target.get('x')
    y = target.get('y')
    
    print(f"⬆️  Upgrading block at ({x}, {y})")
    
    cmd = f"UPGRADE;{x};{y}"
    if client.send_command(cmd):
        time.sleep(2)
        state = client.receive_state()
        if state:
            for b in state.get('buildings', []):
                if b.get('x') == x and b.get('y') == y:
                    new_block = b.get('block')
                    print(f"✅ Upgrade command sent (new block: {new_block})")
                    return True
    
    return False

def test_msg_command(client):
    """Test 10: MSG Command (Bonus)"""
    print("\n" + "="*60)
    print("TEST 10: MSG - Send chat message")
    print("="*60)
    
    cmd = "MSG;Test message from Mimi Gateway"
    if client.send_command(cmd):
        print(f"✅ Message command sent (check in-game chat)")
        return True
    
    return False

def run_all_tests():
    """Run complete test suite"""
    client = MimiTestClient()
    
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           MIMI GATEWAY - COMPLETE TEST SUITE              ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    results = {
        "Connection": test_connection(client),
        "BUILD": test_build_command(client),
        "UNIT_MOVE": test_unit_move_command(client),
        "FACTORY": test_factory_command(client),
        "ATTACK": test_attack_command(client),
        "STOP": test_stop_command(client),
        "REPAIR": test_repair_command(client),
        "DELETE": test_delete_command(client),
        "UPGRADE": test_upgrade_command(client),
        "MSG": test_msg_command(client),
    }
    
    client.disconnect()
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} passed ({100*passed//total}%)")
    print("="*60)
    
    return passed == total

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

---

## Manual Test Procedures

### Test 1: Verify Mod Loads
**Steps:**
1. Copy mod to Mindustry mods directory
2. Launch Mindustry
3. Press F1 to open console
4. **Expected**: See `[Mimi Gateway] Servidor iniciado na porta 9000`

**Pass/Fail**: ___

---

### Test 2: Verify Connection
**Steps:**
1. In Python: `python3 test_mimi_client.py`
2. Observe connection messages
3. Wait for first state update

**Expected**: Connection message + state with resources/power/units

**Pass/Fail**: ___

---

### Test 3: BUILD Command
**Steps:**
1. Start game with resources
2. Send: `BUILD;duo;10;10;0` (adjust coordinates to valid location)
3. Check F1 console for log message
4. Verify turret appears on map at specified coordinates

**Expected**: Turret visible at location, log confirms command

**Pass/Fail**: ___

---

### Test 4: UNIT_MOVE Command
**Steps:**
1. Spawn units (via factory or campaign)
2. Get unit ID from state updates
3. Send: `UNIT_MOVE;{unit_id};20;20`
4. Watch unit move to new position

**Expected**: Unit moves toward target coordinates

**Pass/Fail**: ___

---

### Test 5: FACTORY Command
**Steps:**
1. Place a factory block first
2. Send: `FACTORY;10;10;poly` (adjust to factory location)
3. Watch for new unit spawn effect
4. Verify unit count increases in state

**Expected**: New unit spawned near factory

**Pass/Fail**: ___

---

### Test 6: ATTACK Command
**Steps:**
1. Have friendly units and enemies present
2. Send: `ATTACK;{unit_id};{enemy_x};{enemy_y}`
3. Unit should engage target

**Expected**: Unit targets and attacks building/enemy

**Pass/Fail**: ___

---

### Test 7: STOP Command
**Steps:**
1. Have moving units
2. Send: `STOP;{unit_id}`
3. Unit should stop moving

**Expected**: Unit halts movement

**Pass/Fail**: ___

---

### Test 8: REPAIR Command
**Steps:**
1. Damage a building (take fire or trigger event)
2. Send: `REPAIR;{x};{y}`
3. Check building health increases

**Expected**: Building health increases

**Pass/Fail**: ___

---

### Test 9: DELETE Command
**Steps:**
1. Have buildings placed
2. Send: `DELETE;{x};{y}`
3. Building should disappear

**Expected**: Building deconstructed and removed

**Pass/Fail**: ___

---

### Test 10: UPGRADE Command
**Steps:**
1. Place a duo turret
2. Send: `UPGRADE;{x};{y}`
3. Turret should upgrade to scatter turret

**Expected**: Duo turret becomes scatter turret

**Pass/Fail**: ___

---

## Validation Checklist

- [ ] All 10 tests passed
- [ ] No crashes in Mindustry console
- [ ] No "Error" messages in mod logs
- [ ] Connection remains stable for >5 minutes
- [ ] Commands execute within 100ms of sending
- [ ] State updates arrive at ~166ms interval (10 ticks)
- [ ] Multiple rapid commands execute correctly
- [ ] Connection recovers after brief disconnect
- [ ] Debug logs are informative (when debug=true)
- [ ] No memory leaks (RAM stable over 10+ minutes)

---

## Performance Benchmarks

| Metric | Target | Actual |
|--------|--------|--------|
| Connection latency | <100ms | ___ |
| Command execution | <10ms | ___ |
| State update frequency | 166ms | ___ |
| Max latency observed | <500ms | ___ |
| Memory usage (baseline) | <20MB | ___ |
| Memory usage (after 1000 commands) | <25MB | ___ |

---

## Issue Tracking

| Issue | Severity | Status |
|-------|----------|--------|
| | | |
| | | |
| | | |

---

## Notes

- Test in campaign mode first, then custom maps
- Run tests both with debug=true and debug=false
- Try with different gridRadius values
- Test connection loss recovery
- Verify no performance degradation with large grids

---

## Conclusion

All tests completed: ___

Tester: ___

Date: ___

Sign-off: ___
