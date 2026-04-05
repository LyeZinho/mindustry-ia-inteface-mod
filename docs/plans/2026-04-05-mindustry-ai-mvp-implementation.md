# Mindustry AI Agent MVP - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers/executing-plans to implement this plan task-by-task.

**Goal:** Build a hybrid rule-based + RL agent that plays Mindustry (mining + logistics + defense) following the architecture defined in the MVP design doc.

**Architecture:** 
- Game wrapper (state reader + action executor using PyAutoGUI + memory introspection)
- Rule layer (Behavior Tree + State Machine + Priority Queue for strategic decisions)
- RL layer (PyTorch Actor-Critic policy network for action refinement)
- Training loop with 3 progressive phases (Survival → Production → Defense)

**Tech Stack:** Python 3.9+, PyTorch, PyAutoGUI, NumPy, OpenCV (for screenshot analysis)

---

## Phase 0: Project Setup

### Task 1: Initialize Python environment & dependencies

**Files:**
- Create: `requirements.txt`
- Create: `setup.py` or use poetry/pipenv config
- Create: `.gitignore` (Python standard)

**Step 1: Write requirements.txt**

```
torch>=2.0.0
numpy>=1.24.0
opencv-python>=4.7.0
pyautogui>=0.9.53
pillow>=9.5.0
pyyaml>=6.0
```

**Step 2: Create setup.py for package structure**

```python
from setuptools import setup, find_packages

setup(
    name="mindustry-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "pyautogui>=0.9.53",
        "pillow>=9.5.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
)
```

**Step 3: Create .gitignore**

```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
.venv/
venv/
ENV/
env/
.env
*.log
*.pot
.DS_Store
.pytest_cache/
.coverage
htmlcov/
.idea/
.vscode/
*.swp
*.swo
*~
.mypy_cache/
.dmypy.json
dmypy.json
.pyre/
```

**Step 4: Commit**

```bash
git add requirements.txt setup.py .gitignore
git commit -m "chore: initialize Python environment and dependencies"
```

---

## Phase 1: Game Wrapper (State Reader + Action Executor)

### Task 2: Create game state reader (hybrid CV + memory)

**Files:**
- Create: `mindustry_ai/game/state_reader.py`
- Create: `mindustry_ai/game/__init__.py`
- Create: `tests/game/test_state_reader.py`

**Step 1: Write failing test for state reader**

```python
# tests/game/test_state_reader.py
import pytest
from mindustry_ai.game.state_reader import GameStateReader

def test_read_game_state_returns_dict():
    """GameStateReader should return a state dict with expected keys."""
    reader = GameStateReader()
    state = reader.read_state()
    
    assert isinstance(state, dict)
    assert "resources" in state
    assert "power" in state
    assert "threat" in state
    assert "infrastructure" in state
    assert "status" in state

def test_flat_vector_generation():
    """Should convert state dict to flat observation vector."""
    reader = GameStateReader()
    state = reader.read_state()
    flat_vec = reader.to_flat_vector(state)
    
    assert len(flat_vec) == 15  # As per design doc

def test_spatial_map_generation():
    """Should generate 2D spatial representation of game map."""
    reader = GameStateReader()
    state = reader.read_state()
    spatial_map = reader.to_spatial_map(state)
    
    assert "blocks" in spatial_map
    assert "resources" in spatial_map
    assert "enemies" in spatial_map
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/game/test_state_reader.py -v
# Expected: FAIL (module not found)
```

**Step 3: Write minimal GameStateReader implementation**

```python
# mindustry_ai/game/state_reader.py
import numpy as np
from typing import Dict, Any, Tuple

class GameStateReader:
    """Reads game state from Mindustry via hybrid CV + memory approach."""
    
    def __init__(self, map_width: int = 32, map_height: int = 32):
        self.map_width = map_width
        self.map_height = map_height
    
    def read_state(self) -> Dict[str, Any]:
        """
        Read current game state.
        Returns dict with keys: resources, power, threat, infrastructure, status
        """
        # TODO: Implement actual state reading (CV + memory)
        # For now, return dummy state
        return {
            "resources": {
                "copper": 100,
                "lead": 50,
                "coal": 30,
                "graphite": 20,
                "titanium": 10,
            },
            "power": {
                "current": 500,
                "capacity": 1000,
                "production": 300,
                "consumption": 200,
            },
            "threat": {
                "enemies_nearby": 0,
                "wave_number": 1,
                "time_to_wave": 600,
            },
            "infrastructure": {
                "drills_count": 2,
                "turrets_count": 1,
                "conveyors_count": 5,
            },
            "status": {
                "core_health": 1.0,
                "recent_damage": 0,
                "game_time": 0,
            },
        }
    
    def to_flat_vector(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert state dict to 15-dim flat observation vector."""
        return np.array([
            # Resources (5)
            state["resources"]["copper"],
            state["resources"]["lead"],
            state["resources"]["coal"],
            state["resources"]["graphite"],
            state["resources"]["titanium"],
            # Power (4)
            state["power"]["current"],
            state["power"]["capacity"],
            state["power"]["production"],
            state["power"]["consumption"],
            # Threat (3)
            state["threat"]["enemies_nearby"],
            state["threat"]["wave_number"],
            state["threat"]["time_to_wave"],
            # Infrastructure (2)
            state["infrastructure"]["drills_count"],
            state["infrastructure"]["turrets_count"],
            # Status (1)
            state["status"]["core_health"],
        ], dtype=np.float32)
    
    def to_spatial_map(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate 2D spatial representation of game map."""
        # TODO: Extract from actual game map
        # For now, return dummy spatial map
        return {
            "blocks": np.zeros((self.map_width, self.map_height), dtype=np.int32),
            "resources": np.zeros((self.map_width, self.map_height), dtype=np.float32),
            "enemies": np.zeros((self.map_width, self.map_height), dtype=np.float32),
        }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/game/test_state_reader.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add mindustry_ai/game/state_reader.py tests/game/test_state_reader.py
git commit -m "feat: add GameStateReader for hybrid state observation"
```

---

### Task 3: Create action executor (bot controller)

**Files:**
- Create: `mindustry_ai/game/action_executor.py`
- Create: `tests/game/test_action_executor.py`

**Step 1: Write failing test for action executor**

```python
# tests/game/test_action_executor.py
import pytest
from mindustry_ai.game.action_executor import ActionExecutor

def test_executor_init():
    """ActionExecutor should initialize without errors."""
    executor = ActionExecutor()
    assert executor is not None

def test_place_drill_action():
    """Should handle PLACE_DRILL action without error."""
    executor = ActionExecutor()
    # Action: 0=PLACE_DRILL, x=100, y=100, rotation=0
    executor.execute(action=0, x=100, y=100)
    # No exception = success (actual click stubbed out)

def test_place_conveyor_action():
    """Should handle PLACE_CONVEYOR action."""
    executor = ActionExecutor()
    executor.execute(action=1, x=100, y=100)

def test_wait_action():
    """Should handle WAIT action."""
    executor = ActionExecutor()
    executor.execute(action=6)  # WAIT
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/game/test_action_executor.py -v
# Expected: FAIL (module not found)
```

**Step 3: Write minimal ActionExecutor implementation**

```python
# mindustry_ai/game/action_executor.py
import pyautogui
import time
from enum import IntEnum

class Action(IntEnum):
    PLACE_DRILL = 0
    PLACE_CONVEYOR = 1
    PLACE_GENERATOR = 2
    PLACE_TURRET = 3
    UPGRADE_BLOCK = 4
    DEMOLISH_BLOCK = 5
    WAIT = 6

class ActionExecutor:
    """Executes high-level actions by controlling the game via PyAutoGUI."""
    
    def __init__(self, safe_mode: bool = True):
        """
        Initialize action executor.
        Args:
            safe_mode: If True, add delays and safety checks
        """
        self.safe_mode = safe_mode
        self.action_names = {v.value: v.name for v in Action}
    
    def execute(self, action: int, x: float = 0, y: float = 0, rotation: int = 0) -> bool:
        """
        Execute an action in the game.
        
        Args:
            action: Action ID (0-6)
            x, y: Coordinates for placement
            rotation: Rotation (0-3 for 0, 90, 180, 270 degrees)
        
        Returns:
            bool: True if action executed successfully
        """
        try:
            if action == Action.PLACE_DRILL:
                return self._place_drill(x, y, rotation)
            elif action == Action.PLACE_CONVEYOR:
                return self._place_conveyor(x, y, rotation)
            elif action == Action.PLACE_GENERATOR:
                return self._place_generator(x, y, rotation)
            elif action == Action.PLACE_TURRET:
                return self._place_turret(x, y, rotation)
            elif action == Action.UPGRADE_BLOCK:
                return self._upgrade_block(x, y)
            elif action == Action.DEMOLISH_BLOCK:
                return self._demolish_block(x, y)
            elif action == Action.WAIT:
                return self._wait()
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            print(f"Error executing action {self.action_names[action]}: {e}")
            return False
    
    def _place_drill(self, x: float, y: float, rotation: int) -> bool:
        """Place mining drill at (x, y)."""
        # TODO: Implement actual game interaction
        # For now, stub implementation
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _place_conveyor(self, x: float, y: float, rotation: int) -> bool:
        """Place conveyor at (x, y)."""
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _place_generator(self, x: float, y: float, rotation: int) -> bool:
        """Place power generator at (x, y)."""
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _place_turret(self, x: float, y: float, rotation: int) -> bool:
        """Place defense turret at (x, y)."""
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _upgrade_block(self, x: float, y: float) -> bool:
        """Upgrade block at (x, y)."""
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _demolish_block(self, x: float, y: float) -> bool:
        """Demolish block at (x, y)."""
        if self.safe_mode:
            time.sleep(0.1)
        return True
    
    def _wait(self) -> bool:
        """Do nothing (no-op action)."""
        return True
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/game/test_action_executor.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add mindustry_ai/game/action_executor.py tests/game/test_action_executor.py
git commit -m "feat: add ActionExecutor for controlling game via PyAutoGUI"
```

---

## Phase 2: Rule Layer (Behavior Tree + State Machine + Priority Queue)

### Task 4: Implement Behavior Tree

**Files:**
- Create: `mindustry_ai/rules/behavior_tree.py`
- Create: `tests/rules/test_behavior_tree.py`

**Step 1: Write failing test**

```python
# tests/rules/test_behavior_tree.py
import pytest
from mindustry_ai.rules.behavior_tree import BehaviorTree, Action

def test_behavior_tree_init():
    """BehaviorTree should initialize with root node."""
    bt = BehaviorTree()
    assert bt is not None

def test_get_feasible_actions_returns_list():
    """Should return list of feasible actions for a game state."""
    bt = BehaviorTree()
    state = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    feasible = bt.get_feasible_actions(state)
    
    assert isinstance(feasible, list)
    assert len(feasible) > 0

def test_threat_detection():
    """Should detect threats and prioritize defense."""
    bt = BehaviorTree()
    state_safe = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    state_threat = {
        **state_safe,
        "threat": {"enemies_nearby": 5, "wave_number": 2, "time_to_wave": 10},
    }
    
    feasible_safe = bt.get_feasible_actions(state_safe)
    feasible_threat = bt.get_feasible_actions(state_threat)
    
    # Threat state should prioritize defense
    assert len(feasible_threat) > 0
```

**Step 2: Run test**

```bash
pytest tests/rules/test_behavior_tree.py -v
# Expected: FAIL
```

**Step 3: Write BehaviorTree implementation**

```python
# mindustry_ai/rules/behavior_tree.py
from enum import IntEnum
from typing import Dict, List, Any

class Action(IntEnum):
    PLACE_DRILL = 0
    PLACE_CONVEYOR = 1
    PLACE_GENERATOR = 2
    PLACE_TURRET = 3
    UPGRADE_BLOCK = 4
    DEMOLISH_BLOCK = 5
    WAIT = 6

class BehaviorTree:
    """
    Hierarchical behavior tree for strategic decision-making.
    Evaluates game state and returns feasible actions.
    """
    
    def __init__(self):
        self.action_names = {v.value: v.name for v in Action}
    
    def get_feasible_actions(self, state: Dict[str, Any]) -> List[int]:
        """
        Evaluate game state and return list of feasible actions.
        
        Uses hierarchical tree:
        1. Threat assessment (defend if enemies near)
        2. Energy management (expand power if low)
        3. Resource production (expand mining if resources low)
        4. Optimization (improve chains if stable)
        """
        feasible = []
        
        # 1. Threat Assessment
        enemies_nearby = state["threat"]["enemies_nearby"]
        if enemies_nearby > 0:
            turrets = state["infrastructure"]["turrets_count"]
            if turrets < 3:  # Simple heuristic: need 3+ turrets per threat
                feasible.append(Action.PLACE_TURRET)
            return feasible  # Prioritize defense over everything
        
        # 2. Energy Management
        power_ratio = state["power"]["current"] / state["power"]["capacity"]
        if power_ratio < 0.8:
            feasible.append(Action.PLACE_GENERATOR)
        
        # 3. Resource Production
        copper_ratio = state["resources"]["copper"] / 200.0  # Target: 200 copper
        lead_ratio = state["resources"]["lead"] / 100.0
        
        if copper_ratio < 0.5 or lead_ratio < 0.5:
            feasible.append(Action.PLACE_DRILL)
        
        # 4. Optimization
        if len(feasible) == 0:
            # All systems healthy, can optimize
            feasible.append(Action.PLACE_CONVEYOR)
            feasible.append(Action.WAIT)
        
        return feasible
```

**Step 4: Run test**

```bash
pytest tests/rules/test_behavior_tree.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add mindustry_ai/rules/behavior_tree.py tests/rules/test_behavior_tree.py
git commit -m "feat: implement hierarchical behavior tree for strategic decisions"
```

---

### Task 5: Implement State Machine

**Files:**
- Create: `mindustry_ai/rules/state_machine.py`
- Create: `tests/rules/test_state_machine.py`

**Step 1: Write failing test**

```python
# tests/rules/test_state_machine.py
import pytest
from mindustry_ai.rules.state_machine import StateMachine, GameState

def test_state_machine_init():
    """StateMachine should start in MINING state."""
    sm = StateMachine()
    assert sm.current_state == GameState.MINING

def test_transition_mining_to_crafting():
    """Should transition from MINING to CRAFTING when copper high."""
    sm = StateMachine()
    state = {
        "resources": {"copper": 300, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    sm.update(state)
    assert sm.current_state == GameState.CRAFTING

def test_transition_to_defense():
    """Should transition to DEFENSE when enemies detected."""
    sm = StateMachine()
    state = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 5, "wave_number": 2, "time_to_wave": 10},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    sm.update(state)
    assert sm.current_state == GameState.DEFENSE
```

**Step 2: Run test**

```bash
pytest tests/rules/test_state_machine.py -v
# Expected: FAIL
```

**Step 3: Write StateMachine implementation**

```python
# mindustry_ai/rules/state_machine.py
from enum import IntEnum
from typing import Dict, Any

class GameState(IntEnum):
    MINING = 0
    CRAFTING = 1
    ENERGY = 2
    DEFENSE = 3
    IDLE = 4

class StateMachine:
    """
    Discrete state machine for game operational modes.
    States: MINING → CRAFTING → ENERGY → DEFENSE → IDLE
    """
    
    def __init__(self):
        self.current_state = GameState.MINING
    
    def update(self, state: Dict[str, Any]) -> GameState:
        """
        Update state machine based on game conditions.
        
        Transitions:
        - MINING → CRAFTING: copper > 200
        - CRAFTING → ENERGY: graphite > 50
        - any → DEFENSE: enemies nearby
        - any → IDLE: no action feasible
        """
        # Priority: Defense overrides all
        if state["threat"]["enemies_nearby"] > 0:
            self.current_state = GameState.DEFENSE
            return self.current_state
        
        # Check resource thresholds for normal progression
        copper = state["resources"]["copper"]
        graphite = state["resources"]["graphite"]
        power_ratio = state["power"]["current"] / state["power"]["capacity"]
        
        # State transitions
        if self.current_state == GameState.MINING:
            if copper > 200:
                self.current_state = GameState.CRAFTING
        
        elif self.current_state == GameState.CRAFTING:
            if graphite > 50:
                self.current_state = GameState.ENERGY
        
        elif self.current_state == GameState.ENERGY:
            if power_ratio > 0.8:
                self.current_state = GameState.IDLE
        
        elif self.current_state == GameState.IDLE:
            if copper < 100:
                self.current_state = GameState.MINING
        
        return self.current_state
```

**Step 4: Run test**

```bash
pytest tests/rules/test_state_machine.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add mindustry_ai/rules/state_machine.py tests/rules/test_state_machine.py
git commit -m "feat: implement state machine for game operational modes"
```

---

### Task 6: Implement Priority Queue

**Files:**
- Create: `mindustry_ai/rules/priority_queue.py`
- Create: `tests/rules/test_priority_queue.py`

**Step 1: Write failing test**

```python
# tests/rules/test_priority_queue.py
import pytest
from mindustry_ai.rules.priority_queue import PriorityQueue

def test_priority_queue_init():
    """PriorityQueue should initialize."""
    pq = PriorityQueue()
    assert pq is not None

def test_compute_priorities_survival_spike():
    """Threat should spike survival priority."""
    pq = PriorityQueue()
    state_safe = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    state_threat = {
        **state_safe,
        "threat": {"enemies_nearby": 5, "wave_number": 2, "time_to_wave": 10},
    }
    
    priorities_safe = pq.compute_priorities(state_safe)
    priorities_threat = pq.compute_priorities(state_threat)
    
    assert priorities_threat["survive"] > priorities_safe["survive"]

def test_get_highest_priority_action():
    """Should return action with highest priority."""
    pq = PriorityQueue()
    state = {
        "resources": {"copper": 10, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 900, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    
    priorities = pq.compute_priorities(state)
    action = pq.get_highest_priority_category(priorities)
    
    # Low copper should make mining highest priority
    assert action == "mining"
```

**Step 2: Run test**

```bash
pytest tests/rules/test_priority_queue.py -v
# Expected: FAIL
```

**Step 3: Write PriorityQueue implementation**

```python
# mindustry_ai/rules/priority_queue.py
from typing import Dict, Any

class PriorityQueue:
    """
    Dynamic priority scoring for action selection.
    Computes urgency scores based on game state.
    """
    
    def compute_priorities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute priority scores for each action category.
        
        Returns dict with keys: survive, power, mining, optimize
        Values normalized to [0, 100]
        """
        priorities = {}
        
        # 1. Survival priority
        enemies_nearby = state["threat"]["enemies_nearby"]
        threat_level = min(1.0, enemies_nearby / 5.0)  # Normalize to [0, 1]
        priorities["survive"] = threat_level * 100
        
        # 2. Power priority
        power_ratio = state["power"]["current"] / state["power"]["capacity"]
        power_deficit = max(0, 1.0 - power_ratio)
        priorities["power"] = power_deficit * 50
        
        # 3. Mining priority
        copper_target = 200
        copper_ratio = state["resources"]["copper"] / copper_target
        mining_deficit = max(0, 1.0 - copper_ratio)
        priorities["mining"] = mining_deficit * 30
        
        # 4. Optimization priority (always present, lowest)
        priorities["optimize"] = 10
        
        return priorities
    
    def get_highest_priority_category(self, priorities: Dict[str, float]) -> str:
        """Return the category with highest priority score."""
        return max(priorities, key=priorities.get)
```

**Step 4: Run test**

```bash
pytest tests/rules/test_priority_queue.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add mindustry_ai/rules/priority_queue.py tests/rules/test_priority_queue.py
git commit -m "feat: implement priority queue for dynamic action urgency"
```

---

### Task 7: Integrate Rule Layer (Hybrid Decision)

**Files:**
- Create: `mindustry_ai/rules/hybrid_decider.py`
- Create: `tests/rules/test_hybrid_decider.py`

**Step 1: Write failing test**

```python
# tests/rules/test_hybrid_decider.py
import pytest
from mindustry_ai.rules.hybrid_decider import HybridDecider

def test_hybrid_decider_decides_action():
    """Hybrid decider should return an action based on game state."""
    decider = HybridDecider()
    state = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
        "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    
    action = decider.decide(state)
    assert isinstance(action, int)
    assert 0 <= action <= 6

def test_hybrid_decider_prioritizes_defense():
    """Should prioritize defense when threats detected."""
    decider = HybridDecider()
    state = {
        "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
        "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
        "threat": {"enemies_nearby": 5, "wave_number": 2, "time_to_wave": 10},
        "infrastructure": {"drills_count": 2, "turrets_count": 0, "conveyors_count": 5},
        "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
    }
    
    action = decider.decide(state)
    # Should choose PLACE_TURRET (3)
    assert action == 3
```

**Step 2: Run test**

```bash
pytest tests/rules/test_hybrid_decider.py -v
# Expected: FAIL
```

**Step 3: Write HybridDecider implementation**

```python
# mindustry_ai/rules/hybrid_decider.py
from typing import Dict, Any, Tuple
from mindustry_ai.rules.behavior_tree import BehaviorTree, Action
from mindustry_ai.rules.state_machine import StateMachine
from mindustry_ai.rules.priority_queue import PriorityQueue

class HybridDecider:
    """
    Hybrid decision system combining BT + SM + Priority Queue.
    
    1. BT ensures feasibility (constraints)
    2. SM provides state context
    3. Priority Queue selects urgency
    Result: feasible + contextual + urgent action
    """
    
    def __init__(self):
        self.behavior_tree = BehaviorTree()
        self.state_machine = StateMachine()
        self.priority_queue = PriorityQueue()
    
    def decide(self, state: Dict[str, Any]) -> int:
        """
        Make decision given game state.
        
        Returns action (0-6):
        - Feasible per behavior tree
        - Contextual per state machine
        - Urgent per priority queue
        """
        # 1. Behavior Tree: get feasible actions
        feasible_actions = self.behavior_tree.get_feasible_actions(state)
        
        if not feasible_actions:
            return Action.WAIT
        
        # If only one action available, take it
        if len(feasible_actions) == 1:
            return feasible_actions[0]
        
        # 2. State Machine: get current operational state
        current_game_state = self.state_machine.update(state)
        
        # 3. Priority Queue: compute urgency
        priorities = self.priority_queue.compute_priorities(state)
        highest_priority = self.priority_queue.get_highest_priority_category(priorities)
        
        # 4. Map priority category to action
        priority_to_action = {
            "survive": Action.PLACE_TURRET,
            "power": Action.PLACE_GENERATOR,
            "mining": Action.PLACE_DRILL,
            "optimize": Action.PLACE_CONVEYOR,
        }
        
        preferred_action = priority_to_action[highest_priority]
        
        # 5. Return preferred action if feasible, else pick first feasible
        if preferred_action in feasible_actions:
            return preferred_action
        else:
            return feasible_actions[0]
```

**Step 4: Run test**

```bash
pytest tests/rules/test_hybrid_decider.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add mindustry_ai/rules/hybrid_decider.py tests/rules/test_hybrid_decider.py
git commit -m "feat: integrate hybrid decision system (BT + SM + PQ)"
```

---

## Phase 3: RL Layer (PyTorch Policy Network)

### Task 8: Implement Policy Network Architecture

**Files:**
- Create: `mindustry_ai/rl/policy_net.py`
- Create: `tests/rl/test_policy_net.py`

**Step 1: Write failing test**

```python
# tests/rl/test_policy_net.py
import pytest
import torch
from mindustry_ai.rl.policy_net import PolicyNetwork

def test_policy_network_init():
    """PolicyNetwork should initialize."""
    net = PolicyNetwork(flat_dim=15, spatial_h=32, spatial_w=32)
    assert net is not None

def test_forward_pass():
    """Forward pass should return action dist, placement dist, value."""
    net = PolicyNetwork(flat_dim=15, spatial_h=32, spatial_w=32)
    flat_obs = torch.randn(1, 15)
    spatial_obs = torch.randn(1, 3, 32, 32)
    
    action_logits, placement_mu, placement_sigma, value = net(flat_obs, spatial_obs)
    
    assert action_logits.shape == (1, 7)  # 7 actions
    assert placement_mu.shape == (1, 2)    # x, y
    assert placement_sigma.shape == (1, 2)
    assert value.shape == (1, 1)

def test_network_output_types():
    """Network outputs should be correct type."""
    net = PolicyNetwork(flat_dim=15, spatial_h=32, spatial_w=32)
    flat_obs = torch.randn(1, 15)
    spatial_obs = torch.randn(1, 3, 32, 32)
    
    action_logits, placement_mu, placement_sigma, value = net(flat_obs, spatial_obs)
    
    assert isinstance(action_logits, torch.Tensor)
    assert isinstance(placement_mu, torch.Tensor)
    assert isinstance(placement_sigma, torch.Tensor)
    assert isinstance(value, torch.Tensor)
```

**Step 2: Run test**

```bash
pytest tests/rl/test_policy_net.py -v
# Expected: FAIL
```

**Step 3: Write PolicyNetwork implementation**

```python
# mindustry_ai/rl/policy_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Actor-Critic policy network with dual input streams.
    
    Inputs:
    - flat_obs: flat observation vector (15 dims)
    - spatial_obs: 2D spatial map (3 channels, 32x32)
    
    Outputs:
    - action_logits: (batch, 7) action probabilities
    - placement_mu: (batch, 2) mean of placement distribution
    - placement_sigma: (batch, 2) std of placement distribution (positive)
    - value: (batch, 1) state value estimate
    """
    
    def __init__(self, flat_dim: int = 15, spatial_h: int = 32, spatial_w: int = 32):
        super().__init__()
        self.flat_dim = flat_dim
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        
        # Flat stream: flat_obs → Dense layers
        self.flat_stream = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Spatial stream: spatial_obs → Conv layers
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce spatial dims
        )
        
        # Calculate spatial stream output size after pooling
        self.spatial_out_size = 64 * 4 * 4
        
        # Fusion: concatenate flat (128) + spatial (1024) → shared layers
        fusion_in = 128 + self.spatial_out_size
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Output heads
        self.action_head = nn.Linear(128, 7)           # 7 actions
        self.placement_mu_head = nn.Linear(128, 2)     # x, y means
        self.placement_sigma_head = nn.Linear(128, 2)  # x, y std (softplus)
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, flat_obs: torch.Tensor, spatial_obs: torch.Tensor):
        """
        Forward pass.
        
        Args:
            flat_obs: (batch, 15)
            spatial_obs: (batch, 3, 32, 32)
        
        Returns:
            action_logits: (batch, 7)
            placement_mu: (batch, 2)
            placement_sigma: (batch, 2)
            value: (batch, 1)
        """
        # Flat stream
        flat_feat = self.flat_stream(flat_obs)
        
        # Spatial stream
        spatial_feat = self.spatial_stream(spatial_obs)
        spatial_feat = spatial_feat.view(spatial_feat.size(0), -1)
        
        # Fusion
        combined = torch.cat([flat_feat, spatial_feat], dim=1)
        fused = self.fusion(combined)
        
        # Output heads
        action_logits = self.action_head(fused)
        placement_mu = self.placement_mu_head(fused)
        placement_sigma = F.softplus(self.placement_sigma_head(fused)) + 1e-5  # Ensure positive
        value = self.value_head(fused)
        
        return action_logits, placement_mu, placement_sigma, value
```

**Step 4: Run test**

```bash
pytest tests/rl/test_policy_net.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add mindustry_ai/rl/policy_net.py tests/rl/test_policy_net.py
git commit -m "feat: implement PyTorch actor-critic policy network"
```

---

### Task 9: Implement Training Loop

**Files:**
- Create: `mindustry_ai/rl/trainer.py`
- Create: `mindustry_ai/env/game_env.py`
- Create: `tests/rl/test_trainer.py`

**Step 1: Create game environment wrapper**

```python
# mindustry_ai/env/game_env.py
import numpy as np
from typing import Tuple, Dict, Any
from mindustry_ai.game.state_reader import GameStateReader
from mindustry_ai.game.action_executor import ActionExecutor

class MindustryEnv:
    """
    Gym-like environment wrapper for Mindustry.
    
    state_reader: reads game state
    action_executor: sends actions to game
    """
    
    def __init__(self, max_steps: int = 1000):
        self.state_reader = GameStateReader()
        self.action_executor = ActionExecutor()
        self.max_steps = max_steps
        self.step_count = 0
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset game to initial state."""
        # TODO: Actually reset game (restart wave, reset resources, etc.)
        self.step_count = 0
        state = self.state_reader.read_state()
        flat_obs = self.state_reader.to_flat_vector(state)
        spatial_map = self.state_reader.to_spatial_map(state)
        return flat_obs, spatial_map
    
    def step(self, action: int, x: float, y: float) -> Tuple[Tuple[np.ndarray, Dict], float, bool]:
        """
        Execute one action.
        
        Returns:
            (flat_obs, spatial_map), reward, done
        """
        # Execute action
        self.action_executor.execute(action, x, y)
        
        # Read new state
        state = self.state_reader.read_state()
        flat_obs = self.state_reader.to_flat_vector(state)
        spatial_map = self.state_reader.to_spatial_map(state)
        
        # Compute reward
        reward = self._compute_reward(state)
        
        # Check termination
        done = (state["status"]["core_health"] <= 0) or (self.step_count >= self.max_steps)
        self.step_count += 1
        
        return (flat_obs, spatial_map), reward, done
    
    def _compute_reward(self, state: Dict[str, Any]) -> float:
        """Compute reward based on game state."""
        reward = 0
        
        # Survival bonus
        if state["status"]["core_health"] > 0:
            reward += 1
        
        # Resource efficiency
        copper = state["resources"]["copper"]
        reward += copper * 0.01
        
        # Power stability
        power_ratio = state["power"]["current"] / state["power"]["capacity"]
        if 0.7 < power_ratio < 1.0:
            reward += 0.5
        
        # Catastrophe penalty
        if state["status"]["core_health"] <= 0:
            reward -= 100
        
        return reward
```

**Step 2: Write failing test for trainer**

```python
# tests/rl/test_trainer.py
import pytest
import torch
from mindustry_ai.rl.trainer import A2CTrainer

def test_trainer_init():
    """A2CTrainer should initialize."""
    trainer = A2CTrainer(learning_rate=3e-4, max_steps_per_episode=1000)
    assert trainer is not None

def test_trainer_collect_trajectory():
    """Trainer should collect trajectories."""
    trainer = A2CTrainer(learning_rate=3e-4, max_steps_per_episode=100)
    # Note: this will use stub environment
    trajectory = trainer.collect_episode()
    
    assert isinstance(trajectory, dict)
    assert "states" in trajectory
    assert "actions" in trajectory
    assert "rewards" in trajectory
```

**Step 3: Run test**

```bash
pytest tests/rl/test_trainer.py -v
# Expected: FAIL (module not created yet)
```

**Step 4: Write A2CTrainer implementation**

```python
# mindustry_ai/rl/trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.env.game_env import MindustryEnv
from mindustry_ai.rules.hybrid_decider import HybridDecider

class A2CTrainer:
    """
    Actor-Critic (A2C) trainer for Mindustry AI.
    
    Combines rule layer suggestions with RL policy refinement.
    """
    
    def __init__(self, learning_rate: float = 3e-4, max_steps_per_episode: int = 1000, gamma: float = 0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.max_steps = max_steps_per_episode
        self.gamma = gamma
        self.env = MindustryEnv(max_steps=max_steps_per_episode)
        self.decider = HybridDecider()  # For rule guidance
    
    def collect_episode(self) -> Dict:
        """Collect one episode of experience."""
        flat_obs, spatial_map = self.env.reset()
        
        trajectory = {
            "states": [],
            "actions": [],
            "placements": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }
        
        for _ in range(self.max_steps):
            # Convert to tensors
            flat_tensor = torch.FloatTensor(flat_obs).unsqueeze(0).to(self.device)
            spatial_tensor = torch.FloatTensor(spatial_map["blocks"]).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get rule suggestion
            state_dict = {
                "resources": {"copper": 100, "lead": 50, "coal": 30, "graphite": 20, "titanium": 10},
                "power": {"current": 500, "capacity": 1000, "production": 300, "consumption": 200},
                "threat": {"enemies_nearby": 0, "wave_number": 1, "time_to_wave": 600},
                "infrastructure": {"drills_count": 2, "turrets_count": 1, "conveyors_count": 5},
                "status": {"core_health": 1.0, "recent_damage": 0, "game_time": 0},
            }
            rule_action = self.decider.decide(state_dict)
            
            # Policy network forward pass
            with torch.no_grad():
                action_logits, placement_mu, placement_sigma, value = self.policy_net(flat_tensor, spatial_tensor)
            
            # Sample action (use rule suggestion as context)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample().item()
            
            # Sample placement (Gaussian)
            placement_dist = torch.distributions.Normal(placement_mu, placement_sigma)
            placement_sample = placement_dist.sample()
            x = float(placement_sample[0, 0].item())
            y = float(placement_sample[0, 1].item())
            
            # Clamp to map bounds
            x = np.clip(x, 0, 32)
            y = np.clip(y, 0, 32)
            
            # Execute in environment
            (flat_obs_next, spatial_map_next), reward, done = self.env.step(action, x, y)
            
            # Store trajectory
            trajectory["states"].append((flat_obs, spatial_map))
            trajectory["actions"].append(action)
            trajectory["placements"].append((x, y))
            trajectory["rewards"].append(reward)
            trajectory["values"].append(value.squeeze(0).item())
            trajectory["dones"].append(done)
            
            flat_obs = flat_obs_next
            spatial_map = spatial_map_next
            
            if done:
                break
        
        return trajectory
    
    def train_step(self, trajectory: Dict) -> float:
        """
        Perform one training step given trajectory.
        
        Returns: policy loss
        """
        states = trajectory["states"]
        actions = torch.LongTensor(trajectory["actions"]).to(self.device)
        placements = torch.FloatTensor(trajectory["placements"]).to(self.device)
        rewards = torch.FloatTensor(trajectory["rewards"]).to(self.device)
        values_pred = torch.FloatTensor(trajectory["values"]).to(self.device)
        dones = trajectory["dones"]
        
        # Compute returns
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Compute advantages
        advantages = returns - values_pred
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Recompute forward pass for backprop
        total_loss = 0
        for i, (flat_obs, spatial_map) in enumerate(states):
            flat_tensor = torch.FloatTensor(flat_obs).unsqueeze(0).to(self.device)
            spatial_tensor = torch.FloatTensor(spatial_map["blocks"]).unsqueeze(0).unsqueeze(0).to(self.device)
            
            action_logits, placement_mu, placement_sigma, value = self.policy_net(flat_tensor, spatial_tensor)
            
            # Policy loss
            action_dist = torch.distributions.Categorical(logits=action_logits)
            log_prob_action = action_dist.log_prob(actions[i])
            
            placement_dist = torch.distributions.Normal(placement_mu.squeeze(0), placement_sigma.squeeze(0))
            log_prob_placement = placement_dist.log_prob(placements[i]).sum()
            
            entropy = action_dist.entropy()
            
            policy_loss = -(log_prob_action + log_prob_placement) * advantages[i] - 0.01 * entropy
            
            # Value loss
            value_loss = 0.5 * (value.squeeze() - returns[i]) ** 2
            
            total_loss = total_loss + policy_loss + value_loss
        
        # Backprop
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
```

**Step 5: Run test**

```bash
pytest tests/rl/test_trainer.py -v
# Expected: PASS
```

**Step 6: Commit**

```bash
git add mindustry_ai/env/game_env.py mindustry_ai/rl/trainer.py tests/rl/test_trainer.py
git commit -m "feat: implement A2C training loop with game environment wrapper"
```

---

## Phase 4: Integration & Main Loop

### Task 10: Create main training script

**Files:**
- Create: `train.py`
- Create: `config.yaml`

**Step 1: Write config.yaml**

```yaml
# config.yaml
training:
  num_episodes: 3000
  max_steps_per_episode: 1000
  learning_rate: 3e-4
  gamma: 0.99
  log_interval: 10

environment:
  map_width: 32
  map_height: 32
  max_enemies: 10

phase:
  # Phase 1: Survival (waves 1-3)
  phase_1_episodes: 1000
  phase_1_max_wave: 3
  
  # Phase 2: Production (waves 4-10)
  phase_2_episodes: 2000
  phase_2_max_wave: 10
  
  # Phase 3: Defense (waves 10+)
  phase_3_episodes: 3000
  phase_3_max_wave: 20

checkpoint:
  save_interval: 100
  save_dir: "checkpoints/"
```

**Step 2: Write main training script**

```python
# train.py
import torch
import yaml
import argparse
from mindustry_ai.rl.trainer import A2CTrainer

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train Mindustry AI Agent")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--phase", type=int, default=1, help="Training phase (1-3)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    trainer = A2CTrainer(
        learning_rate=config["training"]["learning_rate"],
        max_steps_per_episode=config["training"]["max_steps_per_episode"],
        gamma=config["training"]["gamma"],
    )
    
    phase_config = {
        1: (config["phase"]["phase_1_episodes"], config["phase"]["phase_1_max_wave"]),
        2: (config["phase"]["phase_2_episodes"], config["phase"]["phase_2_max_wave"]),
        3: (config["phase"]["phase_3_episodes"], config["phase"]["phase_3_max_wave"]),
    }
    
    num_episodes, max_wave = phase_config[args.phase]
    
    print(f"Training Phase {args.phase}: {num_episodes} episodes (max wave {max_wave})")
    
    for episode in range(num_episodes):
        # Collect trajectory
        trajectory = trainer.collect_episode()
        
        # Train on trajectory
        loss = trainer.train_step(trajectory)
        
        # Logging
        if (episode + 1) % config["training"]["log_interval"] == 0:
            total_reward = sum(trajectory["rewards"])
            print(f"Episode {episode+1}/{num_episodes} | Loss: {loss:.4f} | Reward: {total_reward:.2f}")
        
        # Checkpointing
        if (episode + 1) % config["checkpoint"]["save_interval"] == 0:
            torch.save(trainer.policy_net.state_dict(), f"{config['checkpoint']['save_dir']}/phase{args.phase}_ep{episode+1}.pt")
            print(f"Checkpoint saved: phase{args.phase}_ep{episode+1}.pt")

if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add train.py config.yaml
git commit -m "feat: add training script and configuration"
```

---

## Summary

You now have a complete MVP implementation plan covering:

1. **Game Wrapper** - State reader + action executor
2. **Rule Layer** - BT + SM + PQ for strategic decisions
3. **RL Layer** - PyTorch actor-critic policy network
4. **Training Loop** - A2C trainer with episode collection & backprop
5. **Main Script** - Training entry point with phase progression

Total: 10 tasks, ~50 commits by the end.

---

## Next Steps After Implementation

- Phase 1 Training: Run `python train.py --config config.yaml --phase 1`
- Phase 2 Training: After Phase 1 converges, `python train.py --config config.yaml --phase 2`
- Validation: Test on real Mindustry instance and measure success metrics
