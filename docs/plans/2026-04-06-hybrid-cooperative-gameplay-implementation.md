# Hybrid Cooperative Gameplay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement action queue architecture enabling human + AI cooperative play on same Mindustry map with 1s decision cycle.

**Architecture:** Thread-safe action queue receives both human and AI actions, validator detects conflicts, executor processes FIFO with fallback to rule-based system on NN failure.

**Tech Stack:** Python threading, dataclasses, pytest, httpx (for API communication to be added later)

---

## Task 1: ActionQueue - Thread-Safe FIFO

**Files:**
- Create: `mindustry_ai/coordinator/__init__.py`
- Create: `mindustry_ai/coordinator/action_queue.py`
- Create: `tests/coordinator/__init__.py`
- Create: `tests/coordinator/test_action_queue.py`

**Step 1: Write test for ActionQueue basic operations**

```python
# tests/coordinator/test_action_queue.py
import pytest
from dataclasses import dataclass
from mindustry_ai.coordinator.action_queue import ActionQueue, Action


@dataclass
class Action:
    type: str
    position: tuple
    source: str
    timestamp: float


class TestActionQueue:
    def test_enqueue_dequeue_single_action(self):
        queue = ActionQueue()
        action = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        
        queue.enqueue(action)
        assert queue.size() == 1
        
        dequeued = queue.dequeue()
        assert dequeued.source == "human"
        assert dequeued.type == "PLACE_DRILL"
        assert queue.size() == 0
    
    def test_fifo_order(self):
        queue = ActionQueue()
        a1 = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        a2 = Action(type="PLACE_CONVEYOR", position=(6, 5), source="ai", timestamp=0.5)
        
        queue.enqueue(a1)
        queue.enqueue(a2)
        
        assert queue.dequeue().source == "human"
        assert queue.dequeue().source == "ai"
    
    def test_peek_does_not_remove(self):
        queue = ActionQueue()
        action = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        
        queue.enqueue(action)
        peeked = queue.peek()
        
        assert peeked.type == "PLACE_DRILL"
        assert queue.size() == 1
    
    def test_empty_queue_dequeue_returns_none(self):
        queue = ActionQueue()
        assert queue.dequeue() is None
    
    def test_clear_empties_queue(self):
        queue = ActionQueue()
        queue.enqueue(Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0))
        queue.enqueue(Action(type="PLACE_CONVEYOR", position=(6, 5), source="ai", timestamp=0.5))
        
        queue.clear()
        assert queue.size() == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/coordinator/test_action_queue.py::TestActionQueue::test_enqueue_dequeue_single_action -v`

Expected: FAIL - "ModuleNotFoundError: No module named 'mindustry_ai.coordinator'"

**Step 3: Write ActionQueue implementation**

```python
# mindustry_ai/coordinator/action_queue.py
import threading
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Action:
    type: str
    position: Tuple[int, int]
    source: str
    timestamp: float


class ActionQueue:
    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
    
    def enqueue(self, action: Action) -> None:
        with self._lock:
            self._queue.append(action)
    
    def dequeue(self) -> Optional[Action]:
        with self._lock:
            if len(self._queue) == 0:
                return None
            return self._queue.pop(0)
    
    def peek(self) -> Optional[Action]:
        with self._lock:
            if len(self._queue) == 0:
                return None
            return self._queue[0]
    
    def size(self) -> int:
        with self._lock:
            return len(self._queue)
    
    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
```

```python
# mindustry_ai/coordinator/__init__.py
from .action_queue import ActionQueue, Action

__all__ = ["ActionQueue", "Action"]
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/coordinator/test_action_queue.py -v`

Expected: PASS - All 5 tests passing

**Step 5: Commit**

```bash
cd /home/pedro/repo/mindustry-ia-inteface-mod
git add mindustry_ai/coordinator/ tests/coordinator/
git commit -m "feat: implement thread-safe ActionQueue with FIFO ordering"
```

---

## Task 2: ActionValidator - Conflict Detection

**Files:**
- Create: `mindustry_ai/coordinator/validator.py`
- Create: `tests/coordinator/test_validator.py`

**Step 1: Write tests for ActionValidator**

```python
# tests/coordinator/test_validator.py
import pytest
from dataclasses import dataclass
from mindustry_ai.coordinator.action_queue import Action
from mindustry_ai.coordinator.validator import ActionValidator


@dataclass
class GameState:
    structures: dict  # {(x, y): "DRILL", ...}
    resources: dict  # {"copper": 100, ...}
    map_width: int
    map_height: int


class TestActionValidator:
    def test_validate_action_on_empty_cell(self):
        validator = ActionValidator()
        state = GameState(structures={}, resources={"copper": 100}, map_width=20, map_height=20)
        action = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        
        valid, reason = validator.validate(action, state)
        assert valid is True
    
    def test_validate_action_on_occupied_cell_fails(self):
        validator = ActionValidator()
        state = GameState(structures={(5, 5): "DRILL"}, resources={"copper": 100}, map_width=20, map_height=20)
        action = Action(type="PLACE_CONVEYOR", position=(5, 5), source="ai", timestamp=0.0)
        
        valid, reason = validator.validate(action, state)
        assert valid is False
        assert "occupied" in reason.lower()
    
    def test_validate_action_insufficient_resources(self):
        validator = ActionValidator()
        state = GameState(structures={}, resources={"copper": 5}, map_width=20, map_height=20)
        action = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        
        valid, reason = validator.validate(action, state)
        assert valid is False
        assert "resource" in reason.lower()
    
    def test_detect_conflict_same_position(self):
        validator = ActionValidator()
        a1 = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        a2 = Action(type="PLACE_CONVEYOR", position=(5, 5), source="ai", timestamp=0.5)
        
        conflicts = validator.detect_conflict(a1, a2)
        assert conflicts is True
    
    def test_detect_no_conflict_different_positions(self):
        validator = ActionValidator()
        a1 = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        a2 = Action(type="PLACE_CONVEYOR", position=(6, 5), source="ai", timestamp=0.5)
        
        conflicts = validator.detect_conflict(a1, a2)
        assert conflicts is False
```

**Step 2: Run test to verify it fails**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/coordinator/test_validator.py::TestActionValidator::test_validate_action_on_empty_cell -v`

Expected: FAIL - "ModuleNotFoundError: No module named 'mindustry_ai.coordinator.validator'"

**Step 3: Write ActionValidator implementation**

```python
# mindustry_ai/coordinator/validator.py
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass
class GameState:
    structures: dict
    resources: dict
    map_width: int
    map_height: int


class ActionValidator:
    RESOURCE_COSTS = {
        "PLACE_DRILL": {"copper": 20},
        "PLACE_CONVEYOR": {"copper": 10},
        "PLACE_CONTAINER": {"copper": 50},
    }
    
    def validate(self, action, state: GameState) -> Tuple[bool, str]:
        if not self._is_in_bounds(action.position, state):
            return False, f"Position {action.position} out of bounds"
        
        if not self._is_cell_empty(action.position, state):
            return False, f"Position {action.position} already occupied"
        
        if not self._has_resources(action.type, state):
            return False, f"Insufficient resources for {action.type}"
        
        return True, "OK"
    
    def detect_conflict(self, action1, action2) -> bool:
        return action1.position == action2.position
    
    def _is_in_bounds(self, position: Tuple[int, int], state: GameState) -> bool:
        x, y = position
        return 0 <= x < state.map_width and 0 <= y < state.map_height
    
    def _is_cell_empty(self, position: Tuple[int, int], state: GameState) -> bool:
        return position not in state.structures
    
    def _has_resources(self, action_type: str, state: GameState) -> bool:
        if action_type not in self.RESOURCE_COSTS:
            return True
        
        required = self.RESOURCE_COSTS[action_type]
        for resource, amount in required.items():
            if state.resources.get(resource, 0) < amount:
                return False
        
        return True
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/coordinator/test_validator.py -v`

Expected: PASS - All 5 tests passing

**Step 5: Commit**

```bash
cd /home/pedro/repo/mindustry-ia-inteface-mod
git add mindustry_ai/coordinator/validator.py tests/coordinator/test_validator.py
git commit -m "feat: implement ActionValidator with conflict detection"
```

---

## Task 3: InferenceEngine - NN Inference with Fallback

**Files:**
- Create: `mindustry_ai/rl/inference.py`
- Create: `tests/rl/test_inference.py`

**Step 1: Write tests for InferenceEngine**

```python
# tests/rl/test_inference.py
import pytest
import torch
from unittest.mock import Mock, patch
from mindustry_ai.rl.inference import InferenceEngine
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.coordinator.action_queue import Action
from mindustry_ai.rules.hybrid_decider import HybridDecider


class TestInferenceEngine:
    def test_inference_engine_init(self):
        policy_net = PolicyNetwork(flat_dim=15, spatial_h=16, spatial_w=16)
        engine = InferenceEngine(policy_net=policy_net)
        
        assert engine.policy_net is not None
        assert engine.fallback_decider is None
    
    def test_infer_returns_action(self):
        policy_net = PolicyNetwork(flat_dim=15, spatial_h=16, spatial_w=16)
        engine = InferenceEngine(policy_net=policy_net)
        
        flat_state = torch.randn(1, 15)
        spatial_state = torch.randn(1, 1, 16, 16)
        
        action = engine.infer(flat_state, spatial_state)
        
        assert isinstance(action, Action)
        assert action.source == "ai"
        assert hasattr(action, "type")
        assert hasattr(action, "position")
    
    def test_infer_with_fallback_on_error(self):
        policy_net = PolicyNetwork(flat_dim=15, spatial_h=16, spatial_w=16)
        fallback_decider = Mock(spec=HybridDecider)
        fallback_decider.decide.return_value = Action(
            type="PLACE_DRILL", position=(5, 5), source="ai", timestamp=0.0
        )
        
        engine = InferenceEngine(policy_net=policy_net, fallback_decider=fallback_decider)
        
        flat_state = torch.randn(1, 15)
        spatial_state = torch.randn(1, 1, 16, 16)
        
        with patch.object(engine.policy_net, "forward", side_effect=RuntimeError("NN error")):
            action = engine.infer(flat_state, spatial_state)
            
            assert action.type == "PLACE_DRILL"
            fallback_decider.decide.assert_called_once()
    
    def test_load_checkpoint(self):
        policy_net = PolicyNetwork(flat_dim=15, spatial_h=16, spatial_w=16)
        engine = InferenceEngine(policy_net=policy_net)
        
        checkpoint_path = "/tmp/test_checkpoint.pt"
        torch.save(policy_net.state_dict(), checkpoint_path)
        
        engine.load_checkpoint(checkpoint_path)
        
        assert engine.policy_net is not None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/rl/test_inference.py::TestInferenceEngine::test_inference_engine_init -v`

Expected: FAIL - "ModuleNotFoundError: No module named 'mindustry_ai.rl.inference'"

**Step 3: Write InferenceEngine implementation**

```python
# mindustry_ai/rl/inference.py
import torch
import logging
from typing import Optional, Tuple
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.rules.hybrid_decider import HybridDecider
from mindustry_ai.coordinator.action_queue import Action


logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, policy_net: PolicyNetwork, fallback_decider: Optional[HybridDecider] = None):
        self.policy_net = policy_net
        self.fallback_decider = fallback_decider
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
    
    def infer(self, flat_state: torch.Tensor, spatial_state: torch.Tensor) -> Action:
        try:
            with torch.no_grad():
                action_logits, placement_mu, placement_sigma, value = self.policy_net(flat_state, spatial_state)
            
            action_idx = torch.argmax(action_logits, dim=1).item()
            position = self._decode_position(placement_mu[0].cpu().numpy())
            
            action_type = self._idx_to_action_type(action_idx)
            
            return Action(
                type=action_type,
                position=position,
                source="ai",
                timestamp=0.0
            )
        except Exception as e:
            logger.error(f"NN inference failed: {e}")
            
            if self.fallback_decider is not None:
                return self.fallback_decider.decide()
            
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _decode_position(self, placement_mu: list) -> Tuple[int, int]:
        x = max(0, min(19, int(placement_mu[0] * 20)))
        y = max(0, min(19, int(placement_mu[1] * 20)))
        return (x, y)
    
    def _idx_to_action_type(self, idx: int) -> str:
        action_map = {
            0: "PLACE_DRILL",
            1: "PLACE_CONVEYOR",
            2: "PLACE_CONTAINER",
            3: "WAIT",
            4: "REMOVE",
            5: "BUILD_TURRET",
            6: "PLACE_POWER",
        }
        return action_map.get(idx, "WAIT")
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/rl/test_inference.py -v`

Expected: PASS - All 3 tests passing

**Step 5: Commit**

```bash
cd /home/pedro/repo/mindustry-ia-inteface-mod
git add mindustry_ai/rl/inference.py tests/rl/test_inference.py
git commit -m "feat: implement InferenceEngine with checkpoint loading and fallback"
```

---

## Task 4: GameAPIClient - Mindustry Communication (Stub)

**Files:**
- Create: `mindustry_ai/game/api_client.py`
- Create: `tests/game/test_api_client.py`

**Step 1: Write tests for GameAPIClient**

```python
# tests/game/test_api_client.py
import pytest
from unittest.mock import Mock, patch
from mindustry_ai.game.api_client import GameAPIClient


class TestGameAPIClient:
    def test_api_client_init(self):
        client = GameAPIClient(host="localhost", port=8080)
        
        assert client.host == "localhost"
        assert client.port == 8080
        assert client.is_connected() is False
    
    def test_connect_success(self):
        client = GameAPIClient(host="localhost", port=8080)
        
        with patch.object(client, "_establish_connection", return_value=True):
            client.connect()
            assert client.is_connected() is True
    
    def test_disconnect(self):
        client = GameAPIClient(host="localhost", port=8080)
        client._connected = True
        
        client.disconnect()
        assert client.is_connected() is False
    
    def test_get_game_state_returns_dict(self):
        client = GameAPIClient(host="localhost", port=8080)
        client._connected = True
        
        mock_state = {
            "resources": {"copper": 100},
            "health": 0.8,
            "structures": []
        }
        
        with patch.object(client, "_fetch_state", return_value=mock_state):
            state = client.get_game_state()
            
            assert "resources" in state
            assert "health" in state
    
    def test_execute_action_calls_api(self):
        client = GameAPIClient(host="localhost", port=8080)
        client._connected = True
        
        from mindustry_ai.coordinator.action_queue import Action
        action = Action(type="PLACE_DRILL", position=(5, 5), source="ai", timestamp=0.0)
        
        with patch.object(client, "_send_action", return_value=True):
            result = client.execute_action(action)
            
            assert result is True
```

**Step 2: Run test to verify it fails**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/game/test_api_client.py::TestGameAPIClient::test_api_client_init -v`

Expected: FAIL - "ModuleNotFoundError: No module named 'mindustry_ai.game.api_client'"

**Step 3: Write GameAPIClient stub (for now)**

```python
# mindustry_ai/game/api_client.py
import logging
from typing import Dict, Optional, Any
from mindustry_ai.coordinator.action_queue import Action


logger = logging.getLogger(__name__)


class GameAPIClient:
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self._connected = False
    
    def connect(self) -> bool:
        try:
            self._establish_connection()
            self._connected = True
            logger.info(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self) -> None:
        self._connected = False
        logger.info("Disconnected")
    
    def is_connected(self) -> bool:
        return self._connected
    
    def get_game_state(self) -> Dict[str, Any]:
        if not self._connected:
            raise RuntimeError("Not connected to game")
        
        return self._fetch_state()
    
    def execute_action(self, action: Action) -> bool:
        if not self._connected:
            raise RuntimeError("Not connected to game")
        
        return self._send_action(action)
    
    def _establish_connection(self) -> None:
        pass
    
    def _fetch_state(self) -> Dict[str, Any]:
        return {
            "resources": {},
            "health": 1.0,
            "structures": [],
            "map_width": 20,
            "map_height": 20,
        }
    
    def _send_action(self, action: Action) -> bool:
        logger.info(f"Execute action: {action}")
        return True
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/game/test_api_client.py -v`

Expected: PASS - All 5 tests passing

**Step 5: Commit**

```bash
cd /home/pedro/repo/mindustry-ia-inteface-mod
git add mindustry_ai/game/api_client.py tests/game/test_api_client.py
git commit -m "feat: stub GameAPIClient for Mindustry server communication"
```

---

## Task 5: HybridGameLoop - Main Orchestration

**Files:**
- Create: `mindustry_ai/coordinator/game_loop.py`
- Create: `tests/coordinator/test_game_loop.py`

**Step 1: Write tests for HybridGameLoop**

```python
# tests/coordinator/test_game_loop.py
import pytest
import time
import threading
from unittest.mock import Mock, patch
from mindustry_ai.coordinator.game_loop import HybridGameLoop
from mindustry_ai.coordinator.action_queue import ActionQueue, Action


class TestHybridGameLoop:
    def test_game_loop_init(self):
        api_client = Mock()
        policy_net = Mock()
        action_queue = ActionQueue()
        
        loop = HybridGameLoop(api_client=api_client, policy_net=policy_net, action_queue=action_queue)
        
        assert loop.api_client is not None
        assert loop.action_queue is not None
        assert loop.running is False
    
    def test_game_loop_start_stop(self):
        api_client = Mock()
        api_client.is_connected.return_value = True
        api_client.get_game_state.return_value = {
            "resources": {"copper": 100},
            "health": 1.0,
            "structures": {}
        }
        
        policy_net = Mock()
        action_queue = ActionQueue()
        
        loop = HybridGameLoop(api_client=api_client, policy_net=policy_net, action_queue=action_queue)
        
        loop.start()
        assert loop.running is True
        
        time.sleep(0.1)
        loop.stop()
        
        assert loop.running is False
    
    def test_game_loop_executes_queued_actions(self):
        api_client = Mock()
        api_client.is_connected.return_value = True
        api_client.get_game_state.return_value = {
            "resources": {"copper": 100},
            "health": 1.0,
            "structures": {}
        }
        api_client.execute_action.return_value = True
        
        policy_net = Mock()
        action_queue = ActionQueue()
        
        loop = HybridGameLoop(
            api_client=api_client,
            policy_net=policy_net,
            action_queue=action_queue,
            cycle_interval=0.01
        )
        
        action = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        action_queue.enqueue(action)
        
        loop.start()
        time.sleep(0.05)
        loop.stop()
        
        api_client.execute_action.assert_called()
```

**Step 2: Run test to verify it fails**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/coordinator/test_game_loop.py::TestHybridGameLoop::test_game_loop_init -v`

Expected: FAIL - "ModuleNotFoundError: No module named 'mindustry_ai.coordinator.game_loop'"

**Step 3: Write HybridGameLoop implementation**

```python
# mindustry_ai/coordinator/game_loop.py
import threading
import time
import logging
from typing import Optional
import torch

from mindustry_ai.game.api_client import GameAPIClient
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.rl.inference import InferenceEngine
from mindustry_ai.coordinator.action_queue import ActionQueue
from mindustry_ai.coordinator.validator import ActionValidator


logger = logging.getLogger(__name__)


class HybridGameLoop:
    def __init__(
        self,
        api_client: GameAPIClient,
        policy_net: PolicyNetwork,
        action_queue: ActionQueue,
        inference_engine: Optional[InferenceEngine] = None,
        cycle_interval: float = 1.0,
    ):
        self.api_client = api_client
        self.policy_net = policy_net
        self.action_queue = action_queue
        self.inference_engine = inference_engine or InferenceEngine(policy_net)
        self.validator = ActionValidator()
        
        self.cycle_interval = cycle_interval
        self.running = False
        self._main_thread = None
        self._inference_thread = None
    
    def start(self) -> None:
        if self.running:
            logger.warning("Game loop already running")
            return
        
        self.running = True
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        
        self._main_thread.start()
        self._inference_thread.start()
        
        logger.info("Game loop started")
    
    def stop(self) -> None:
        self.running = False
        
        if self._main_thread:
            self._main_thread.join(timeout=2.0)
        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)
        
        logger.info("Game loop stopped")
    
    def _main_loop(self) -> None:
        while self.running:
            try:
                if not self.api_client.is_connected():
                    logger.warning("API not connected, waiting...")
                    time.sleep(0.5)
                    continue
                
                game_state = self.api_client.get_game_state()
                
                while not self.action_queue.size() == 0:
                    action = self.action_queue.peek()
                    
                    valid, reason = self.validator.validate(action, game_state)
                    if valid:
                        self.action_queue.dequeue()
                        self.api_client.execute_action(action)
                        logger.debug(f"Executed {action.source} action: {action.type}")
                    else:
                        self.action_queue.dequeue()
                        logger.warning(f"Invalid action discarded: {reason}")
                
                time.sleep(self.cycle_interval)
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(0.5)
    
    def _inference_loop(self) -> None:
        while self.running:
            try:
                if not self.api_client.is_connected():
                    time.sleep(0.5)
                    continue
                
                game_state = self.api_client.get_game_state()
                
                flat_state = self._dict_to_tensor(game_state)
                spatial_state = torch.randn(1, 1, 16, 16)
                
                action = self.inference_engine.infer(flat_state, spatial_state)
                self.action_queue.enqueue(action)
                
                time.sleep(self.cycle_interval)
            
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                time.sleep(self.cycle_interval)
    
    def _dict_to_tensor(self, game_state: dict) -> torch.Tensor:
        resources = list(game_state.get("resources", {}).values())[:15]
        resources += [0] * (15 - len(resources))
        return torch.tensor([resources], dtype=torch.float32)
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/coordinator/test_game_loop.py -v`

Expected: PASS - All 3 tests passing

**Step 5: Commit**

```bash
cd /home/pedro/repo/mindustry-ia-inteface-mod
git add mindustry_ai/coordinator/game_loop.py tests/coordinator/test_game_loop.py
git commit -m "feat: implement HybridGameLoop with threaded inference and execution"
```

---

## Task 6: CooperativePlayManager - Entry Point

**Files:**
- Create: `mindustry_ai/hybrid/cooperative_play.py`
- Create: `tests/test_cooperative_play.py`

**Step 1: Write tests**

```python
# tests/test_cooperative_play.py
import pytest
from unittest.mock import Mock, patch
from mindustry_ai.hybrid.cooperative_play import CooperativePlayManager


class TestCooperativePlayManager:
    def test_manager_init(self):
        manager = CooperativePlayManager(
            host="localhost",
            port=8080,
            model_checkpoint="model.pt"
        )
        
        assert manager.host == "localhost"
        assert manager.port == 8080
    
    def test_manager_start_stop(self):
        manager = CooperativePlayManager(
            host="localhost",
            port=8080,
            model_checkpoint="model.pt"
        )
        
        with patch.object(manager, "game_loop") as mock_loop:
            manager.start()
            mock_loop.start.assert_called_once()
            
            manager.stop()
            mock_loop.stop.assert_called_once()
```

**Step 2: Run test**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/test_cooperative_play.py::TestCooperativePlayManager::test_manager_init -v`

Expected: FAIL

**Step 3: Write CooperativePlayManager**

```python
# mindustry_ai/hybrid/cooperative_play.py
import logging
from mindustry_ai.game.api_client import GameAPIClient
from mindustry_ai.rl.policy_net import PolicyNetwork
from mindustry_ai.rl.inference import InferenceEngine
from mindustry_ai.coordinator.action_queue import ActionQueue
from mindustry_ai.coordinator.game_loop import HybridGameLoop


logger = logging.getLogger(__name__)


class CooperativePlayManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        model_checkpoint: str = "model.pt",
        flat_dim: int = 15,
        spatial_h: int = 16,
        spatial_w: int = 16,
    ):
        self.host = host
        self.port = port
        self.model_checkpoint = model_checkpoint
        
        self.api_client = GameAPIClient(host=host, port=port)
        self.policy_net = PolicyNetwork(flat_dim=flat_dim, spatial_h=spatial_h, spatial_w=spatial_w)
        self.action_queue = ActionQueue()
        self.inference_engine = InferenceEngine(policy_net=self.policy_net)
        self.game_loop = HybridGameLoop(
            api_client=self.api_client,
            policy_net=self.policy_net,
            action_queue=self.action_queue,
            inference_engine=self.inference_engine,
        )
        
        self._load_model()
    
    def start(self) -> None:
        logger.info(f"Connecting to Mindustry at {self.host}:{self.port}")
        
        if not self.api_client.connect():
            raise RuntimeError(f"Failed to connect to {self.host}:{self.port}")
        
        self.game_loop.start()
        logger.info("Cooperative play started")
    
    def stop(self) -> None:
        self.game_loop.stop()
        self.api_client.disconnect()
        logger.info("Cooperative play stopped")
    
    def _load_model(self) -> None:
        try:
            self.inference_engine.load_checkpoint(self.model_checkpoint)
            logger.info(f"Model loaded from {self.model_checkpoint}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using untrained network.")


# Entry point for CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cooperative play with Mindustry")
    parser.add_argument("--host", default="localhost", help="Mindustry server host")
    parser.add_argument("--port", type=int, default=8080, help="Mindustry server port")
    parser.add_argument("--model", default="model.pt", help="Path to trained model checkpoint")
    
    args = parser.parse_args()
    
    manager = CooperativePlayManager(host=args.host, port=args.port, model_checkpoint=args.model)
    
    try:
        manager.start()
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        manager.stop()
```

```python
# mindustry_ai/hybrid/__init__.py
from .cooperative_play import CooperativePlayManager

__all__ = ["CooperativePlayManager"]
```

**Step 4: Run tests**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/test_cooperative_play.py -v`

Expected: PASS - All tests

**Step 5: Commit**

```bash
cd /home/pedro/repo/mindustry-ia-inteface-mod
git add mindustry_ai/hybrid/ tests/test_cooperative_play.py
git commit -m "feat: implement CooperativePlayManager entry point for hybrid gameplay"
```

---

## Task 7: Integration Test + Full Verification

**Files:**
- Create: `tests/test_integration_cooperative.py`

**Step 1: Write integration test**

```python
# tests/test_integration_cooperative.py
import pytest
import time
from unittest.mock import Mock, patch
from mindustry_ai.hybrid.cooperative_play import CooperativePlayManager
from mindustry_ai.coordinator.action_queue import Action


class TestCooperativePlayIntegration:
    def test_full_cooperative_play_cycle(self):
        with patch("mindustry_ai.game.api_client.GameAPIClient") as MockAPIClient:
            mock_api = Mock()
            mock_api.is_connected.return_value = True
            mock_api.connect.return_value = True
            mock_api.get_game_state.return_value = {
                "resources": {"copper": 100, "lead": 50},
                "health": 1.0,
                "structures": {},
                "map_width": 20,
                "map_height": 20,
            }
            mock_api.execute_action.return_value = True
            
            MockAPIClient.return_value = mock_api
            
            manager = CooperativePlayManager(
                host="localhost",
                port=8080,
                model_checkpoint="/tmp/dummy.pt"
            )
            
            with patch.object(manager.api_client, "connect", return_value=True):
                with patch.object(manager.api_client, "is_connected", return_value=True):
                    with patch.object(manager.api_client, "get_game_state") as mock_get_state:
                        mock_get_state.return_value = {
                            "resources": {"copper": 100},
                            "health": 1.0,
                            "structures": {},
                            "map_width": 20,
                            "map_height": 20,
                        }
                        
                        manager.start()
                        
                        action = Action(
                            type="PLACE_DRILL",
                            position=(5, 5),
                            source="human",
                            timestamp=0.0
                        )
                        manager.game_loop.action_queue.enqueue(action)
                        
                        time.sleep(0.2)
                        manager.stop()
                        
                        assert manager.api_client.execute_action.called
```

**Step 2: Run integration test**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/test_integration_cooperative.py -v`

Expected: PASS

**Step 3: Run ALL tests to verify nothing broke**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && .venv/bin/python -m pytest tests/ -v`

Expected: All tests pass (40+ tests including new ones)

**Step 4: Commit**

```bash
cd /home/pedro/repo/mindustry-ia-inteface-mod
git add tests/test_integration_cooperative.py
git commit -m "test: add integration test for cooperative play"
```

---

## Summary

**Total: 7 Tasks**
- Task 1: ActionQueue (FIFO thread-safe)
- Task 2: ActionValidator (conflict detection)
- Task 3: InferenceEngine (NN + fallback)
- Task 4: GameAPIClient (stub for API)
- Task 5: HybridGameLoop (main orchestration)
- Task 6: CooperativePlayManager (entry point)
- Task 7: Integration test + verification

**After completion:**
- 50+ tests passing
- Ready for actual Mindustry API implementation
- Entry point: `python mindustry_ai/hybrid/cooperative_play.py --model model.pt`

---

## Next Phase After Implementation

Once all tasks pass:
1. Implement actual Mindustry Server API client (WebSocket/HTTP)
2. Create Mindustry mod to expose server API
3. Live testing with actual Mindustry instance
4. Performance optimization and profiling
