# Hybrid Cooperative Gameplay Design
**Date:** 2026-04-06  
**Author:** Sisyphus  
**Status:** Approved by User

## Vision
Enable trained RL model to play cooperatively with human player on the same Mindustry map. Human and AI operate autonomously with coordination via action queue, allowing seamless hybrid gameplay where both contribute to base building and defense.

---

## Requirements

### Functional
- AI makes decisions autonomously every 1 second using trained NN
- Human input (clicks, keyboard) queued alongside AI actions
- Actions executed in FIFO order with conflict detection
- Real-time state reading from Mindustry (via Server API)
- Fallback to rule-based HybridDecider if NN fails

### Non-Functional
- **Latency**: <2s from decision to execution
- **Throughput**: Process 2-5 actions/second from queue
- **Robustness**: Graceful degradation if API disconnects
- **Debuggability**: Log all actions, decisions, conflicts

---

## Architecture

### High-Level Data Flow

```
MINDUSTRY GAME STATE
    ↓ (read via API)
GAME STATE READER (GameStateReader adapted)
    ↓
ACTION QUEUE COORDINATOR
    ├─ HUMAN I/O (clicks, keyboard)
    └─ RL INFERENCE (PolicyNetwork @ T=1s, T=2s, ...)
    ↓
ACTION VALIDATOR (conflict detection)
    ↓
ACTION EXECUTOR (execute via API)
    ↓
MINDUSTRY GAME (state updated)
```

### Component Breakdown

#### 1. **GameAPIClient** (NEW)
- Communicates with Mindustry server (HTTP REST or WebSocket)
- Methods:
  - `get_game_state()` → GameState dict (resources, health, map, units)
  - `execute_action(action)` → bool (success/failure)
  - `connect()`, `disconnect()`, `is_connected()`
- Location: `mindustry_ai/game/api_client.py`

#### 2. **ActionQueue** (NEW)
- Thread-safe FIFO queue for actions from human + AI
- Methods:
  - `enqueue(source: "human"|"ai", action: Action)` → None
  - `dequeue() → (source, Action)`
  - `peek() → (source, Action)`
  - `size() → int`
  - `clear() → None`
- Mutex lock for thread safety
- Location: `mindustry_ai/coordinator/action_queue.py`

#### 3. **ActionValidator** (NEW)
- Detects conflicts before execution
- Methods:
  - `validate(action, game_state) → (valid: bool, reason: str)`
  - `detect_conflict(action1, action2) → bool`
- Rules:
  - Can't place 2 structures on same cell
  - Can't remove structure being used by other action
  - Resource checks (enough copper, power, etc.)
- Location: `mindustry_ai/coordinator/validator.py`

#### 4. **InferenceEngine** (NEW)
- Wraps PolicyNetwork for live inference
- Methods:
  - `load_checkpoint(path) → None`
  - `infer(game_state) → Action`
  - `set_fallback_strategy(decider: HybridDecider) → None`
- Runs every 1 second in background thread
- Falls back to HybridDecider on error
- Location: `mindustry_ai/rl/inference.py`

#### 5. **HybridGameLoop** (NEW)
- Orchestrates all components
- Threads:
  1. **MainLoop**: State reader every 1s, validator, executor
  2. **InferenceThread**: AI decision every 1s
  3. **IOThread**: Listen for human input (keyboard/mouse)
- Location: `mindustry_ai/coordinator/game_loop.py`

#### 6. **GameStateReader** (ADAPT)
- Read from API instead of simulation
- Update observation format for live game
- Location: `mindustry_ai/game/state_reader.py` (modify)

---

## Data Structures

### Action
```python
@dataclass
class Action:
    type: str                    # "PLACE_DRILL", "PLACE_CONVEYOR", etc.
    position: Tuple[int, int]   # (x, y)
    metadata: Dict[str, Any]    # additional params
    timestamp: float            # when action was created
    source: str                 # "human" or "ai"
```

### GameState
```python
@dataclass
class GameState:
    resources: Dict[str, float]  # {"copper": 100, "lead": 50, ...}
    health: float                # core health %
    map_width: int
    map_height: int
    structures: List[Structure]  # existing buildings
    units: List[Unit]            # enemy waves
    timestamp: float
```

---

## Execution Flow (1 Second Cycle)

```
T=0.0s: 
  - Start of cycle
  - Human clicks or AI adds action to queue

T=0.5s:
  - RL Inference thread calls PolicyNetwork
  - Generates next action for time T=1.0s

T=1.0s:
  - MainLoop reads game state from API
  - Dequeues action from queue
  - Validator checks: valid? conflicts?
  - If valid: execute via API
  - If invalid: log, discard, continue
  - Repeat until queue empty or conflicts unresolved
  - MainLoop goes idle until T=2.0s

T=2.0s:
  - Cycle repeats
```

---

## Integration with MVP

### Reusable Components
- `PolicyNetwork` - load trained checkpoint, call forward()
- `HybridDecider` - use as fallback strategy
- `ActionExecutor` - adapt to use API

### New Dependencies
- `httpx` or `websockets` for API communication
- `threading` for concurrent inference + I/O
- `dataclasses` for type safety

---

## Error Handling

| Error | Handling |
|---|---|
| API disconnected | Pause all inference, retry connection, log |
| Invalid action | Remove from queue, continue |
| NN inference fails | Fall back to HybridDecider, log |
| Conflicting actions | Validator removes older action, prioritizes human |
| Timeout on execute | Retry once, then discard |

---

## Testing Strategy

1. **Unit Tests**: ActionQueue, Validator, InferenceEngine (mocked API)
2. **Integration Tests**: GameLoop with mock GameStateReader
3. **Manual Tests**: Connect to local Mindustry server, observe behavior

---

## Success Criteria

- ✅ AI runs 1 decision/second without blocking human input
- ✅ Actions execute without crashes or conflicts
- ✅ Fallback to HybridDecider on NN failure
- ✅ Human can play freely alongside AI
- ✅ All actions logged for debugging
- ✅ <2s latency from decision to visible effect

---

## File Changes Summary

| File | Type | Purpose |
|---|---|---|
| `mindustry_ai/game/api_client.py` | NEW | Mindustry server communication |
| `mindustry_ai/game/state_reader.py` | MODIFY | Adapt to use API |
| `mindustry_ai/rl/inference.py` | NEW | Live NN inference + fallback |
| `mindustry_ai/coordinator/action_queue.py` | NEW | Thread-safe FIFO queue |
| `mindustry_ai/coordinator/validator.py` | NEW | Conflict detection |
| `mindustry_ai/coordinator/game_loop.py` | NEW | Main orchestration |
| `mindustry_ai/hybrid/cooperative_play.py` | NEW | Entry point |
| `tests/coordinator/test_action_queue.py` | NEW | Queue tests |
| `tests/coordinator/test_validator.py` | NEW | Validator tests |
| `tests/coordinator/test_game_loop.py` | NEW | GameLoop tests |
| `tests/rl/test_inference.py` | NEW | Inference tests |

---

## Next Phase
Invoke `writing-plans` skill to create detailed implementation plan with task breakdown and success criteria per task.
