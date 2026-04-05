# Mindustry AI Agent - MVP Design Document

**Date:** 2026-04-05  
**Scope:** Mining + Logistics + Defense (First Milestone)  
**Approach:** Hybrid Rule-Based + RL Refinement  

---

## 1. Executive Summary

Build an AI agent that plays Mindustry by:
1. **Mining resources** and transporting them to core via conveyors
2. **Managing power grid** by expanding generation as needed
3. **Defending against waves** by strategically placing turrets

The agent uses a **hybrid architecture**: hand-coded rules (Behavior Tree + State Machine + Priority Queue) handle high-level strategy, while an **RL policy (PyTorch)** learns to optimize placements, timing, and refinements.

**Success Metrics:**
- Survive waves 1-10
- Accumulate 1000+ resources per minute at stable state
- Efficiency bonus for minimal waste/idle time
- Sequential training phases for stable learning

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────┐
│         Mindustry Game Instance         │
│  (Screen + Process Memory)              │
└────────────────┬────────────────────────┘
                 │ Hybrid: CV + Memory Read
                 ↓
┌─────────────────────────────────────────┐
│     Game Wrapper / Bot Controller       │
│  (PyAutoGUI + Screenshot OCR + Memread) │
└────────────────┬────────────────────────┘
                 │ 60 FPS game state updates
                 ↓
┌─────────────────────────────────────────┐
│        AI Agent Main Loop               │
├─────────────────────────────────────────┤
│ Rule Layer (Hybrid Decision System)     │
│  • Behavior Tree (hierarchical logic)   │
│  • State Machine (discrete states)      │
│  • Priority Queue (dynamic priorities)  │
├─────────────────────────────────────────┤
│ RL Layer (Policy Network)               │
│  • Observation: flat vector + 2D map   │
│  • Action: high-level + granular       │
│  • Network: Actor-Critic (PyTorch)     │
└─────────────────────────────────────────┘
```

**Per-Tick Flow:**
1. Read game state (screenshot analysis + memory introspection)
2. Rule layer decides primary action (MINE, DEFEND, BUILD, OPTIMIZE)
3. RL layer refines action (placement coordinates, timing adjustments)
4. Execute action via bot controller (send inputs to game)
5. Collect reward (survival points, resource bonuses, efficiency metrics)
6. Update policy network (backprop gradient if in training mode)

---

## 3. Rule Layer: Hybrid Decision System

### 3.1 Behavior Tree (Hierarchical Constraints)

The BT ensures the AI doesn't make catastrophic mistakes (e.g., ignoring enemies, running out of power).

```
Root Selector
├─ Threat Assessment (highest priority)
│  ├─ Enemies detected within range?
│  │  ├─ YES → Activate DEFEND sequence
│  │  │  ├─ Check turret coverage
│  │  │  ├─ Prioritize placement
│  │  │  └─ Execute PLACE_TURRET
│  │  └─ NO → Proceed
│  └─
├─ Energy Management
│  ├─ Power ratio < 80%?
│  │  ├─ YES → Activate EXPAND_GENERATION
│  │  │  ├─ Check available space
│  │  │  ├─ Choose generator type (combustion/steam/reactor)
│  │  │  └─ Execute BUILD_GENERATOR
│  │  └─ NO → Proceed
│  └─
├─ Resource Production
│  ├─ Copper OR Lead < 100 units?
│  │  ├─ YES → Activate EXPAND_MINING
│  │  │  ├─ Analyze ore distribution
│  │  │  ├─ Place drills optimally
│  │  │  └─ Execute PLACE_DRILL
│  │  └─ NO → Proceed
│  └─
└─ Optimization (lowest priority)
   ├─ All systems stable?
   │  ├─ YES → OPTIMIZE_CHAINS
   │  │  ├─ Analyze production inefficiency
   │  │  ├─ Add conveyor shortcuts
   │  │  └─ Execute BUILD_CONVEYOR
   │  └─ NO → WAIT
```

**BT Properties:**
- Runs every 10 ticks (update constraint check)
- Leaf nodes return SUCCESS/FAIL based on feasibility
- Parent selectors use first-success (top-down priority)
- Enables RL to refine within successful branches

### 3.2 State Machine (Discrete States)

States represent the AI's operational mode:

```
States: {MINING, CRAFTING, ENERGY, DEFENSE, IDLE}

Transitions:
  MINING      → CRAFTING   [when copper > 200]
  CRAFTING    → ENERGY     [when graphite > 50]
  ENERGY      → DEFENSE    [when enemies detected]
  DEFENSE     → MINING     [when enemies cleared]
  any         → IDLE       [when no action available]
```

**State Properties:**
- Each state has associated action bias (RL learns within-state adjustments)
- Transitions triggered by resource thresholds or threat detection
- Prevents thrashing between incompatible actions

### 3.3 Priority Queue (Dynamic Urgency)

At each decision point, compute urgency scores:

```python
priorities = {
    "survive": threat_level * 100,           # Spike when enemies nearby
    "power": max(0, 1 - power_ratio) * 50,   # Increase if power low
    "mining": max(0, 1 - copper_ratio) * 30, # Increase if resources low
    "optimize": 10                            # Always present, lowest
}

selected_action = argmax(priorities)
```

**Properties:**
- Normalized to [0, 100] per category
- Threat multiplier: 0-1 based on enemy proximity
- Resource ratios: resource_count / resource_capacity
- RL learns weight adjustments via policy optimization

### 3.4 Hybrid Decision Logic

```python
def decide_action(game_state):
    # 1. BT feasibility check
    feasible_actions = behavior_tree.get_feasible(game_state)
    if not feasible_actions:
        return WAIT
    
    # 2. SM state context
    current_state = state_machine.get_state(game_state)
    state_bias = state_weights[current_state]  # RL learns these
    
    # 3. Priority urgency
    priorities = compute_priorities(game_state)
    
    # 4. RL refinement
    rl_action, rl_placement = policy_network.sample(
        observation=(flat_vector, spatial_map),
        context=(current_state, priorities, feasible_actions)
    )
    
    # Final action = feasible + urgency-weighted + RL-optimized
    return refine_action(rl_action, priorities, state_bias)
```

---

## 4. Observation & Action Spaces

### 4.1 Observation Space

**Flat Vector (15 dimensions):**
```python
flat_obs = [
    # Resources (5)
    copper_count,        lead_count,        coal_count,
    graphite_count,      titanium_count,
    
    # Power (4)
    power_current,       power_capacity,    power_production,
    power_consumption,
    
    # Threat (3)
    enemies_nearby,      wave_number,       time_to_wave,
    
    # Infrastructure (2)
    drills_count,        turrets_count,
    
    # Status (1)
    core_health_ratio
]
```

**2D Spatial Map (32x32 or map_size):**
```python
spatial_map = {
    "blocks": matrix(32, 32),    # 0=empty, 1=drill, 2=conveyor, 3=turret, 4=core, 5=generator
    "resources": matrix(32, 32), # intensity: copper/lead/coal concentration
    "enemies": matrix(32, 32),   # intensity: enemy proximity (high near threats)
}
```

### 4.2 Action Space

**High-Level Actions (Discrete, 7 choices):**
```python
actions = {
    0: PLACE_DRILL,        # Expand mining
    1: PLACE_CONVEYOR,     # Expand logistics
    2: PLACE_GENERATOR,    # Expand power
    3: PLACE_TURRET,       # Expand defense
    4: UPGRADE_BLOCK,      # Improve existing infrastructure
    5: DEMOLISH_BLOCK,     # Optimize/make space
    6: WAIT                # Do nothing
}
```

**Granular Refinement (Continuous):**
```python
placements = {
    "x": float in [0, map_width],       # Gaussian distribution, RL learns center + variance
    "y": float in [0, map_height],
    "rotation": int in [0, 3],          # Discrete: 0, 90, 180, 270 degrees
    "type_variant": int,                # E.g., drill tier, turret class
}
```

**Mapping:**
- High-level action selects what to build
- Granular refinement selects where/how to build it
- RL learns joint distribution P(action, placement | state)

---

## 5. RL Layer: PyTorch Policy Network

### 5.1 Network Architecture

```
Input: flat_obs (15,) + spatial_map (32, 32, 3)

┌─ Flat Stream ──────────────────────────────┐
│ Dense(15) → ReLU → Dense(128) → ReLU      │
└────────────────────┬──────────────────────┘
                     │
┌─ Spatial Stream ───┴──────────────────────┐
│ Conv2D(spatial, 32) → ReLU                │
│ Conv2D(32) → ReLU                         │
│ Conv2D(32 → 64) → ReLU                    │
│ AdaptiveAvgPool2D → Flatten (256)         │
└────────────────────┬──────────────────────┘
                     │
         ┌───────────┴───────────┐
         ↓                       ↓
    Concatenate (128 + 256 = 384)
         │
    Dense(256) → ReLU
    Dense(128) → ReLU
         │
    ┌────┴────┬─────────┬─────────┐
    ↓         ↓         ↓         ↓
  Action   X Coord   Y Coord    Value
  Logits   (Gauss)   (Gauss)    (scalar)
  (7,)     (μ,σ)     (μ,σ)      (1,)
```

### 5.2 Loss Functions

**Policy Loss (Actor):**
```
L_policy = -log(π(a|s)) * Advantage + entropy_coeff * H(π)
```
where `Advantage = R_t - V(s)`

**Value Loss (Critic):**
```
L_value = (V(s) - R_t)^2
```

**Total Loss:**
```
L_total = L_policy + 0.5 * L_value - 0.01 * H(π)
```

**Entropy Bonus:** Encourages exploration (prevents premature convergence)

### 5.3 Training Loop

```python
for episode in range(num_episodes):
    observation = env.reset()
    cumulative_reward = 0
    
    for step in range(max_steps):
        # 1. Forward pass
        action_dist, placement_dist, value = policy_net(observation)
        
        # 2. Sample action + placement
        action = action_dist.sample()
        x, y = placement_dist.sample()
        
        # 3. Execute in game
        observation_next, reward, done = env.step(action, x, y)
        
        # 4. Store trajectory
        trajectory.append((observation, action, reward, value, done))
        
        observation = observation_next
        cumulative_reward += reward
        
        if done:
            break
    
    # 5. Compute returns & advantages
    returns = compute_returns(trajectory, gamma=0.99)
    advantages = returns - trajectory["values"]
    advantages = (advantages - mean) / (std + 1e-8)  # normalize
    
    # 6. Backprop
    loss = compute_loss(trajectory, returns, advantages)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    # 7. Log metrics
    log({"episode": episode, "reward": cumulative_reward, "loss": loss.item()})
```

---

## 6. Reward Function (Composite)

```python
reward = 0

# 1. Survival Bonus (per tick)
if core_health > 0:
    reward += 1  # Bonus for staying alive

# 2. Resource Efficiency
resources_gained = (copper + lead + coal + graphite + titanium) - prev_resources
reward += resources_gained * 0.1

# 3. Power Stability
power_ratio = power_current / power_capacity
if 0.7 < power_ratio < 1.0:
    reward += 0.5  # Bonus for healthy power grid
else:
    reward -= 0.2  # Penalty for imbalance

# 4. Defense Success
if wave_active and enemies_cleared:
    reward += 10  # Bonus for surviving wave

# 5. Inefficiency Penalty
if action_taken == WAIT and core_has_resources:
    reward -= 0.1  # Penalty for idle when work available

# 6. Catastrophe Penalty
if core_destroyed:
    reward -= 100

return reward
```

---

## 7. Training Progression (Sequential Goals)

### Phase 1: Survival (Waves 1-3, ~1000 episodes)
**Objective:** Stay alive, learn basic mining + conveyor logic  
**Rules dominate:** BT prevents catastrophic errors  
**RL learns:** Resource thresholds, basic placement safety  

**Success Criteria:**
- Win rate > 80% against waves 1-3
- Zero core destruction in 100 consecutive episodes

### Phase 2: Production (Waves 4-10, ~2000 episodes)
**Objective:** Accumulate resources efficiently  
**Rules stable:** Transition to CRAFTING + ENERGY states  
**RL learns:** Conveyor optimization, drill clustering, timing  

**Success Criteria:**
- 500+ resources/min at stable state
- Complete production chains without deadlock

### Phase 3: Defense (Waves 10+, ~3000 episodes)
**Objective:** Survive high-intensity waves  
**Rules adjusted:** Threat priority spike, aggressive turret placement  
**RL learns:** Turret placement strategy, defensive coordination  

**Success Criteria:**
- Survive waves 10+ in 70% of attempts
- Minimize casualties (core damage < 20%)

---

## 8. Data Flow & Integration

### 8.1 Game State Capture

**Hybrid approach (CV + Memory):**
```python
def read_game_state():
    # Fast path: memory introspection
    state_mem = read_mindustry_memory()  # Direct struct read
    
    # Verification: screenshot analysis for anomalies
    screen = capture_screenshot()
    resource_counts_cv = ocr_resources(screen)
    
    # Reconcile if mismatch
    if abs(state_mem.copper - resource_counts_cv.copper) > 5:
        log_anomaly("resource sync issue")
    
    return state_mem
```

### 8.2 Action Execution

```python
def execute_action(action, x, y, rotation=0):
    # High-level → granular mapping
    if action == PLACE_DRILL:
        click_menu("mining")
        click_drill_tool()
        click_at(x, y)
    elif action == PLACE_CONVEYOR:
        click_menu("logistics")
        click_conveyor_tool()
        rotate_to(rotation)
        drag_from_to((x, y), (x + delta_x, y + delta_y))
    # ... etc for other actions
    
    # Wait for game to process
    time.sleep(0.2)
```

---

## 9. Success Criteria & Metrics

### Quantitative (Phase 1 MVP):
- **Survival Rate:** ≥ 80% on waves 1-3
- **Resource Accumulation:** ≥ 500 resources/min at stable state
- **Efficiency Ratio:** Resources/Time ratio ≥ 5.0
- **Training Convergence:** Loss plateau after 1000 episodes

### Qualitative:
- No catastrophic failures (core destruction < 5% of episodes)
- Visibly intelligent mining site layout
- Conveyor chains connect drills to core without deadlock
- Turrets placed in reasonable defensive positions

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Rule layer too rigid** | AI can't adapt to anomalies | Parameterize rules, RL learns weights |
| **RL exploration chaos** | Unstable training, catastrophes | Entropy regularization + BT constraints |
| **State space explosion** | Training too slow | Normalize observations, spatial downsampling |
| **Sim-to-reality gap** | Performance degrades in real game | Validate on actual game frequently |
| **Memory read failures** | State inconsistency | Fallback to CV-only mode |

---

## 11. Next Steps

1. **Environment Setup:** Build game wrapper (state reader + action executor)
2. **Rule Layer Impl:** Code BT + SM + Priority queue
3. **RL Layer Impl:** PyTorch policy network + training loop
4. **Integration:** Connect rule + RL layers
5. **Phase 1 Training:** Run 1000 episodes on waves 1-3
6. **Validation:** Test on real Mindustry instance
7. **Iterate:** Refine reward function based on observed behavior

---

**Approved by:** [user]  
**Date:** 2026-04-05
