# Phase 2: Socket Optimization - Sparse State Format

> **For Claude:** Use superpowers:subagent-driven-development to execute this plan task-by-task.

**Goal:** Compress JSON state from 961 tiles (50KB) to sparse features (500 bytes), reducing latency from 179ms → <50ms and eliminating 800ms peaks.

**Architecture:** Replace full 31×31 grid matrix with sparse observation: resource vector (5), unit status (4), top-5 enemies (5×2), top-5 ores (5×2). Block names → integer IDs. Python parser updated to consume sparse format.

**Tech Stack:** JavaScript (Mindustry mod), NumPy (Python RL), no new dependencies

---

## Task 1: Create Block ID Enum in main.js

**Files:**
- Modify: `scripts/main.js:1-50` (add enum after config block)

**Step 1: Read current config section**

Run: `head -50 scripts/main.js`

**Step 2: Add Block ID enum**

After line 13 (after `const config = {...}`), insert:

```javascript
// ============================================================================
// BLOCK ID ENUMERATION
// ============================================================================
// Maps block names to integer IDs for compact state transmission.
// Must match BLOCK_IDS in rl/env/spaces.py exactly.
const BLOCK_IDS = {
    "air": 0,
    "copper-wall": 1,
    "copper-wall-large": 2,
    "duo": 3,
    "scatter": 4,
    "hail": 5,
    "lancer": 6,
    "wave": 7,
    "swarmer": 8,
    "mechanical-drill": 9,
    "pneumatic-drill": 10,
    "conveyor": 11,
    "titanium-conveyor": 12,
    "router": 13,
    "junction": 14,
    "overflow-gate": 15,
    "sorter": 16,
    "solar-panel": 17,
    "solar-panel-large": 18,
    "battery": 19,
    "battery-large": 20,
    "power-node": 21,
    "power-node-large": 22,
    "thermal-generator": 23,
    "core-shard": 24,
    "vault": 25,
    "container": 26,
    "mender": 27,
    "mend-projector": 28,
    "overdrive-projector": 29,
    "force-projector": 30
};

function getBlockId(blockName) {
    return BLOCK_IDS[blockName] !== undefined ? BLOCK_IDS[blockName] : 31; // 31 = unknown
}
```

**Step 3: Commit**

```bash
git add scripts/main.js
git commit -m "feat: add block ID enum for sparse state compression"
```

---

## Task 2: Add Ore Detection Function in main.js

**Files:**
- Modify: `scripts/main.js:240-306` (add new functions before `captureGameState`)

**Step 1: Read current tile scanning section**

Run: `sed -n '240,306p' scripts/main.js`

**Step 2: Add ore detection helper after config**

Before `function captureGameState()` (around line 160), insert:

```javascript
// ============================================================================
// SPARSE STATE HELPERS
// ============================================================================

/**
 * Find top-N nearest ore tiles in grid around center.
 * Returns array of {distance, angle_deg, block_id, x, y}
 * Ores: copper-ore, lead-ore, coal, graphite-ore, titanium-ore, thorium-ore, scrap
 */
function findNearestOres(centerX, centerY, radius, maxOres) {
    let ores = [];
    let oreNames = ["copper-ore", "lead-ore", "coal", "graphite-ore", "titanium-ore", "thorium-ore", "scrap"];
    
    for (let dx = -radius; dx <= radius; dx++) {
        for (let dy = -radius; dy <= radius; dy++) {
            let tile = Vars.world.tile(centerX + dx, centerY + dy);
            
            if (tile != null) {
                let block = tile.block();
                let blockName = block != null ? block.name : "air";
                
                // Check if this tile is an ore
                if (oreNames.includes(blockName)) {
                    let distance = Math.sqrt(dx * dx + dy * dy);
                    let angle = Math.atan2(dy, dx) * 180 / Math.PI; // degrees [-180, 180]
                    
                    ores.push({
                        distance: distance,
                        angle: angle,
                        block_id: getBlockId(blockName),
                        x: centerX + dx,
                        y: centerY + dy
                    });
                }
            }
        }
    }
    
    // Sort by distance, take top maxOres
    ores.sort((a, b) => a.distance - b.distance);
    return ores.slice(0, maxOres);
}

/**
 * Find top-N nearest enemy units by distance.
 * Returns array of {distance, angle_deg, type_id, hp}
 */
function findNearestEnemies(playerX, playerY, maxEnemies) {
    let enemies = [];
    
    Units.nearbyEnemies(Team.sharded, playerX * 8, playerY * 8, 300, (unit) => {
        let dx = Math.floor(unit.x / 8) - playerX;
        let dy = Math.floor(unit.y / 8) - playerY;
        let distance = Math.sqrt(dx * dx + dy * dy);
        let angle = Math.atan2(dy, dx) * 180 / Math.PI;
        
        enemies.push({
            distance: distance,
            angle: angle,
            hp: Math.floor((unit.health / unit.maxHealth) * 100) / 100
        });
    });
    
    enemies.sort((a, b) => a.distance - b.distance);
    return enemies.slice(0, maxEnemies);
}
```

**Step 3: Commit**

```bash
git add scripts/main.js
git commit -m "feat: add ore and enemy detection functions for sparse state"
```

---

## Task 3: Replace Grid Matrix with Sparse Format in captureGameState()

**Files:**
- Modify: `scripts/main.js:267-285` (replace grid generation loop)

**Step 1: Read current grid generation**

Run: `sed -n '267,285p' scripts/main.js`

**Step 2: Replace grid loop with sparse format**

Replace lines 267-285 (the full grid matrix loop) with:

```javascript
        // PHASE 2 OPTIMIZATION: Replace full grid matrix with sparse features
        // Old: 961 tiles × 30-50 char strings = ~50KB
        // New: ~50 bytes for sparse enemies/ores + resource vector
        
        let playerX = state.player.x;
        let playerY = state.player.y;
        
        // Nearby ores: top 5 closest
        state.nearbyOres = findNearestOres(playerX, playerY, config.gridRadius, 5);
        
        // Nearby enemies: top 5 closest
        state.nearbyEnemies = findNearestEnemies(playerX, playerY, 5);
        
        // Keep grid array for now (optional) but empty for JSON compression
        state.grid = [];
```

**Step 3: Verify changes**

Run: `grep -A 15 "PHASE 2 OPTIMIZATION" scripts/main.js`

Expected: New sparse extraction code visible

**Step 4: Commit**

```bash
git add scripts/main.js
git commit -m "feat: replace full grid matrix with sparse ore/enemy features"
```

---

## Task 4: Update parse_observation() in spaces.py for Sparse Grid

**Files:**
- Modify: `rl/env/spaces.py:119-140` (replace `_parse_grid` function)

**Step 1: Read current parse_observation**

Run: `sed -n '119,140p' rl/env/spaces.py`

**Step 2: Replace _parse_grid function**

Replace lines 129-140 with:

```python
def _parse_grid(grid: List[Dict[str, Any]]) -> np.ndarray:
    """
    Parse sparse grid format (empty array, ores/enemies in separate fields).
    Backward compatible: returns (4, 31, 31) zeros if grid is empty.
    """
    arr = np.zeros((4, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    # Sparse grid format: full matrix replaced with nearbyOres/nearbyEnemies
    # Kept for backward compatibility but will be empty array in Phase 2
    for tile in grid:
        x = int(tile.get("x", 0))
        y = int(tile.get("y", 0))
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            continue
        arr[0, y, x] = _encode_block(tile.get("block", "air"))
        arr[1, y, x] = float(tile.get("hp", 0.0))
        arr[2, y, x] = _encode_team(tile.get("team", "neutral"))
        arr[3, y, x] = float(tile.get("rotation", 0)) / 3.0
    
    return arr
```

**Step 3: Run tests to verify backward compatibility**

Run: `pytest rl/tests/test_spaces.py::test_parse_observation -v`

Expected: PASS (grid is now empty but still valid shape)

**Step 4: Commit**

```bash
git add rl/env/spaces.py
git commit -m "feat: update grid parser for sparse state format (backward compatible)"
```

---

## Task 5: Update _parse_features to Include Sparse Ores/Enemies

**Files:**
- Modify: `rl/env/spaces.py:143-220` (extend features to include sparse observations)

**Step 1: Read current features parser**

Run: `sed -n '143,220p' rl/env/spaces.py`

**Step 2: Extend OBS_FEATURES_DIM constant**

Update line 24:

```python
OBS_FEATURES_DIM = 72   # Old 47 + 5 enemies (dist/angle) + 5 ores (dist/angle) + 2 padding = 72
```

**Step 3: Add ore/enemy feature extraction in _parse_features**

Before `return feat` (around line 220), add:

```python
    # PHASE 2: Sparse ore/enemy features (25 dims)
    # Top 5 nearest ores: 5 × (distance + angle + block_id) = 15 dims
    # Top 5 nearest enemies: 5 × (distance + angle + hp) = 15 dims
    # Offset from old 47 → new 72
    
    nearby_ores = state.get("nearbyOres", [])
    for i in range(min(5, len(nearby_ores))):
        ore = nearby_ores[i]
        offset = 47 + i * 3
        feat[offset] = float(ore.get("distance", 0.0)) / 50.0  # normalize by typical radius
        feat[offset + 1] = float(ore.get("angle", 0.0)) / 180.0  # normalize to [-1, 1]
        feat[offset + 2] = float(ore.get("block_id", 0)) / 32.0  # normalize block ID
    
    nearby_enemies = state.get("nearbyEnemies", [])
    for i in range(min(5, len(nearby_enemies))):
        enemy = nearby_enemies[i]
        offset = 47 + 15 + i * 3  # After ores (15 dims)
        feat[offset] = float(enemy.get("distance", 0.0)) / 50.0
        feat[offset + 1] = float(enemy.get("angle", 0.0)) / 180.0
        feat[offset + 2] = float(enemy.get("hp", 0.0))  # 0-1 normalized already
```

**Step 4: Run tests**

Run: `pytest rl/tests/test_spaces.py -v`

Expected: PASS (feature vector now 72-dim instead of 47)

**Step 5: Commit**

```bash
git add rl/env/spaces.py
git commit -m "feat: add sparse ore/enemy features to observation (72-dim instead of 47)"
```

---

## Task 6: Verify All Tests Pass

**Files:**
- Test: `rl/tests/test_spaces.py` (no modifications, verify pass)

**Step 1: Run full test suite**

Run: `pytest rl/tests/ -v --tb=short`

Expected: 89+ tests PASS

**Step 2: Specifically check spaces tests**

Run: `pytest rl/tests/test_spaces.py -v`

Expected: All parsing tests pass (grid, features, encode_block)

**Step 3: Verify no type errors**

Run: `python -m py_compile rl/env/spaces.py`

Expected: No output (success)

**Step 4: Commit if all pass**

```bash
git add rl/tests/
git commit -m "test: verify all spaces tests pass with sparse format (89/89)"
```

---

## Task 7: Measure Socket Latency Improvement

**Files:**
- Monitor: `rl/logs/live_metrics.json` (read-only)
- Test: `rl/tests/test_env.py` (run to measure)

**Step 1: Check baseline latency**

Run: `cat rl/logs/live_metrics.json | jq '.step_latency_ms'`

Expected: Shows current μ=179ms, σ=316ms

**Step 2: Run training for 100 steps to capture new latency**

Run: `cd /home/pedro/repo/mindustry-ia-inteface-mod && python rl/train.py --steps 100 --log-live true`

Wait for 100 steps to complete (~2-5 min depending on Mindustry server speed)

**Step 3: Check new latency in live_metrics**

Run: `cat rl/logs/live_metrics.json | jq '.step_latency_ms | {mean: (map(.) | add / length), max: max, min: min}'`

Expected: New μ < 50ms (target), σ < 10ms (target)

**Step 4: Document improvement**

If improvement confirmed:
```bash
git add rl/logs/live_metrics.json
git commit -m "perf: socket optimization complete — latency 179ms → <50ms"
```

---

## Verification Checklist

- [ ] Block ID enum added to main.js (matches spaces.py)
- [ ] Ore detection function working (findNearestOres returns top-5)
- [ ] Enemy detection function working (findNearestEnemies returns top-5)
- [ ] Grid matrix replaced with sparse format in mod
- [ ] parse_observation() backward compatible (empty grid still works)
- [ ] Feature vector extended to 72-dim (47 + 25 sparse)
- [ ] All tests pass (89+)
- [ ] Socket latency measured: <50ms (from 179ms)
- [ ] Jitter <10ms (from 316ms)
- [ ] All changes committed atomically

---

## Rollback Steps

If latency doesn't improve or tests break:

1. Revert sparse format: `git revert HEAD~N` (where N = number of commits to revert)
2. Restore original grid matrix in captureGameState()
3. Restore parse_observation() to original 47-dim features
4. Verify tests pass again with original code

---

## Next Phase (After Verification)

Phase 3: Entropy Coefficient Tuning (boost exploration in first 50k steps, decay back)
- Modify `rl/train.py`: adjust `entropy_coef` schedule in PPO config
- Target: Increase exploration → reduce WAIT action bias → improve action diversity
