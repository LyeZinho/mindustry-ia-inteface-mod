"""Tests for observation/action space definitions."""
import numpy as np
import pytest
from gymnasium import spaces
from rl.env.spaces import (
    make_obs_space, make_action_space, parse_observation,
    compute_action_mask, NUM_ACTION_TYPES, NUM_SLOTS,
    BLOCK_TURRET, BLOCK_WALL, BLOCK_POWER, BLOCK_DRILL,
    _encode_block, BLOCK_IDS,
)

MINIMAL_STATE = {
    "tick": 1000, "time": 500, "wave": 3, "waveTime": 300,
    "resources": {"copper": 450, "lead": 120, "graphite": 75, "titanium": 50, "thorium": 0},
    "power": {"produced": 120.5, "consumed": 80.2, "stored": 500, "capacity": 1000},
    "core": {"hp": 0.95, "x": 15, "y": 15, "size": 3},
    "player": {"x": 16, "y": 17, "alive": True, "hp": 0.8},
    "enemies": [],
    "friendlyUnits": [],
    "buildings": [],
    "grid": [],  # Empty grid (sparse format)
    "nearbyOres": [],
    "nearbyEnemies": [],
}


def test_action_space_structure():
    act = make_action_space()
    assert isinstance(act, spaces.MultiDiscrete)
    assert list(act.nvec) == [NUM_ACTION_TYPES, NUM_SLOTS]


def test_obs_space_shape():
    obs = make_obs_space()
    assert isinstance(obs, spaces.Dict)
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (92,)


def test_parse_observation_returns_correct_shapes():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["grid"].shape == (4, 31, 31)
    assert obs["features"].shape == (92,)


def test_parse_observation_grid_dtype():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["grid"].dtype == np.float32


def test_parse_observation_features_dtype():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["features"].dtype == np.float32


def test_parse_observation_grid_within_range():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["grid"].min() >= 0.0
    assert obs["grid"].max() <= 1.0


def test_parse_observation_player_features():
    """feat[45..48] encode player position relative to core and alive/hp."""
    obs = parse_observation(MINIMAL_STATE)
    feat = obs["features"]
    # player at (16,17), core at (15,15) → dx=1, dy=2, both ÷15
    assert feat[45] == pytest.approx(1.0 / 15.0, abs=1e-5)
    assert feat[46] == pytest.approx(2.0 / 15.0, abs=1e-5)
    assert feat[47] == pytest.approx(1.0)   # alive
    assert feat[48] == pytest.approx(0.8)   # hp


def test_parse_observation_player_dead():
    """feat[47]=0 and feat[48]=0 when player not alive."""
    state = {**MINIMAL_STATE, "player": {"x": 0, "y": 0, "alive": False, "hp": 0.0}}
    obs = parse_observation(state)
    assert obs["features"][47] == pytest.approx(0.0)
    assert obs["features"][48] == pytest.approx(0.0)


def test_parse_observation_zero_pads_missing_enemies():
    obs = parse_observation(MINIMAL_STATE)
    enemy_block = obs["features"][13:33]
    assert np.all(enemy_block == 0.0)


def test_obs_space_contains_parsed_obs():
    obs_space = make_obs_space()
    obs = parse_observation(MINIMAL_STATE)
    assert obs_space["grid"].contains(obs["grid"])
    assert obs["features"].shape == obs_space["features"].shape


def test_encode_block_deterministic():
    """Known blocks map to unique, stable floats."""
    seen = set()
    for name in BLOCK_IDS:
        val = _encode_block(name)
        assert 0.0 <= val < 1.0
        assert val not in seen, f"collision for {name}"
        seen.add(val)


def test_encode_block_unknown_maps_to_last():
    """Unknown blocks map to the 'unknown' slot."""
    val = _encode_block("nonexistent-block-xyz")
    assert 0.0 < val < 1.0


# ---- Action masking tests ---- #

def test_action_mask_shape():
    mask = compute_action_mask(MINIMAL_STATE)
    assert mask.shape == (NUM_ACTION_TYPES + NUM_SLOTS,)
    assert mask.dtype == np.bool_


def test_action_mask_wait_always_valid():
    mask = compute_action_mask(MINIMAL_STATE)
    assert mask[0] is np.bool_(True)

    dead_state = {**MINIMAL_STATE, "player": {"alive": False, "hp": 0.0}}
    mask_dead = compute_action_mask(dead_state)
    assert mask_dead[0] is np.bool_(True)


def test_action_mask_dead_player_blocks_actions():
    dead_state = {**MINIMAL_STATE, "player": {"alive": False, "hp": 0.0}}
    mask = compute_action_mask(dead_state)
    assert mask[0] == True
    assert np.all(mask[1:NUM_ACTION_TYPES] == False)
    assert np.all(mask[NUM_ACTION_TYPES:] == True)


def test_action_mask_no_resources_blocks_build():
    broke_state = {
        **MINIMAL_STATE,
        "resources": {"copper": 0, "lead": 0, "graphite": 0},
        "buildings": [],
    }
    mask = compute_action_mask(broke_state)
    assert mask[0] == True   # WAIT
    assert mask[1] == True   # MOVE
    assert mask[2] == False  # BUILD_TURRET (needs copper >= 35)
    assert mask[3] == False  # BUILD_WALL (needs copper >= 6)
    assert mask[4] == False  # BUILD_POWER (needs copper >= 40 and lead >= 35)
    assert mask[5] == False  # BUILD_DRILL (needs copper >= 12)
    assert mask[6] == False  # REPAIR (no buildings)


def test_action_mask_with_enough_resources():
    rich_state = {
        **MINIMAL_STATE,
        "resources": {"copper": 100, "lead": 100, "graphite": 50},
        "buildings": [{"block": "duo", "hp": 0.5}],
    }
    mask = compute_action_mask(rich_state)
    assert np.all(mask[:NUM_ACTION_TYPES] == True)
    assert np.all(mask[NUM_ACTION_TYPES:] == True)


def test_action_mask_partial_resources():
    state = {
        **MINIMAL_STATE,
        "resources": {"copper": 35, "lead": 0, "graphite": 0},
        "buildings": [],
    }
    mask = compute_action_mask(state)
    assert mask[2] == True   # BUILD_TURRET (35 copper — exact cost)
    assert mask[3] == True   # BUILD_WALL (35 >= 6)
    assert mask[4] == False  # BUILD_POWER (needs lead >= 35)
    assert mask[5] == True   # BUILD_DRILL (35 copper >= 12)


def test_num_action_types_matches_registry():
    from rl.env.spaces import ACTION_REGISTRY, NUM_ACTION_TYPES
    assert NUM_ACTION_TYPES == len(ACTION_REGISTRY)
    assert NUM_ACTION_TYPES == 12


def test_action_names_length_matches_num_action_types():
    from rl.env.spaces import ACTION_NAMES, NUM_ACTION_TYPES
    assert len(ACTION_NAMES) == NUM_ACTION_TYPES


def test_new_action_masks_no_resources():
    from rl.env.spaces import compute_action_mask, NUM_ACTION_TYPES
    broke_state = {
        "tick": 1000, "time": 500, "wave": 3, "waveTime": 300,
        "resources": {"copper": 0, "lead": 0, "graphite": 0},
        "power": {"produced": 0, "consumed": 0, "stored": 0, "capacity": 1},
        "core": {"hp": 0.95, "x": 15, "y": 15, "size": 3},
        "player": {"x": 16, "y": 17, "alive": True, "hp": 0.8},
        "enemies": [], "friendlyUnits": [], "buildings": [], "grid": [],
        "nearbyOres": [], "nearbyEnemies": [],
    }
    mask = compute_action_mask(broke_state)
    assert mask.shape == (NUM_ACTION_TYPES + 9,)
    assert mask[7]  == False
    assert mask[8]  == False
    assert mask[9]  == False
    assert mask[10] == False
    assert mask[11] == False


def test_new_action_masks_with_resources():
    from rl.env.spaces import compute_action_mask
    rich_state = {
        "tick": 1000, "time": 500, "wave": 3, "waveTime": 300,
        "resources": {"copper": 100, "lead": 100, "graphite": 50},
        "power": {"produced": 0, "consumed": 0, "stored": 0, "capacity": 1},
        "core": {"hp": 0.95, "x": 15, "y": 15, "size": 3},
        "player": {"x": 16, "y": 17, "alive": True, "hp": 0.8},
        "enemies": [], "friendlyUnits": [],
        "buildings": [{"block": "duo", "hp": 0.5}],
        "grid": [], "nearbyOres": [], "nearbyEnemies": [],
    }
    mask = compute_action_mask(rich_state)
    assert mask[7]  == True
    assert mask[8]  == True
    assert mask[9]  == True
    assert mask[10] == True
    assert mask[11] == True


def test_pneumatic_drill_needs_graphite():
    from rl.env.spaces import compute_action_mask
    state = {
        "tick": 1000, "time": 500, "wave": 3, "waveTime": 300,
        "resources": {"copper": 100, "lead": 100, "graphite": 0},
        "power": {"produced": 0, "consumed": 0, "stored": 0, "capacity": 1},
        "core": {"hp": 0.95, "x": 15, "y": 15, "size": 3},
        "player": {"x": 16, "y": 17, "alive": True, "hp": 0.8},
        "enemies": [], "friendlyUnits": [], "buildings": [], "grid": [],
        "nearbyOres": [], "nearbyEnemies": [],
    }
    mask = compute_action_mask(state)
    assert mask[11] == False


# ---- New 83-dim observation tests (Task 2) ---- #

def test_obs_features_dim_is_83():
    from rl.env.spaces import OBS_FEATURES_DIM
    assert OBS_FEATURES_DIM == 92


def test_obs_space_shape_is_83():
    obs = make_obs_space()
    assert obs["features"].shape == (92,)


def test_parse_observation_shape_is_83():
    obs = parse_observation(MINIMAL_STATE)
    assert obs["features"].shape == (92,)


def test_extended_resources_in_obs():
    from rl.env.spaces import parse_observation, EXTENDED_RESOURCES
    state = {
        "tick": 1000, "time": 500, "wave": 3, "waveTime": 300,
        "resources": {
            "copper": 500, "lead": 200, "graphite": 100,
            "titanium": 50, "thorium": 10, "coal": 30, "sand": 80,
            "silicon": 150, "oil": 75, "water": 200, "metaglass": 40,
        },
        "power": {"produced": 120.5, "consumed": 80.2, "stored": 500, "capacity": 1000},
        "core": {"hp": 0.95, "x": 15, "y": 15, "size": 3},
        "player": {"x": 16, "y": 17, "alive": True, "hp": 0.8},
        "enemies": [], "friendlyUnits": [], "buildings": [], "grid": [],
        "nearbyOres": [], "nearbyEnemies": [],
    }
    obs = parse_observation(state)
    assert obs["features"].shape == (92,)
    assert obs["features"][79] == pytest.approx(150 / 1000.0)  # silicon
    assert obs["features"][80] == pytest.approx(75  / 1000.0)  # oil
    assert obs["features"][81] == pytest.approx(200 / 1000.0)  # water
    assert obs["features"][82] == pytest.approx(40  / 1000.0)  # metaglass


def test_extended_resources_zero_when_absent():
    from rl.env.spaces import parse_observation
    obs = parse_observation(MINIMAL_STATE)
    assert obs["features"][79] == 0.0
    assert obs["features"][80] == 0.0
    assert obs["features"][81] == 0.0
    assert obs["features"][82] == 0.0


def test_obs_features_dim_is_92():
    from rl.env.spaces import OBS_FEATURES_DIM
    assert OBS_FEATURES_DIM == 92


def test_ore_in_slot_parsed_from_slots_ore_type():
    from rl.env.spaces import parse_observation
    state = {
        "core": {"hp": 1.0, "x": 10, "y": 10},
        "resources": {},
        "power": {},
        "player": {"alive": True, "x": 10, "y": 10, "hp": 1.0},
        "slotsOreType": [1, 0, 2, 0, 3, 0, 4, 5, 6],
    }
    obs = parse_observation(state)
    feats = obs["features"]
    assert len(feats) == 92
    assert abs(feats[83] - 1/7) < 1e-5
    assert feats[84] == 0.0
    assert abs(feats[85] - 2/7) < 1e-5
    assert abs(feats[87] - 3/7) < 1e-5


def test_ore_in_slot_missing_slotsOreType_gives_zeros():
    from rl.env.spaces import parse_observation
    state = {"core": {}, "resources": {}, "power": {}, "player": {}}
    obs = parse_observation(state)
    assert all(obs["features"][83:92] == 0.0)


def test_build_drill_mask_matches_mod_cost():
    """BUILD_DRILL mask should require exactly 12 copper (real Mindustry v7 cost).

    Regression guard: Python mask must match what scripts/main.js blockCosts charges.
    As of mimi-gateway-v1.0.6, mechanical-drill costs [["copper", 12]].
    """
    from rl.env.spaces import ACTION_REGISTRY, _action_idx

    idx = _action_idx("BUILD_DRILL")
    drill_def = ACTION_REGISTRY[idx]

    # With exactly 12 copper and no lead/graphite — should be allowed
    assert drill_def.mask_fn({"copper": 12, "lead": 0, "graphite": 0}) is True, \
        "BUILD_DRILL should be allowed with exactly 12 copper"

    # With 11 copper — should be blocked
    assert drill_def.mask_fn({"copper": 11, "lead": 100, "graphite": 100}) is False, \
        "BUILD_DRILL should be blocked with only 11 copper"

    # Lead and graphite are NOT required
    assert drill_def.mask_fn({"copper": 12, "lead": 0, "graphite": 0}) is True, \
        "BUILD_DRILL should not require lead or graphite"
