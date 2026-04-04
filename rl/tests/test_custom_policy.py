import numpy as np
import pytest
import torch


def test_extractor_output_shape():
    from rl.models.custom_policy import MindustryFeatureExtractor
    from rl.env.spaces import make_obs_space

    obs_space = make_obs_space()
    extractor = MindustryFeatureExtractor(obs_space, features_dim=256)
    batch = {
        "grid": torch.zeros(2, 8, 31, 31),
        "features": torch.zeros(2, 121),
    }
    with torch.no_grad():
        out = extractor(batch)
    assert out.shape == (2, 256)


def test_extractor_no_nan():
    from rl.models.custom_policy import MindustryFeatureExtractor
    from rl.env.spaces import make_obs_space

    obs_space = make_obs_space()
    extractor = MindustryFeatureExtractor(obs_space)
    batch = {
        "grid": torch.rand(4, 8, 31, 31),
        "features": torch.rand(4, 121),
    }
    with torch.no_grad():
        out = extractor(batch)
    assert not torch.any(torch.isnan(out))


def test_multi_head_policy_instantiates():
    from sb3_contrib import MaskablePPO
    from rl.models.custom_policy import MindustryActorCriticPolicy
    from rl.env.spaces import make_obs_space, make_action_space
    import gymnasium as gym

    class _DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = make_obs_space()
            self.action_space = make_action_space()

        def reset(self, **kw):
            return {k: np.zeros(v.shape, dtype=np.float32) for k, v in self.observation_space.items()}, {}

        def step(self, a):
            return {k: np.zeros(v.shape, dtype=np.float32) for k, v in self.observation_space.items()}, 0.0, False, False, {}

        def action_masks(self):
            return np.ones(21, dtype=bool)

    env = _DummyEnv()
    model = MaskablePPO(MindustryActorCriticPolicy, env, verbose=0)
    assert model.policy is not None


def test_multi_head_policy_predict():
    from sb3_contrib import MaskablePPO
    from rl.models.custom_policy import MindustryActorCriticPolicy
    from rl.env.spaces import make_obs_space, make_action_space
    import gymnasium as gym

    class _DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = make_obs_space()
            self.action_space = make_action_space()

        def reset(self, **kw):
            return {k: np.zeros(v.shape, dtype=np.float32) for k, v in self.observation_space.items()}, {}

        def step(self, a):
            return {k: np.zeros(v.shape, dtype=np.float32) for k, v in self.observation_space.items()}, 0.0, False, False, {}

        def action_masks(self):
            return np.ones(21, dtype=bool)

    env = _DummyEnv()
    model = MaskablePPO(MindustryActorCriticPolicy, env, verbose=0)
    obs = {k: np.zeros(v.shape, dtype=np.float32) for k, v in env.observation_space.items()}
    action, _ = model.predict(obs, action_masks=np.ones(21, dtype=bool))
    assert action is not None
