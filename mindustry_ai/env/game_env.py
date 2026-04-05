import numpy as np
import torch
from mindustry_ai.game.state_reader import GameStateReader
from mindustry_ai.game.action_executor import ActionExecutor
from mindustry_ai.rules.hybrid_decider import HybridDecider


class MindustryEnv:
    def __init__(self, max_steps=1000, map_size=16):
        self.max_steps = max_steps
        self.map_size = map_size
        self.step_count = 0

        self.state_reader = GameStateReader(map_width=map_size, map_height=map_size)
        self.action_executor = ActionExecutor()
        self.hybrid_decider = HybridDecider()

        self.observation_space = {
            "flat_state": (15,),
            "spatial_state": (map_size, map_size),
        }
        self.action_space = 7

        self.episode_reward = 0.0
        self.last_resources = 0.0
        self.last_power = 0.0
        self.last_health = 100.0

    def reset(self):
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_resources = 0.0
        self.last_power = 0.0
        self.last_health = 100.0

        game_state = self.state_reader.read_state()
        obs = self._package_observation(game_state)
        return obs

    def step(self, action):
        self.step_count += 1
        game_state = self.state_reader.read_state()
        self.action_executor.execute(action)
        game_state = self.state_reader.read_state()

        reward = self.compute_reward()
        self.episode_reward += reward
        done = self.step_count >= self.max_steps

        obs = self._package_observation(game_state)
        info = {"episode_reward": self.episode_reward}

        return obs, reward, done, info

    def compute_reward(self):
        game_state = self.state_reader.read_state()

        resources = game_state["resources"]["copper"] + game_state["resources"]["lead"]
        power = game_state["power"]["current"]
        health = game_state["status"]["core_health"]

        resource_delta = resources - self.last_resources
        power_delta = power - self.last_power
        health_delta = health - self.last_health

        self.last_resources = resources
        self.last_power = power
        self.last_health = health

        resource_reward = max(0, resource_delta) * 0.1
        power_reward = max(0, power_delta) * 0.05
        survival_penalty = max(0, -health_delta) * 0.2

        reward = resource_reward + power_reward - survival_penalty
        return float(reward)

    def _package_observation(self, game_state):
        flat_state = self.state_reader.to_flat_vector(game_state)
        spatial_maps = self.state_reader.to_spatial_map(game_state)
        spatial_state = spatial_maps["blocks"].astype(np.float32)

        return {"flat_state": flat_state, "spatial_state": spatial_state}
