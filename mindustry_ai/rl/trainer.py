import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np


class A2CTrainer:
    def __init__(self, policy_net, learning_rate=0.0003, gamma=0.99, gae_lambda=0.95):
        self.policy_net = policy_net
        self.optimizer = Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def collect_trajectory(self, env, max_steps=1000):
        env.reset()
        device = next(self.policy_net.parameters()).device

        states_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        log_probs_list = []

        for step in range(max_steps):
            obs, reward, done, info = env.step(0)

            flat_state = torch.tensor(obs["flat_state"], dtype=torch.float32).to(
                device
            )
            spatial_state = torch.tensor(
                obs["spatial_state"], dtype=torch.float32
            ).to(device)
            
            spatial_state = spatial_state.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

            with torch.no_grad():
                action_logits, placement_mu, placement_sigma, value = self.policy_net(
                    flat_state.unsqueeze(0), spatial_state
                )

                action_dist = torch.distributions.Categorical(
                    logits=action_logits.squeeze(0)
                )
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

            states_list.append(
                {"flat": obs["flat_state"], "spatial": obs["spatial_state"]}
            )
            actions_list.append(action.item())
            rewards_list.append(reward)
            values_list.append(value.squeeze().item())
            log_probs_list.append(log_prob.item())

            if done:
                break

        next_obs, _, _, _ = env.step(0)
        flat_state = torch.tensor(
            next_obs["flat_state"], dtype=torch.float32
        ).to(device)
        spatial_state = torch.tensor(
            next_obs["spatial_state"], dtype=torch.float32
        ).to(device)
        
        spatial_state = spatial_state.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

        with torch.no_grad():
            _, _, _, next_value = self.policy_net(
                flat_state.unsqueeze(0), spatial_state
            )
            next_value = next_value.squeeze().item()

        trajectory = {
            "states": states_list,
            "actions": torch.tensor(actions_list, dtype=torch.long).to(device),
            "rewards": torch.tensor(rewards_list, dtype=torch.float32).to(device),
            "values": torch.tensor(values_list, dtype=torch.float32).to(device),
            "log_probs": torch.tensor(log_probs_list, dtype=torch.float32).to(device),
            "next_value": torch.tensor(next_value, dtype=torch.float32).to(device),
        }

        return trajectory

    def compute_gae(self, rewards, values, next_value):
        device = rewards.device
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, dtype=torch.float32).to(device)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def training_step(self, trajectory):
        device = next(self.policy_net.parameters()).device

        actions = trajectory["actions"].to(device)
        rewards = trajectory["rewards"].to(device)
        values = trajectory["values"].to(device)
        log_probs_old = trajectory["log_probs"].to(device)
        next_value = trajectory["next_value"].to(device)

        advantages, returns = self.compute_gae(rewards, values, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_states = []
        spatial_states = []
        for state in trajectory["states"]:
            flat_states.append(state["flat"])
            spatial_states.append(state["spatial"])

        flat_states_t = torch.tensor(
            np.array(flat_states), dtype=torch.float32
        ).to(device)
        spatial_states_t = torch.tensor(
            np.array(spatial_states), dtype=torch.float32
        ).to(device)
        
        spatial_states_t = spatial_states_t.unsqueeze(1).repeat(1, 3, 1, 1)

        action_logits, placement_mu, placement_sigma, new_values = self.policy_net(
            flat_states_t, spatial_states_t
        )

        dist = torch.distributions.Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(actions)

        actor_loss = -(new_log_probs * advantages).mean()
        critic_loss = F.smooth_l1_loss(new_values.squeeze(), returns)

        total_loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss
