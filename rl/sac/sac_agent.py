"""
sac_agent.py

Pure PyTorch SAC — no stable-baselines3, no gymnasium.

Components:
  ReplayBuffer   — circular experience store
  GaussianActor  — squashed-Gaussian policy (reparameterisation trick)
  DoubleCritic   — twin Q-networks
  SAC            — agent: select_action / store / update / save / load
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0

        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),       dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)

    def _dist(self, obs):
        x       = self.net(obs)
        mean    = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, obs):
        mean, std = self._dist(obs)
        dist      = torch.distributions.Normal(mean, std)
        u         = dist.rsample()
        action    = torch.tanh(u)
        log_prob  = (dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob

    def select_action(self, obs, deterministic: bool = False):
        with torch.no_grad():
            mean, std = self._dist(obs)
            if deterministic:
                return torch.tanh(mean)
            u = torch.distributions.Normal(mean, std).sample()
            return torch.tanh(u)


class DoubleCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        def _mlp():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),             nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs, action):
        q1, q2 = self(obs, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# SAC agent
# ---------------------------------------------------------------------------

class SAC:
    def __init__(
        self,
        obs_dim:        int,
        act_dim:        int,
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        tau:            float = 0.005,
        auto_alpha:     bool  = True,
        target_entropy: float = None,
        buffer_size:    int   = 100_000,
        batch_size:     int   = 256,
        hidden:         int   = 256,
        device:         str   = "cpu",
    ):
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.device     = torch.device(device)
        self.auto_alpha = auto_alpha

        self.actor         = GaussianActor(obs_dim, act_dim, hidden).to(self.device)
        self.critic        = DoubleCritic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = DoubleCritic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt  = Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        if auto_alpha:
            self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha          = self.log_alpha.exp().item()
            self.alpha_opt      = Adam([self.log_alpha], lr=lr)
            self.target_entropy = target_entropy if target_entropy is not None else float(-act_dim)
        else:
            self.alpha = 0.2

        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor.select_action(obs_t, deterministic)
        return action.cpu().numpy().flatten()

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, float(reward), next_obs, float(done))

    def update(self) -> dict:
        if self.buffer.size < self.batch_size:
            return {}

        obs, action, reward, next_obs, done = self.buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_obs)
            q_next   = self.critic_target.q_min(next_obs, next_action)
            q_target = reward + (1.0 - done) * self.gamma * (q_next - self.alpha * next_log_pi)

        q1, q2      = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_action, log_pi = self.actor.sample(obs)
        actor_loss = (self.alpha * log_pi - self.critic.q_min(obs, new_action)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_tgt.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_tgt.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
        if self.auto_alpha:
            payload["log_alpha"] = self.log_alpha.data
        torch.save(payload, path + ".pt")

    def load(self, path: str):
        ckpt = torch.load(path + ".pt", map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        if self.auto_alpha and "log_alpha" in ckpt:
            self.log_alpha.data = ckpt["log_alpha"]
            self.alpha = self.log_alpha.exp().item()
