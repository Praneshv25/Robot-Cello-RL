import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path

from models.gp_gate import GPGate


# ---------------------------------------------------------------------------
# Observation layout
# obs: [tcp(6), ft(6), speed, accel, bow_pos, force] — 16-dim
# act: scalar force in [force_min, force_max]
# ---------------------------------------------------------------------------

OBS_DIM = 16
ACT_DIM = 1

IDX_TCP   = slice(0, 6)
IDX_FT    = slice(6, 12)
IDX_SPEED = 12
IDX_ACCEL = 13
IDX_BOW   = 14
IDX_FORCE = 15

LOG_STD_MIN = -5
LOG_STD_MAX  = 2


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0
        self.obs      = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.actions  = np.zeros((capacity, ACT_DIM), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),       dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = float(reward)
        self.dones[self.ptr]    = float(done)
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

class GaussianActor(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden, ACT_DIM)
        self.log_std_layer = nn.Linear(hidden, ACT_DIM)

    def _dist(self, obs):
        x       = self.net(obs)
        mean    = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, obs):
        """Stochastic action in [-1, 1] + log-prob (for gradient updates)."""
        mean, std = self._dist(obs)
        u         = torch.distributions.Normal(mean, std).rsample()
        action    = torch.tanh(u)
        log_prob  = (
            torch.distributions.Normal(mean, std).log_prob(u)
            - torch.log(1.0 - action.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        return action, log_prob

    def select_action(self, obs, deterministic: bool = False):
        """Deterministic or stochastic action in [-1, 1], no gradient."""
        with torch.no_grad():
            mean, std = self._dist(obs)
            if deterministic:
                return torch.tanh(mean)
            return torch.tanh(torch.distributions.Normal(mean, std).sample())


class DoubleCritic(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        def _mlp():
            return nn.Sequential(
                nn.Linear(OBS_DIM + ACT_DIM, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),              nn.ReLU(),
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
    """
    Soft Actor-Critic.

    Actions are force values in [force_min, force_max] N (read from config).
    The actor internally works in tanh-space [-1, 1]; select_action() and
    store() handle the bijective scaling so the replay buffer and networks
    stay consistent.

    No hardware interaction: select_action() returns the commanded force and
    the caller (training loop) is responsible for applying it and collecting
    the next observation and reward.
    """

    def __init__(self, gp: GPGate, config: dict):
        force_cfg = config['policy_params']['force']
        self.force_min   = float(force_cfg['min'])
        self.force_max   = float(force_cfg['max'])
        self._force_mid  = (self.force_max + self.force_min) / 2.0
        self._force_half = (self.force_max - self.force_min) / 2.0

        scfg = config['sac']
        self.device = torch.device("cpu")

        self.gp = gp

        hidden = scfg['hidden']
        self.actor         = GaussianActor(hidden).to(self.device)
        self.critic        = DoubleCritic(hidden).to(self.device)
        self.critic_target = DoubleCritic(hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        lr = scfg['lr']
        self.actor_opt  = Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha          = self.log_alpha.exp().item()
        self.alpha_opt      = Adam([self.log_alpha], lr=lr)
        self.target_entropy = float(-ACT_DIM)

        self.gamma      = scfg['gamma']
        self.tau        = scfg['tau']
        self.batch_size = scfg['batch_size']
        self.buffer     = _ReplayBuffer(scfg['buffer_size'])

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> float:
        """Return commanded force in [force_min, force_max] N."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        raw   = self.actor.select_action(obs_t, deterministic).item()
        return float(np.clip(
            self._force_mid + self._force_half * raw,
            self.force_min, self.force_max,
        ))

    def random_action(self) -> float:
        """Uniform random force for warm-up exploration."""
        return float(np.random.uniform(self.force_min, self.force_max))

    # ------------------------------------------------------------------
    # Safety gate
    # ------------------------------------------------------------------

    def is_safe(self, obs: np.ndarray) -> bool:
        return self.gp.is_approved(obs[IDX_FT], obs[IDX_TCP])

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store(self, obs, force: float, reward, next_obs, done):
        """Store a transition. force must be in [force_min, force_max] N."""
        raw = np.array(
            [(force - self._force_mid) / self._force_half],
            dtype=np.float32,
        )
        self.buffer.add(obs, raw, reward, next_obs, done)

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
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha.data,
        }, path + ".pt")

    def load(self, path: str):
        ckpt = torch.load(path + ".pt", map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.alpha = self.log_alpha.exp().item()
