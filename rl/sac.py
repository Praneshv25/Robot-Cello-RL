"""
sac.py — SAC algorithm implementation

External dependencies (placeholders point to where they live):
  hardware/rtde.py            — get_obs(), apply_force()
  hardware/audio_interface.py — get_audio()
  reward/classifier.py        — SoundClassifier
  reward/reward.py            — Reward
  models/policy.py            — Policy   (force limits)
  models/gp_gate.py           — GPGate   (safety gate)
  models/surrogate.py         — SurrogateModel (Dyna rewards)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path

from reward.classifier import SoundClassifier
from reward.reward import Reward
from models.policy import Policy
from models.gp_gate import GPGate
from models.surrogate import SurrogateModel


# ---------------------------------------------------------------------------
# Hardware placeholders
# Implementations live in hardware/rtde.py and hardware/audio_interface.py
# ---------------------------------------------------------------------------

def get_obs() -> np.ndarray:
    """15-dim robot state: [tcp(6), ft(6), speed, accel, bow_pos]. → hardware/rtde.py"""
    raise NotImplementedError

def apply_force(force: float):
    """Send commanded normal force to robot via RTDE. → hardware/rtde.py"""
    raise NotImplementedError

def get_audio() -> np.ndarray:
    """Return float32 mono audio buffer (~50 ms). → hardware/audio_interface.py"""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------

OBS_DIM = 16
ACT_DIM = 1

IDX_TCP   = slice(0, 6)
IDX_FT    = slice(6, 12)
IDX_SPEED = 12
IDX_ACCEL = 13
IDX_BOW   = 14
IDX_FORCE = 15

W_CLASSIFIER  = 1.0
W_FORCE_DELTA = 0.02
W_FORCE_RANGE = 0.01


# ---------------------------------------------------------------------------
# Replay buffer (SAC-internal — off-policy requires experience replay)
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

LOG_STD_MIN = -5
LOG_STD_MAX = 2


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
        """Stochastic action + log-prob for training."""
        mean, std = self._dist(obs)
        u         = torch.distributions.Normal(mean, std).rsample()
        action    = torch.tanh(u)
        log_prob  = (
            torch.distributions.Normal(mean, std).log_prob(u)
            - torch.log(1.0 - action.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        return action, log_prob

    def select_action(self, obs, deterministic: bool = False):
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
    def __init__(
        self,
        policy:    Policy,
        gp:        GPGate,
        surrogate: SurrogateModel,
        config:    dict,
    ):
        scfg = config['sac']
        self.device = torch.device("cpu")

        # Force limits from Policy (models/policy.py)
        self._force_min  = policy.min_force
        self._force_max  = policy.max_force
        self._force_init = policy._curr_force
        self._max_delta  = policy.max_delta

        self.gp        = gp           # models/gp_gate.py
        self.surrogate = surrogate    # models/surrogate.py
        self.classifier = SoundClassifier()   # reward/classifier.py

        self.actor         = GaussianActor(scfg['hidden']).to(self.device)
        self.critic        = DoubleCritic(scfg['hidden']).to(self.device)
        self.critic_target = DoubleCritic(scfg['hidden']).to(self.device)
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

        self._force     = self._force_init
        self._step      = 0
        self._last_hw_obs = np.zeros(15, dtype=np.float32)

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._step  = 0
        self._force = self._force_init
        hw_obs      = get_obs()               # hardware/rtde.py
        self._last_hw_obs = hw_obs
        return np.append(hw_obs, self._force).astype(np.float32)

    def step(self, action: np.ndarray):
        """
        Apply action on the live robot. Returns (next_obs, reward, done, info).
        Calls:
          apply_force()  → hardware/rtde.py
          get_obs()      → hardware/rtde.py
          get_audio()    → hardware/audio_interface.py
          classifier     → reward/classifier.py
          Reward         → reward/reward.py
        """
        prev_force  = self._force
        delta       = float(action[0]) * self._max_delta
        self._force = float(np.clip(self._force + delta, self._force_min, self._force_max))

        apply_force(self._force)              # hardware/rtde.py

        hw_obs            = get_obs()         # hardware/rtde.py
        self._last_hw_obs = hw_obs
        obs               = np.append(hw_obs, self._force).astype(np.float32)

        audio  = get_audio()                  # hardware/audio_interface.py
        score  = float(self.classifier.predict(audio))   # reward/classifier.py
        reward = self._compute_reward(score, prev_force)  # reward/reward.py

        self._step += 1
        done = self._step >= self._max_steps

        info = {"step": self._step, "force": self._force, "score": score, "reward": reward}
        return obs, reward, done, info

    def sample_action(self) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)

    # ------------------------------------------------------------------
    # Safety gate (models/gp_gate.py)
    # ------------------------------------------------------------------

    def is_safe(self, obs: np.ndarray) -> bool:
        return self.gp.is_approved(obs[IDX_FT], obs[IDX_TCP])

    # ------------------------------------------------------------------
    # Surrogate reward (models/surrogate.py)
    # ------------------------------------------------------------------

    def surrogate_score(self, obs: np.ndarray) -> float:
        """Predict sound score for obs without running the robot."""
        return float(self.surrogate.predict_score(obs[IDX_FT], obs[IDX_TCP]))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

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

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "alpha": self.alpha}

    # ------------------------------------------------------------------
    # Reward (reward/reward.py)
    # ------------------------------------------------------------------

    def _compute_reward(self, score: float, prev_force: float) -> float:
        r_class  = W_CLASSIFIER  * Reward.calculate_reward_from_score(score)
        r_smooth = -W_FORCE_DELTA * abs(self._force - prev_force)
        r_range  = -W_FORCE_RANGE * abs(self._force - self._force_init)
        return r_class + r_smooth + r_range

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
