import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path


# ---------------------------------------------------------------------------
# Observation layout — 16-dim state vector
# obs: [tcp(6), ft(6), speed(1), accel(1), bow_pos(1), force(1)]
#
# tcp      — 6DoF TCP position [x, y, z, rx, ry, rz]
# ft       — 6DoF force/torque [fx, fy, fz, tx, ty, tz]
# speed    — scalar magnitude of TCP velocity
# accel    — scalar magnitude of TCP acceleration
# bow_pos  — position along bow length (frog=0, tip=1)
# force    — current z-axis force being applied (N)
#
# act: scalar z-force in [force_min, force_max]
# ---------------------------------------------------------------------------

OBS_DIM = 16
ACT_DIM = 1

# index slices for unpacking state vector
IDX_TCP     = slice(0, 6)
IDX_FT      = slice(6, 12)
IDX_SPEED   = 12
IDX_ACCEL   = 13
IDX_BOW_POS = 14
IDX_FORCE   = 15

LOG_STD_MIN = -5
LOG_STD_MAX  = 2


# ---------------------------------------------------------------------------
# Replay buffer
# Internal to SAC — training loop does not manage this directly.
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0

        # pre-allocate arrays for efficiency
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

        # circular buffer — overwrite oldest transitions when full
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

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class GaussianActor(nn.Module):
    """
    Policy network. Takes 16-dim state, outputs a distribution over
    z-forces via mean and log_std. Actions are sampled from this
    distribution during training for exploration, and taken
    deterministically (mean) during evaluation.
    """
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        # separate heads for mean and log_std
        self.mean_layer    = nn.Linear(hidden, ACT_DIM)
        self.log_std_layer = nn.Linear(hidden, ACT_DIM)

    def _dist(self, obs):
        """Compute distribution parameters from observation."""
        x       = self.net(obs)
        mean    = self.mean_layer(x)

        # clamp log_std to prevent numerical instability
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, obs):
        """
        Stochastic action in [-1, 1] + log-prob.
        Used during gradient updates — rsample() allows gradients
        to flow through the sampling operation.
        tanh squashes action to [-1, 1] before scaling to force range.
        """
        mean, std = self._dist(obs)
        u         = torch.distributions.Normal(mean, std).rsample()
        action    = torch.tanh(u)

        # correct log_prob for tanh transformation
        log_prob  = (
            torch.distributions.Normal(mean, std).log_prob(u)
            - torch.log(1.0 - action.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        return action, log_prob

    def select_action(self, obs, deterministic: bool = False):
        """
        Returns action in [-1, 1], no gradient.
        Deterministic (mean) during evaluation, stochastic during training.
        Caller scales this to [force_min, force_max].
        """
        with torch.no_grad():
            mean, std = self._dist(obs)
            if deterministic:
                return torch.tanh(mean)
            return torch.tanh(torch.distributions.Normal(mean, std).sample())


class DoubleCritic(nn.Module):
    """
    Two Q-networks. Using two critics and taking the minimum
    reduces overestimation bias — a core part of SAC.
    """
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
        """Returns minimum of Q1 and Q2 — used to reduce overestimation."""
        q1, q2 = self(obs, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# SAC agent
# ---------------------------------------------------------------------------

class SAC:
    """
    Soft Actor-Critic.

    Owns the actor (policy), critics, replay buffer, and all update logic.
    GP gating is NOT handled here — that is the training loop's responsibility.
    
    Actions are z-force values in [force_min, force_max] N.
    Actor internally works in tanh-space [-1, 1].
    select_action() scales back to force range before returning.

    Interface for training_loop.py:
        force = sac.select_action(obs)          # get next force
        sac.store(obs, force, reward, next_obs, done)  # log transition
        sac.update()                            # update networks
        sac.save(path) / sac.load(path)         # checkpointing
    """

    def __init__(self, config: dict):
        """
        Note: GP is NOT passed in here. GP gating is handled
        entirely in training_loop.py — SAC has no knowledge of it.
        """
        force_cfg = config['policy_params']['force']
        self.force_min   = float(force_cfg['min'])
        self.force_max   = float(force_cfg['max'])

        # scaling factors for converting between tanh-space and force range
        self._force_mid  = (self.force_max + self.force_min) / 2.0
        self._force_half = (self.force_max - self.force_min) / 2.0

        scfg = config['sac']
        self.device = torch.device("cpu")

        # --- Networks ---
        hidden = scfg['hidden']
        self.actor         = GaussianActor(hidden).to(self.device)
        self.critic        = DoubleCritic(hidden).to(self.device)

        # target critic — slowly tracks critic via soft update (tau)
        # used for stable Q-value targets during updates
        self.critic_target = DoubleCritic(hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # target critic is never directly optimized — only updated via soft copy
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # --- Optimizers ---
        lr = scfg['lr']
        self.actor_opt  = Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        # --- Entropy temperature (alpha) ---
        # alpha balances exploration vs exploitation.
        # learned automatically via dual optimization.
        # target_entropy is a heuristic: -action_dim
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha          = self.log_alpha.exp().item()
        self.alpha_opt      = Adam([self.log_alpha], lr=lr)
        self.target_entropy = float(-ACT_DIM)

        # --- Hyperparameters ---
        self.gamma      = scfg['gamma']       # discount factor
        self.tau        = scfg['tau']          # soft update rate for target critic
        self.batch_size = scfg['batch_size']   # transitions sampled per update

        # --- Replay buffer ---
        # owned internally — training loop calls sac.store(), not buffer.add()
        self.buffer = _ReplayBuffer(scfg['buffer_size'])

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> float:
        """
        Returns commanded z-force in [force_min, force_max] N.
        Called by training_loop.py after sac.update() to get next force.
        
        deterministic=False during training (exploration via sampling)
        deterministic=True  during evaluation (use mean action)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        raw   = self.actor.select_action(obs_t, deterministic).item()

        # scale from tanh-space [-1, 1] to force range [force_min, force_max]
        return float(np.clip(
            self._force_mid + self._force_half * raw,
            self.force_min, self.force_max,
        ))

    def random_action(self) -> float:
        """
        Uniform random force for initial exploration before buffer
        has enough transitions for meaningful updates.
        """
        return float(np.random.uniform(self.force_min, self.force_max))

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------

    def store(self, obs: np.ndarray, force: float, reward: float,
              next_obs: np.ndarray, done: bool):
        """
        Store a transition in the replay buffer.
        Converts force from [force_min, force_max] to tanh-space [-1, 1]
        for consistency with how the actor represents actions internally.
        """
        raw = np.array(
            [(force - self._force_mid) / self._force_half],
            dtype=np.float32,
        )
        self.buffer.add(obs, raw, reward, next_obs, done)

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        One SAC gradient update step.
        Updates critic, actor, and entropy temperature (alpha).
        Returns dict of losses for logging — empty dict if buffer too small.

        Called by training_loop.py after every real robot interval.
        GP gating happens AFTER this call in the training loop,
        not inside here.
        """
        if self.buffer.size < self.batch_size:
            # not enough transitions yet — skip update
            return {}

        obs, action, reward, next_obs, done = self.buffer.sample(
            self.batch_size, self.device
        )

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_obs)
            q_next   = self.critic_target.q_min(next_obs, next_action)

            # Bellman target — entropy term (alpha * log_pi) encourages exploration
            q_target = reward + (1.0 - done) * self.gamma * (
                q_next - self.alpha * next_log_pi
            )

        q1, q2      = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor update ---
        new_action, log_pi = self.actor.sample(obs)
        actor_loss = (
            self.alpha * log_pi - self.critic.q_min(obs, new_action)
        ).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- Alpha (entropy temperature) update ---
        alpha_loss = -(
            self.log_alpha * (log_pi + self.target_entropy).detach()
        ).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # --- Soft update of target critic ---
        # slowly blend critic weights into target critic
        # prevents unstable Q-value targets
        for p, p_tgt in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
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
        """Save actor, critic, and alpha to disk for checkpointing."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha.data,
        }, path + ".pt")

    def load(self, path: str):
        """Load previously saved checkpoint to resume training."""
        ckpt = torch.load(path + ".pt", map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.alpha = self.log_alpha.exp().item()
