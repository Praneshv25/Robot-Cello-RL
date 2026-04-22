"""
cello_env.py

Live CelloEnv — no gymnasium, no stable-baselines3.

Observation (16-dim):
    tcp_pose       [0:6]   — x, y, z (m), rx, ry, rz (rad)
    force_torque   [6:12]  — fx, fy, fz (N), tx, ty, tz (Nm)
    bow_speed      [12]    — m/s
    bow_accel      [13]    — m/s²
    bow_position   [14]    — 0.0 (frog) → 1.0 (tip)
    current_force  [15]    — commanded normal force (N)

Action (1-dim):
    delta_force    [0]     — in [-1, 1], scaled to ±max_delta per step

Episode = one note stroke (max_steps timesteps).
"""

import numpy as np
from reward.reward import Reward


class CelloEnv:
    # Observation layout
    IDX_TCP   = slice(0, 6)
    IDX_FT    = slice(6, 12)
    IDX_SPEED = 12
    IDX_ACCEL = 13
    IDX_BOW   = 14
    IDX_FORCE = 15
    OBS_DIM   = 16

    # Reward shaping weights
    W_CLASSIFIER  = 1.0
    W_FORCE_DELTA = 0.02  # penalise large force jumps
    W_FORCE_RANGE = 0.01  # penalise drifting far from force init

    def __init__(self, classifier, policy, max_steps: int = 150, audio_samples: int = 2205):
        """
        classifier — predict(audio: np.ndarray) -> float in [0, 1]
        policy     — models.policy.Policy; provides force limits
        """
        self.classifier    = classifier
        self.max_steps     = max_steps
        self.audio_samples = audio_samples

        # Force limits sourced from Policy (not hardcoded)
        self._force_min  = policy.min_force
        self._force_max  = policy.max_force
        self._force_init = policy._curr_force
        self._max_delta  = policy.max_delta

        self._force       = self._force_init
        self._step        = 0
        self._last_hw_obs = np.zeros(15, dtype=np.float32)  # exposed for SurrogateClassifier

    def set_classifier(self, classifier):
        self.classifier = classifier

    # ------------------------------------------------------------------
    # Hardware stubs — override in a subclass for real hardware
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return 15-dim state from RTDE: [tcp(6), ft(6), speed, accel, bow_pos]."""
        return np.zeros(15, dtype=np.float32)

    def _get_audio(self) -> np.ndarray:
        """Return float32 mono audio buffer (~50 ms) for the classifier."""
        return np.zeros(self.audio_samples, dtype=np.float32)

    def _apply_force(self, force: float):
        """Send commanded normal force to the robot."""
        pass

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._step  = 0
        self._force = self._force_init
        hw_obs      = self._get_obs()
        self._last_hw_obs = hw_obs
        return np.append(hw_obs, self._force).astype(np.float32)

    def step(self, action: np.ndarray):
        """
        action  — shape (1,), values in [-1, 1]
        returns (obs, reward, done, info)
        """
        prev_force  = self._force
        delta       = float(action[0]) * self._max_delta
        self._force = float(np.clip(self._force + delta, self._force_min, self._force_max))

        self._apply_force(self._force)

        hw_obs            = self._get_obs()
        self._last_hw_obs = hw_obs
        obs               = np.append(hw_obs, self._force).astype(np.float32)

        audio  = self._get_audio()
        score  = float(self.classifier.predict(audio))
        reward = self._compute_reward(score, prev_force)

        self._step += 1
        done = self._step >= self.max_steps

        info = {
            "step":   self._step,
            "force":  self._force,
            "score":  score,
            "reward": reward,
        }
        return obs, reward, done, info

    def sample_action(self) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, score: float, prev_force: float) -> float:
        r_class  = self.W_CLASSIFIER  * Reward.calculate_reward_from_score(score)
        r_smooth = -self.W_FORCE_DELTA * abs(self._force - prev_force)
        r_range  = -self.W_FORCE_RANGE * abs(self._force - self._force_init)
        return r_class + r_smooth + r_range
