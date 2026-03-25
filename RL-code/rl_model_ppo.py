"""
PPO Reinforcement Learning Model for Robot Cello
Upgraded from REINFORCE to Proximal Policy Optimization

Key changes from REINFORCE:
1. Experience buffer — collects multiple bow strokes before learning
2. Critic network — predicts expected score for better advantage estimation
3. Ratio clipping — limits policy updates to ±20% per batch
4. Safety constraints — hard clipping + rate limiting (unchanged from v1)
"""

import numpy as np
import csv
from typing import Dict, List
from dataclasses import dataclass


# ============================================================
# Safety Limits (unchanged from REINFORCE version)
# ============================================================

FORCE_LIMITS = {
    "force_1": (-5.0, 15.0),    # Force X
    "force_2": (-10.0, 22.0),   # Force Y
    "force_3": (-5.0, 5.0),     # Force Z
    "force_4": (-1.0, 2.5),     # Force Rx
    "force_5": (-0.5, 0.5),     # Force Ry
    "force_6": (-2.0, 1.0),     # Force Rz
}

MAX_DELTA = {
    "force_1": 2.0,
    "force_2": 3.0,
    "force_3": 1.5,
    "force_4": 0.3,
    "force_5": 0.1,
    "force_6": 0.3,
}


# ============================================================
# WaveObject (unchanged)
# ============================================================

@dataclass
class WaveObject:
    data: np.ndarray
    sample_rate: int = 44100


# ============================================================
# SoundClassifier (unchanged — still placeholder)
# ============================================================

class SoundClassifier:
    def __init__(self):
        self.model = None

    def predict(self, wave: WaveObject) -> float:
        if wave.data is None or len(wave.data) == 0:
            return 0.5
        rms = np.sqrt(np.mean(wave.data ** 2))
        score = np.clip(rms, 0.0, 1.0)
        return float(score)


# ============================================================
# NEW: Simple Neural Network (for Critic)
# ============================================================

class SimpleNetwork:
    """
    A small 2-layer neural network using only numpy.
    Used as the Critic to predict expected reward given current forces.
    
    Architecture: 6 inputs → 32 hidden (ReLU) → 1 output
    """

    def __init__(self, input_size: int, hidden_size: int = 32, learning_rate: float = 0.001):
        self.lr = learning_rate

        # Initialize weights with small random values
        # Layer 1: input → hidden
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)

        # Layer 2: hidden → output (single value)
        self.W2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass: input → predicted value
        """
        # Layer 1 + ReLU activation
        self.z1 = x @ self.W1 + self.b1          # Linear transform
        self.a1 = np.maximum(0, self.z1)           # ReLU: negative → 0

        # Layer 2 (no activation — output can be any value)
        output = self.a1 @ self.W2 + self.b2

        # Store input for backward pass
        self.x = x
        return float(output[0])

    def update(self, x: np.ndarray, target: float):
        """
        One step of gradient descent to make prediction closer to target.
        """
        # Forward pass
        prediction = self.forward(x)
        error = prediction - target                # How far off were we?

        # Backward pass (chain rule)
        # Layer 2 gradients
        d_W2 = self.a1.reshape(-1, 1) * error
        d_b2 = np.array([error])

        # Layer 1 gradients (ReLU derivative: 1 if positive, 0 if negative)
        d_a1 = (self.W2.flatten() * error)
        d_z1 = d_a1 * (self.z1 > 0).astype(float)  # ReLU gradient
        d_W1 = self.x.reshape(-1, 1) @ d_z1.reshape(1, -1)
        d_b1 = d_z1

        # Update weights (gradient descent)
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1


# ============================================================
# NEW: PPO Policy (replaces REINFORCE RLPolicy)
# ============================================================

class PPOPolicy:
    """
    PPO Policy with:
    - Gaussian action sampling (same as REINFORCE)
    - Probability calculation (new — needed for ratio)
    - Ratio clipping (new — core of PPO)
    """

    def __init__(self, force_keys: list, learning_rate: float = 0.005):
        self.force_keys = force_keys
        self.learning_rate = learning_rate
        self.num_actions = len(force_keys)

        # Same as REINFORCE
        self.policy_mean = np.zeros(self.num_actions)
        self.policy_std = np.ones(self.num_actions) * 0.5

    def select_action(self, forces: Dict[str, float]) -> np.ndarray:
        """Same as REINFORCE — sample from Gaussian"""
        force_values = np.array([forces[key] for key in self.force_keys])
        action = (self.policy_mean
                  + force_values * 0.1
                  + self.policy_std * np.random.randn(self.num_actions))
        return action

    def get_log_prob(self, action: np.ndarray, forces: Dict[str, float]) -> float:
        """
        NEW: Calculate log probability of an action under current policy.
        
        This is how we measure "how much does the current policy like this action?"
        Used to compute the ratio: new_prob / old_prob
        
        Math: log probability of Gaussian = -0.5 * sum((action - mean)^2 / std^2) + constant
        """
        force_values = np.array([forces[key] for key in self.force_keys])
        mean = self.policy_mean + force_values * 0.1

        # Log probability of Gaussian distribution
        log_prob = -0.5 * np.sum(
            ((action - mean) / self.policy_std) ** 2
            + 2 * np.log(self.policy_std)
            + np.log(2 * np.pi)
        )
        return log_prob

    def update_ppo(self, action: np.ndarray, forces: Dict[str, float],
                   advantage: float, old_log_prob: float, epsilon: float = 0.2):
        """
        NEW: PPO update with ratio clipping.
        
        Instead of: policy_mean += lr * advantage * gradient  (REINFORCE)
        We do:      clip the update so policy doesn't change more than ±epsilon
        """
        # Step 1: Calculate current log probability
        new_log_prob = self.get_log_prob(action, forces)

        # Step 2: Calculate ratio
        # ratio = new_prob / old_prob = exp(new_log_prob - old_log_prob)
        ratio = np.exp(new_log_prob - old_log_prob)

        # Step 3: Clip the ratio to [1-epsilon, 1+epsilon]
        clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)

        # Step 4: Take the more conservative estimate
        # If advantage > 0 (good action): don't over-reinforce
        # If advantage < 0 (bad action): don't over-penalize
        surrogate1 = ratio * advantage
        surrogate2 = clipped_ratio * advantage
        loss = -min(surrogate1, surrogate2)

        # Step 5: Calculate gradient and update
        # Only update if not clipped (if clipped, gradient is 0)
        if abs(ratio - clipped_ratio) < 1e-6:  # Not clipped — normal update
            force_values = np.array([forces[key] for key in self.force_keys])
            gradient = (action - self.policy_mean - force_values * 0.1) / (self.policy_std ** 2)
            self.policy_mean += self.learning_rate * advantage * gradient


# ============================================================
# NEW: Experience Buffer
# ============================================================

class ExperienceBuffer:
    """
    Stores experiences from multiple bow strokes.
    PPO collects a batch before learning, unlike REINFORCE which learns immediately.
    """

    def __init__(self):
        self.buffer = []

    def add(self, forces: Dict[str, float], action: np.ndarray,
            reward: float, old_log_prob: float):
        """Store one bow stroke's experience"""
        self.buffer.append({
            'forces': forces.copy(),
            'action': action.copy(),
            'reward': reward,
            'old_log_prob': old_log_prob,
        })

    def size(self) -> int:
        return len(self.buffer)

    def get_all(self) -> List[dict]:
        return self.buffer

    def clear(self):
        self.buffer = []


# ============================================================
# PPO RL Model (replaces REINFORCE RLModel)
# ============================================================

class RLModel_PPO:
    """
    PPO-based RL Model for Robot Cello.
    
    Differences from REINFORCE version:
    - Uses PPOPolicy instead of RLPolicy
    - Uses Critic (SimpleNetwork) instead of fixed baseline
    - Collects experiences in buffer before learning
    - Learns multiple epochs per batch with ratio clipping
    - Safety constraints (_apply_safety) unchanged
    """

    def __init__(self, learning_rate: float = 0.005, batch_size: int = 20, n_epochs: int = 4):
        self.force_keys = ["force_1", "force_2", "force_3",
                           "force_4", "force_5", "force_6"]

        # Same as before
        self.classifier = SoundClassifier()

        # NEW: PPO components
        self.policy = PPOPolicy(self.force_keys, learning_rate)
        self.critic = SimpleNetwork(input_size=6, hidden_size=32, learning_rate=0.001)
        self.buffer = ExperienceBuffer()

        # PPO hyperparameters
        self.batch_size = batch_size   # How many bow strokes before learning
        self.n_epochs = n_epochs       # How many times to learn from same batch
        self.epsilon = 0.2             # Clip range for ratio

        # For safety constraints (unchanged)
        self.last_action = None
        self.last_output = None
        self.force_limits_low = np.array([FORCE_LIMITS[k][0] for k in self.force_keys])
        self.force_limits_high = np.array([FORCE_LIMITS[k][1] for k in self.force_keys])
        self.max_delta = np.array([MAX_DELTA[k] for k in self.force_keys])

    def _apply_safety(self, action: np.ndarray) -> np.ndarray:
        """Unchanged from REINFORCE version"""
        safe_action = action.copy()

        # Step 1: Rate limiting
        if self.last_output is not None:
            delta = safe_action - self.last_output
            delta = np.clip(delta, -self.max_delta, self.max_delta)
            safe_action = self.last_output + delta

        # Step 2: Hard clipping
        safe_action = np.clip(safe_action, self.force_limits_low, self.force_limits_high)

        return safe_action

    def forward(self, wave: WaveObject, forces: Dict[str, float]) -> Dict[str, float]:
        """
        Process one bow stroke.
        
        Unlike REINFORCE, does NOT update policy immediately.
        Instead, stores experience in buffer. Learning happens in learn().
        """
        assert len(forces) == 6, "Forces dictionary must contain exactly 6 entries"

        # ① Classifier scores the sound
        score = self.classifier.predict(wave)

        # ② Select action from current policy
        action = self.policy.select_action(forces)

        # ③ Calculate log probability (needed later for ratio)
        old_log_prob = self.policy.get_log_prob(action, forces)

        # ④ Apply safety constraints
        safe_action = self._apply_safety(action)

        # ⑤ Store experience in buffer (NOT learning yet)
        self.buffer.add(forces, action, score, old_log_prob)

        # ⑥ Remember for next step
        self.last_action = action
        self.last_output = safe_action

        # ⑦ Output safe action
        output = {key: float(safe_action[i]) for i, key in enumerate(self.force_keys)}
        return output

    def should_learn(self) -> bool:
        """Check if buffer has enough experiences to learn"""
        return self.buffer.size() >= self.batch_size

    def learn(self):
        """
        PPO learning: use buffered experiences to update policy and critic.
        This is the core difference from REINFORCE.
        
        REINFORCE: 1 bow → 1 update → discard
        PPO:       20 bows → 4 epochs of updates → then discard
        """
        experiences = self.buffer.get_all()

        for epoch in range(self.n_epochs):
            for exp in experiences:
                forces = exp['forces']
                action = exp['action']
                reward = exp['reward']
                old_log_prob = exp['old_log_prob']

                # Convert forces to array for critic
                force_array = np.array([forces[key] for key in self.force_keys])

                # Critic predicts: "in this situation, how much score do I expect?"
                predicted_value = self.critic.forward(force_array)

                # Advantage: "was this better or worse than expected?"
                advantage = reward - predicted_value

                # Update policy with PPO clipping
                self.policy.update_ppo(action, forces, advantage,
                                       old_log_prob, self.epsilon)

                # Update critic to predict better next time
                self.critic.update(force_array, reward)

        # Clear buffer after learning
        self.buffer.clear()

        return {
            'batch_size': len(experiences),
            'epochs': self.n_epochs,
        }

    def reset(self):
        """Reset model state for a new episode"""
        self.last_action = None
        self.last_output = None
        self.buffer.clear()


# ============================================================
# CSV Data Loading (unchanged)
# ============================================================

def load_csv(filepath: str) -> list:
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            forces = {
                "force_1": float(row["Force X"]),
                "force_2": float(row["Force Y"]),
                "force_3": float(row["Force Z"]),
                "force_4": float(row["Force Rx"]),
                "force_5": float(row["Force Ry"]),
                "force_6": float(row["Force Rz"]),
            }
            data.append({"forces": forces})
    return data


# ============================================================
# Test: Compare REINFORCE vs PPO
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PPO Robot Cello RL Model — Test Run")
    print("=" * 70)

    # Create model
    model = RLModel_PPO(
        learning_rate=0.005,
        batch_size=10,      # Learn after every 10 bow strokes
        n_epochs=4           # Learn 4 times from same batch
    )

    # Fake audio (classifier is still placeholder)
    sample_wave = WaveObject(data=np.random.randn(44100))

    # Fake forces for testing
    forces = {
        "force_1": 0.5, "force_2": 0.3, "force_3": 0.7,
        "force_4": 0.2, "force_5": 0.8, "force_6": 0.4
    }

    print(f"\nBatch size: {model.batch_size}")
    print(f"Epochs per batch: {model.n_epochs}")
    print(f"Clip epsilon: {model.epsilon}")
    print()

    # Run 30 bow strokes
    for i in range(30):
        output = model.forward(sample_wave, forces)

        # Check if buffer is full
        if model.should_learn():
            result = model.learn()
            print(f"Bow {i+1:>3}: LEARNED — "
                  f"batch={result['batch_size']}, "
                  f"epochs={result['epochs']}, "
                  f"policy_mean=[{', '.join(f'{m:.4f}' for m in model.policy.policy_mean)}]")
        else:
            print(f"Bow {i+1:>3}: Buffering... ({model.buffer.size()}/{model.batch_size})")

    # Final state
    print(f"\nFinal policy_mean: {model.policy.policy_mean}")
    print(f"Buffer size: {model.buffer.size()}")

    # Safety check
    print("\n" + "=" * 70)
    print("Safety Verification")
    print("=" * 70)
    model.reset()
    all_safe = True
    for i in range(100):
        output = model.forward(sample_wave, forces)
        for key in model.force_keys:
            val = output[key]
            lo, hi = FORCE_LIMITS[key]
            if val < lo or val > hi:
                print(f"  ✗ Step {i}: {key} = {val:.3f} out of range [{lo}, {hi}]")
                all_safe = False
        if model.should_learn():
            model.learn()

    if all_safe:
        print("✓ All 100 outputs within safety limits!")
