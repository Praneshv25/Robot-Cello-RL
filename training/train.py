import numpy as np
from pathlib import Path

from rl.sac.cello_env import CelloEnv
from rl.sac.sac_agent import SAC
from reward.classifier import SoundClassifier
from reward.reward import Reward


class SurrogateClassifier:
    """
    Wraps SurrogateModel to match the SoundClassifier interface.

    During the random-exploration phase, the real classifier may be noisy or
    uncalibrated. This substitute uses the surrogate MLP (pre-trained on
    warmup data) to predict sound score from the robot's current ft+tcp state,
    giving SAC a better reward signal from day one.

    Requires env._last_hw_obs to be populated each step (set in CelloEnv.step).
    """
    def __init__(self, surrogate, env: CelloEnv):
        self.surrogate = surrogate
        self.env       = env

    def predict(self, audio: np.ndarray) -> float:
        ft  = self.env._last_hw_obs[CelloEnv.IDX_FT]
        tcp = self.env._last_hw_obs[CelloEnv.IDX_TCP]
        return float(self.surrogate.predict_score(ft, tcp))


class Train:
    def __init__(self, gp, policy, surrogate, config):
        """
        gp        — GPGate  (models/gp_gate.py)
        policy    — Policy  (models/policy.py)  provides force limits
        surrogate — SurrogateModel (models/surrogate.py)  reward during exploration
        config    — dict from config.yaml
        """
        self.gp        = gp
        self.surrogate = surrogate
        tcfg  = config['training']
        scfg  = config['sac']
        gcfg  = config['gp']
        sucfg = config['surrogate']

        # Build env with surrogate classifier for the exploration phase
        self.surrogate_clf = SurrogateClassifier(surrogate, None)  # env linked below
        self.env = CelloEnv(
            classifier = self.surrogate_clf,
            policy     = policy,
            max_steps  = tcfg['max_steps_per_episode'],
        )
        self.surrogate_clf.env = self.env  # complete the circular reference

        self.real_clf = SoundClassifier()  # switched in after random phase

        self.agent = SAC(
            obs_dim     = CelloEnv.OBS_DIM,
            act_dim     = 1,
            lr          = scfg['lr'],
            gamma       = scfg['gamma'],
            tau         = scfg['tau'],
            buffer_size = scfg['buffer_size'],
            batch_size  = scfg['batch_size'],
            hidden      = scfg['hidden'],
            auto_alpha  = True,
        )

        self.save_dir         = Path(tcfg['checkpoint_dir'])
        self.episodes         = tcfg['episodes']
        self.random_steps     = tcfg['random_steps']
        self.save_every       = tcfg['save_every']
        self.gp_refit_every   = gcfg['refit_every']
        self.gp_threshold_fin = gcfg['threshold_final']
        self.sur_refit_every  = sucfg['refit_every']
        self.virtual_updates  = sucfg['virtual_updates']

    # ------------------------------------------------------------------

    def _safe_step(self, obs, action):
        """Gate each action through the GP; hold force constant if rejected."""
        if not self.gp.is_approved(obs[CelloEnv.IDX_FT], obs[CelloEnv.IDX_TCP]):
            action = np.zeros(1, dtype=np.float32)
        return self.env.step(action)

    def train(self):
        self.save_dir.mkdir(exist_ok=True)
        total_steps     = 0
        using_surrogate = True

        print(f"Starting training — {self.episodes} episodes")
        print(f"Surrogate classifier active for first {self.random_steps} steps\n")

        for episode in range(1, self.episodes + 1):

            # Switch from surrogate to real classifier once exploration is done
            if using_surrogate and total_steps >= self.random_steps:
                self.env.set_classifier(self.real_clf)
                using_surrogate = False
                print("  [classifier] switched surrogate → real SoundClassifier")

            obs            = self.env.reset()
            episode_reward = 0.0
            done           = False

            while not done:
                if total_steps < self.random_steps:
                    action = self.env.sample_action()
                else:
                    action = self.agent.select_action(obs)

                next_obs, reward, done, info = self._safe_step(obs, action)

                self.agent.store(obs, action, reward, next_obs, done)
                self.agent.update()

                # Virtual updates: K extra gradient steps from the replay buffer.
                # During the surrogate phase the buffer holds surrogate-predicted
                # rewards, giving SAC more signal per real robot step.
                for _ in range(self.virtual_updates):
                    self.agent.update()

                # Feed real observation back into the surrogate so it keeps learning
                self.surrogate.add_observation(
                    force = obs[CelloEnv.IDX_FT],
                    tcp   = obs[CelloEnv.IDX_TCP],
                    score = info['score'],
                )

                # GP grows its knowledge of visited (ft, tcp) regions
                self.gp.add_observation(
                    force = obs[CelloEnv.IDX_FT],
                    tcp   = obs[CelloEnv.IDX_TCP],
                )

                obs             = next_obs
                episode_reward += reward
                total_steps    += 1

            print(
                f"ep {episode:04d} | "
                f"steps {info['step']:3d} | "
                f"reward {episode_reward:+7.3f} | "
                f"force {info['force']:.3f} N | "
                f"score {info['score']:.3f}"
            )

            # Refit surrogate on accumulated real observations
            if episode % self.sur_refit_every == 0:
                self.surrogate.fit()
                print(f"  [surrogate] refit on {len(self.surrogate.training_data)} samples")

            # Refit GP and linearly anneal its safety threshold
            if episode % self.gp_refit_every == 0:
                self.gp.fit()
                progress      = episode / self.episodes
                new_threshold = self.gp.threshold - progress * (
                    self.gp.threshold - self.gp_threshold_fin
                )
                self.gp.anneal_threshold(new_threshold)
                print(f"  [gp] refit | threshold → {new_threshold:.3f}")

            if episode % self.save_every == 0:
                ckpt = str(self.save_dir / f"sac_ep{episode:04d}")
                self.agent.save(ckpt)
                print(f"  [ckpt] {ckpt}.pt")

        final = str(self.save_dir / "sac_final")
        self.agent.save(final)
        print(f"\nDone. {final}.pt")
