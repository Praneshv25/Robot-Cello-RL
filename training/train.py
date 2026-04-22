import numpy as np
from pathlib import Path

from rl.sac.cello_env import CelloEnv
from rl.sac.sac_agent import SAC
from reward.classifier import SoundClassifier


class Train:
    def __init__(self, gp, config):
        self.gp   = gp
        tcfg      = config['training']
        scfg      = config['sac']

        self.env = CelloEnv(
            classifier = SoundClassifier(),
            max_steps  = tcfg['max_steps_per_episode'],
        )
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
        self.save_dir     = Path(tcfg['checkpoint_dir'])
        self.episodes     = tcfg['episodes']
        self.random_steps = tcfg['random_steps']
        self.save_every   = tcfg['save_every']

    def _safe_step(self, obs, action):
        """Gate each action through the GP; hold force constant if rejected."""
        tcp = obs[CelloEnv.IDX_TCP]
        ft  = obs[CelloEnv.IDX_FT]
        if not self.gp.is_approved(ft, tcp):
            action = np.zeros(1, dtype=np.float32)  # zero delta = hold current force
        return self.env.step(action)

    def train(self):
        self.save_dir.mkdir(exist_ok=True)
        total_steps = 0

        print(f"Starting training — {self.episodes} episodes")
        print(f"Random exploration for first {self.random_steps} steps\n")

        for episode in range(1, self.episodes + 1):
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

                # expand GP knowledge with each real observation
                self.gp.add_observation(force=obs[CelloEnv.IDX_FT], tcp=obs[CelloEnv.IDX_TCP])

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

            if episode % self.save_every == 0:
                ckpt = str(self.save_dir / f"sac_ep{episode:04d}")
                self.agent.save(ckpt)
                print(f"  -> {ckpt}.pt")

        final = str(self.save_dir / "sac_final")
        self.agent.save(final)
        print(f"\nDone. {final}.pt")
