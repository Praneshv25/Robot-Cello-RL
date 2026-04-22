"""
infer.py

Run a trained SAC policy for one note stroke on the live robot.

    obs = env.reset()
    while not done:
        action          = agent.select_action(obs)   # delta force in [-1, 1]
        obs, _, done, _ = env.step(action)           # applies force, reads RTDE, scores audio

Usage:
    python -m rl.sac.infer --model checkpoints/sac_final
    python -m rl.sac.infer --model checkpoints/sac_ep0200 --steps 150 --note A
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.sac.cello_env import CelloEnv
from rl.sac.sac_agent import SAC
from reward.classifier import SoundClassifier


def play_note(agent: SAC, env: CelloEnv, verbose: bool = True) -> dict:
    obs  = env.reset()
    done = False
    info = {}

    while not done:
        action             = agent.select_action(obs, deterministic=True)
        obs, _, done, info = env.step(action)

        if verbose:
            print(
                f"  step {info['step']:03d} | "
                f"force {info['force']:.3f} N | "
                f"score {info['score']:.3f} | "
                f"reward {info['reward']:+.3f}"
            )

    return info


def main(args):
    model_path = args.model
    if not Path(model_path + ".pt").exists():
        raise FileNotFoundError(f"No checkpoint at {model_path}.pt")

    env   = CelloEnv(classifier=SoundClassifier(), max_steps=args.steps)
    agent = SAC(obs_dim=CelloEnv.OBS_DIM, act_dim=1)
    agent.load(model_path)
    print(f"Loaded: {model_path}.pt\n")

    print(f"Playing note {args.note} for up to {args.steps} steps ...\n")
    info = play_note(agent, env, verbose=args.verbose)
    print(f"\nDone. force={info['force']:.3f} N | score={info['score']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="checkpoints/sac_final",
                        help="Checkpoint path (omit .pt)")
    parser.add_argument("--note",    type=str, default="A", choices=["A", "D", "G", "C"])
    parser.add_argument("--steps",   type=int, default=150)
    parser.add_argument("--verbose", action="store_true", default=True)
    main(parser.parse_args())
