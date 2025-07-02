import sys
import os
import importlib
# Ensure numpy.core.numeric is loaded if stable_baselines3 has issues
sys.modules['numpy._core.numeric'] = importlib.import_module('numpy.core.numeric')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Import your modified environment
from rl_csv_trajectory import UR5eCelloTrajectoryEnv, extract_joint_angles_and_timestamps

# Assuming parsemidi is available
from parsemidi import parse_midi
import pandas as pd

# --- Configuration ---
# PATHS
# Adjust these paths to where your files are located
BASELINE_LOG_CSV = '/Users/skamanski/Documents/GitHub/Robot-Cello/biglogs/minuet_no_2v2-log-detailed.csv' # Output from URSim logging
MIDI_FILE_PATH = '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/MIDI-Files/minuet_no_2v2.mid' # Your MIDI file
MUJOCO_MODEL_PATH = '/Users/skamanski/Documents/GitHub/Robot-Cello/MuJoCo_RL_UR5/env_experiment/universal_robots_ur5e/scene.xml'

LOG_DIR = './ppo_residual_logs/' # Directory for RL training logs and models
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(model_path, baseline_csv_path, note_sequence, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        # Load initial joint positions from the start of the baseline trajectory
        # This will be passed to the environment for consistent starting point
        baseline_df = pd.read_csv(baseline_csv_path)
        start_joint_positions = [baseline_df[f'q_{joint}'].iloc[0] for joint in ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']]

        env = UR5eCelloTrajectoryEnv(
            model_path=model_path,
            baseline_csv_path=baseline_csv_path,
            note_sequence=note_sequence,
            render_mode=None, # <-- CHANGE THIS: Set to None for training environments
            action_scale=0.01, # Scale of residual actions (tune carefully!)
            residual_penalty=0.005,
            contact_penalty=0.5, # High penalty for unwanted contact
            torque_penalty=0.0001,
            perpendicularity_penalty=5.0, # Strong penalty for non-perpendicular bowing
            timing_penalty=2.0,           # Penalty for deviation from baseline timing
            kp=150.0, # PID gains - tune these based on robot response in MuJoCo
            kd=5.0,
            ki=0.2,
            start_joint_positions=start_joint_positions,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == '__main__':
    print("Starting Residual RL Training...")

    # 1. Parse MIDI to get note sequence for musical context
    print(f"Parsing MIDI file: {MIDI_FILE_PATH}")
    note_sequence = parse_midi(MIDI_FILE_PATH)
    if not note_sequence:
        print("Warning: No notes parsed from MIDI file.")
    
    # 2. Extract initial joint positions from baseline CSV for environment reset
    
    # 3. Create VecEnv for Stable Baselines3
    num_envs = 1 # Number of parallel environments for faster training
    print(f"Creating {num_envs} parallel environments...")
    env = DummyVecEnv([make_env(MUJOCO_MODEL_PATH, BASELINE_LOG_CSV, note_sequence, i) for i in range(num_envs)])

    # 4. Define PPO Model
    print("Initializing PPO model...")
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        tensorboard_log=LOG_DIR
    )

    # 5. Callbacks for logging and saving models
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=LOG_DIR,
        name_prefix="ur5e_cello_residual_ppo"
    )
    # The eval_env correctly uses render_mode="human" to visualize evaluation.
    eval_env = UR5eCelloTrajectoryEnv(
        model_path=MUJOCO_MODEL_PATH,
        baseline_csv_path=BASELINE_LOG_CSV,
        note_sequence=note_sequence,
        render_mode="human", # This one is intended to render the evaluation
        action_scale=0.01, residual_penalty=0.005, contact_penalty=0.5, torque_penalty=0.0001,
        perpendicularity_penalty=5.0, timing_penalty=2.0,
        kp=150.0, kd=5.0, ki=0.2,
        start_joint_positions=[env.envs[0].data.qpos[:6].tolist()] # Get a valid start pos
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=LOG_DIR,
        eval_freq=50000,
        deterministic=True,
        render=False # Set render=False here for EvalCallback itself, as you're rendering via eval_env's own render_mode
    )

    # 6. Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=1000000,
        callback=[checkpoint_callback, eval_callback]
    )

    # 7. Save the final model
    model.save(os.path.join(LOG_DIR, "ur5e_cello_residual_ppo_final"))
    print("Training finished. Model saved.")

