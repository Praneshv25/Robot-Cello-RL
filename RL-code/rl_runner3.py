# In your geminiRunv2.py (or your main RL training script)

import time, sys, importlib
sys.modules['numpy._core.numeric'] = importlib.import_module('numpy.core.numeric')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_trajectory import UR5eCelloTrajectoryEnv, Preprocessor, BehavioralCloningModel # Import the Preprocessor and BC model
from parsemidi import parse_midi
import pandas as pd
import torch.nn as nn 


def extract_joint_angles(csv_filename):
    df = pd.read_csv(csv_filename)
    cols = ['q_base','q_shoulder','q_elbow','q_wrist1','q_wrist2','q_wrist3']
    return df[cols].values.tolist()
def extract_joint_angles_and_timestamps(csv_filename):
    df = pd.read_csv(csv_filename)
    cols = ['q_base','q_shoulder','q_elbow','q_wrist1','q_wrist2','q_wrist3']
    return df[cols].values.tolist(), df["timestamp_robot"].values.tolist()




if __name__ == '__main__':
    # Define paths for your data and model
    # Make sure these CSV_FILES paths match where your detailed logs are saved
    CSV_FILES = [
    '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/allegro-log-detailed-test.csv',
    '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/twinkle-log-detailed-test.csv',
    '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/perpetual-log-detailed-test.csv',
    '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/long-log-detailed-test.csv',
    '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/RL-code/minuet-log-detailed-test.csv',
    ]   

    # You still need to extract a `trajectory` and `note_sequence` to pass to the environment
    # The `trajectory` can be the overall desired path, or just an initial joint state.
    # For residual RL, `self.trajectory` in the env often acts as a reference.
    
    # For demonstration, let's use the first CSV's trajectory as the env's reference
    # You might want to combine trajectories or use a single "ideal" one.
    example_trajectory, _ = extract_joint_angles_and_timestamps(CSV_FILES[0]) # Assuming your log_runner extract function
    
    # If using `parse_midi`, ensure `your_midi_file.mid` exists
    # this is what i need to change to add in bowing
    notes_sequence = parse_midi('/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/MIDI-Files/allegro.mid') # Replace with actual MIDI file

    # Dummy notes for now if you don't have a MIDI file set up yet
    #notes_sequence = [{"note_number": 60, "duration": 1.0, "start_time": 0.0}]


    scene_path = '/Users/skamanski/Documents/GitHub/Robot-Cello/MuJoCo_RL_UR5/env_experiment/universal_robots_ur5e/scene.xml'
    # Ensure this path is correct for your MuJoCo model

    # Instantiate the environment with BC model paths
    def make_env():
        return UR5eCelloTrajectoryEnv(
            model_path=scene_path,
            trajectory=example_trajectory, # Provide a reference trajectory for the env
            note_sequence=notes_sequence, # Provide note sequence for musical context
            render_mode='human', # Or None for faster training
            action_scale=0.01, # This is the MAX RESIDUAL action (e.g., 0.01 radians +/-)
            residual_penalty=0.02,
            contact_penalty=0.1,
            torque_penalty=0.001,
            kp=100.0, kd=2.0, ki=0.1,
            start_joint_positions=example_trajectory[0],
            # BC model specific parameters
            bc_policy_path="bc_policy.pth",
            bc_scalers_path="bc_scalers.pkl",
            bc_encoders_path="bc_encoders.pkl",
            bc_hidden_size=256, # Match training
        )

    # For stable_baselines3, it's good practice to use a vectorized environment
    env = DummyVecEnv([make_env])

    # Initialize the PPO agent
    # The policy MlpPolicy automatically infers input/output dims from env.observation_space/action_space
    model = PPO("MlpPolicy", env, verbose=1,
                n_steps=2048, # Number of steps to collect per environment before updating the network
                batch_size=64, # Batch size for policy and value updates
                n_epochs=10, # Number of epochs when optimizing the surrogate loss
                gamma=0.99, # Discount factor
                gae_lambda=0.95, # Factor for Generalized Advantage Estimator
                clip_range=0.2, # Clipping parameter for PPO
                ent_coef=0.01, # Entropy coefficient for exploration
                learning_rate=0.0003, # Learning rate for the Adam optimizer
                # Add more parameters as needed
               )

    # Train the agent
    print("Starting Residual RL training...")
    model.learn(total_timesteps=1_000_000) # Train for 1 million time steps
    print("Training finished.")

    # Save the trained RL model
    model.save("residual_ppo_policy")
    print("Residual PPO policy saved.")

    # You can then load and test the policy
    # loaded_model = PPO.load("residual_ppo_policy")
    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = loaded_model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()

    env.close()