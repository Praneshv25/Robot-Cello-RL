import gym
from gym import spaces
import mujoco
import contact
from parsemidi import parse_midi
import pandas as pd
import numpy as np
import time
import sys
import os
import mujoco.viewer 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Baseline-Runners')))
import robot_runner  # baseline controller
import torch.nn as nn  # Ensure PyTorch is imported for the model
import torch
import pickle

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder # Ensure these are imported

# --- Define your BC Model architecture (copy from newest_bc.py) ---
# It's crucial that this matches the model you saved
class BehavioralCloningModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(BehavioralCloningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu2(x) # Corrected from relu1
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# --- Static helper methods for preprocessing (copy from CelloTrajectoryDataset) ---
# Make sure these are static methods or standalone functions
# This is to correctly preprocess single observations from the environment.
class Preprocessor:
    # INPUT_FEATURE_COLS and TARGET_COLS should be accessible or passed.
    # For now, let's redefine them here for clarity or import them from a config file.
    INPUT_FEATURE_COLS = [
        'q_base', 'q_shoulder', 'q_elbow', 'q_wrist1', 'q_wrist2', 'q_wrist3',
        'TCP_pose_x', 'TCP_pose_y', 'TCP_pose_z', 'TCP_pose_rx', 'TCP_pose_ry', 'TCP_pose_rz',
        'time_elapsed_sec', 'remaining_duration_sec', 'current_note_number',
        'current_string', 'event_label', 'event_flag'
    ]
    TARGET_COLS = [
        'q_base', 'q_shoulder', 'q_elbow', 'q_wrist1', 'q_wrist2', 'q_wrist3'
    ]
    HIDDEN_SIZE = 256 # Should match what you used for BC training

    @staticmethod
    def _infer_bow_direction(event_label):
        if 'a_bow' in str(event_label).lower():
            return 'up'
        elif 'd_bow' in str(event_label).lower():
            return 'down'
        else:
            return 'none'

    @staticmethod
    def _infer_is_transition(event_flag, event_label):
        if 'TRANSITION' in str(event_label).upper() or not (1 <= event_flag <= 6):
            return 1
        else:
            return 0

    @staticmethod
    def preprocess_observation(raw_obs_dict, loaded_scalers, loaded_encoders, input_feature_cols):
        # Create a single-row DataFrame from the raw observation dictionary
        single_obs_df = pd.DataFrame([raw_obs_dict])

        # --- Handle current_note_number cleaning ---
        single_obs_df['current_note_number'] = pd.to_numeric(single_obs_df['current_note_number'], errors='coerce')
        single_obs_df['current_note_number'] = single_obs_df['current_note_number'].fillna(0)

        processed_features = []

        numerical_cols = [col for col in input_feature_cols if col not in ['current_string', 'event_label', 'event_flag']]
        for col in numerical_cols:
            if col in loaded_scalers:
                processed_features.append(loaded_scalers[col].transform(single_obs_df[[col]]))
            else:
                # Fallback if a scaler isn't found (shouldn't happen if setup correctly)
                processed_features.append(single_obs_df[[col]].values)

        # current_string one-hot encoding
        current_string_encoded = loaded_encoders['current_string'].transform(single_obs_df[['current_string']])
        processed_features.append(current_string_encoded) # Already dense numpy array

        # bow_direction inference and one-hot encoding
        bow_dir = Preprocessor._infer_bow_direction(single_obs_df['event_label'].iloc[0])
        bow_direction_encoded = loaded_encoders['bow_direction'].transform(np.array([[bow_dir]]))
        processed_features.append(bow_direction_encoded) # Already dense numpy array

        # is_transition inference
        is_transition_val = Preprocessor._infer_is_transition(single_obs_df['event_flag'].iloc[0], single_obs_df['event_label'].iloc[0])
        processed_features.append(np.array([[is_transition_val]]))

        # Stack horizontally and flatten for a single observation
        return np.hstack(processed_features).flatten()


class UR5eCelloTrajectoryEnv(gym.Env):
    """
    Residual-RL environment: learns to align the robot
    end-effector (TCP) to ideal linear bow paths rather
    than simply following a baseline trajectory.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(
        self,
        model_path: str,
        trajectory: list,
        note_sequence: list,
        render_mode=None,
        action_scale: float = 0.05,
        residual_penalty: float = 0.01,
        contact_penalty: float = 0.1,
        torque_penalty: float = 0.001,
        kp: float = 100.0,
        kd: float = 2.0,
        ki: float = 0.1,
        start_joint_positions=None,
        bc_policy_path: str = "bc_policy.pth",
        bc_scalers_path: str = "bc_scalers.pkl",
        bc_encoders_path: str = "bc_encoders.pkl",
        bc_hidden_size: int = 256,
    ):
        super().__init__()
        # --- MuJoCo setup ---
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = self.model.opt.timestep

        # --- RL hyperparameters ---
        # self.action_scale = action_scale # This line is replaced/repurposed below
        self.residual_penalty = residual_penalty
        self.contact_penalty = contact_penalty
        self.torque_penalty = torque_penalty

        # --- PID gains ---
        self.kp, self.kd, self.ki = kp, kd, ki
        self.total_pid_error = np.zeros(6)

        # self.base_ctrl = robot_runner.CelloController(...) # REMOVED THIS BLOCK
        # self.demo_traj = np.array(trajectory) # Keep this if used for reward calculation, otherwise remove

        # --- Musical notes & string mapping ---
        self.note_sequence = note_sequence


        # --- Load BC Model and Preprocessors ---
        self.bc_scalers = None
        self.bc_encoders = None
        self.bc_model = None
        try:
            with open(bc_scalers_path, 'rb') as f:
                self.bc_scalers = pickle.load(f)
            with open(bc_encoders_path, 'rb') as f:
                self.bc_encoders = pickle.load(f)

            # Determine input_dim for BC model from a dummy observation
            dummy_raw_obs_dict = {
                'q_base': 0.0, 'q_shoulder': 0.0, 'q_elbow': 0.0, 'q_wrist1': 0.0, 'q_wrist2': 0.0, 'q_wrist3': 0.0,
                'TCP_pose_x': 0.0, 'TCP_pose_y': 0.0, 'TCP_pose_z': 0.0, 'TCP_pose_rx': 0.0, 'TCP_pose_ry': 0.0, 'TCP_pose_rz': 0.0,
                'time_elapsed_sec': 0.0, 'remaining_duration_sec': 0.0, 'current_note_number': 60,
                'current_string': 'A', 'event_label': 'START a_bow', 'event_flag': 1
            }
            
            
            bc_input_dim = Preprocessor.preprocess_observation(
                dummy_raw_obs_dict, self.bc_scalers, self.bc_encoders, Preprocessor.INPUT_FEATURE_COLS
            ).shape[0]
            print(f"Calculated bc_input_dim for observation_space: {bc_input_dim}") # <-- Add this line
            bc_output_dim = len(Preprocessor.TARGET_COLS)

            self.bc_model = BehavioralCloningModel(bc_input_dim, bc_output_dim, bc_hidden_size)
            # ...
           
            self.bc_model.load_state_dict(torch.load(bc_policy_path))
            self.bc_model.eval() # Set to evaluation mode

            print(f"Successfully loaded BC model and preprocessors.")

        except Exception as e:
            print(f"Error loading BC model/preprocessors: {e}. Running without BC baseline.")
            self.bc_model = None # Ensure it's None if loading fails

        # --- Action Space for RL Agent (Residuals) ---
        # The RL agent will predict small deltas for each joint.
        # Max change per step for each joint. Adjust this based on how much correction
        # you expect the RL agent to make.
        # This will be in RADIANS for joint angles.
        self.action_space_range = action_scale # Renamed to be more explicit for residual
        self.action_space = spaces.Box(
            low=-self.action_space_range,
            high=self.action_space_range,
            shape=(len(Preprocessor.TARGET_COLS),), # 6 joints for UR5e
            dtype=np.float32,
        )

        # --- Observation Space for RL Agent (Same as BC input) ---
        self.observation_space = spaces.Box(
            low=-np.inf, # Scaled values can be negative
            high=np.inf, # Scaled values can be positive
            shape=(bc_input_dim,), # This is the preprocessed feature vector size
            dtype=np.float32,
        )

        # --- Tracking ---
        self.prev_torque = np.zeros(6)
        self.current_time = 0.0
        # self.current_idx = 0 # This might be less relevant if not directly following a fixed demo_traj index
        self.total_duration = len(self.note_sequence) * 0.5 # Example, adjust based on notes # You'll need to define how total_duration is calculated
        self.start_positions = start_joint_positions
        self.string_sites = {}
        for s in ('A','D','G','C'):
            frog = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_frog')
            tip  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_tip')
            self.string_sites[s] = (frog, tip)
        print(f'String sites: {self.string_sites}')

        # Add this block to initialize tcp_site_id
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "TCP_point")
        if self.tcp_site_id == -1:
            # It's good practice to raise an error if a critical site is not found
            print("Available sites in model:", self.model.site_names)
            raise ValueError("TCP_point site not found in MuJoCo model. Please check your XML model.")
        print(f'TCP site ID: {self.tcp_site_id}')
       

        # TODO : from each string site, compute the ideal string line
        self.string_lines = {}
        for site in self.string_sites.values():
            frog, tip = site
            p1 = self.data.site_xpos[frog]
            p2 = self.data.site_xpos[tip]
            string_line = p2 - p1
            norm = np.linalg.norm(string_line)
            if norm > 1e-6:
                string_line /= norm
            else:
                string_line[:] = 1.0  # or set to a default unit vector
            # allow for a vertical offset 

            self.string_lines[(frog, tip)] = string_line
        self.render_mode = render_mode
        self.viewer = None

        self.reset()

    def reset(self):
        # reset qpos/qvel
        if self.start_positions is not None:
            self.data.qpos[:6] = np.array(self.start_positions)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # reset trackers
        self.current_time = 0.0
        self.current_idx = 0
        self.total_pid_error = np.zeros(6)
        self.prev_torque = np.zeros(6)
        return self._get_obs()

    def step(self, action):
        # 'action' here is the RL agent's *residual* output (e.g., 6 values from -action_scale_range to +action_scale_range)

        # 1. Get current observation (preprocessed for BC model)
        current_processed_obs = self._get_obs() # This gets the scaled/encoded obs for BC
        obs_tensor = torch.tensor(current_processed_obs, dtype=torch.float32).unsqueeze(0) # Add batch dimension

        # 2. Get baseline action from BC model
        bc_action_scaled = None
        if self.bc_model is not None:
            with torch.no_grad():
                bc_action_tensor_scaled = self.bc_model(obs_tensor)
            # Convert BC output from scaled tensor to unscaled numpy array
            bc_action_unscaled = np.zeros(len(Preprocessor.TARGET_COLS))
            for i, col in enumerate(Preprocessor.TARGET_COLS):
                val_scaled = bc_action_tensor_scaled.squeeze(0).numpy()[i]
                # Inverse transform each joint angle individually
                bc_action_unscaled[i] = self.bc_scalers[col].inverse_transform(np.array([[val_scaled]]))[0][0]
        else:
            # Fallback if BC model not loaded: use the current joint positions as baseline,
            # or implement a simple non-learning baseline if needed.
            # For true Residual RL, the BC model should always be available.
            bc_action_unscaled = np.array(self.data.qpos[:6]) # Using current qpos as a "do nothing" baseline

        # 3. Combine BC baseline with RL residual
        # Ensure 'action' from RL agent is a numpy array of shape (6,)
        rl_residual_unscaled = action # The RL action is already unscaled residual in radians
                                    # if your action_space is defined in radians.
                                    # If your RL agent predicts scaled residuals, you'd need to inverse scale them here.
                                    # Assuming action is already in radians (actual delta).

        # Ensure output is within joint limits after adding residual
        # Joint limits are `self.model.jnt_range`. Check the order!
        # For UR5e, q_base, q_shoulder, q_elbow, q_wrist1, q_wrist2, q_wrist3 are usually first 6.
        
        # Calculate desired total joint positions
        desired_q = bc_action_unscaled + rl_residual_unscaled
        
        # Clamp to model's joint limits
        for i in range(len(desired_q)):
            joint_id = self.model.joint_name2id(Preprocessor.TARGET_COLS[i].replace('q_', 'UR5e_joint_')) # Adjust name mapping
            low_limit = self.model.jnt_range[joint_id, 0]
            high_limit = self.model.jnt_range[joint_id, 1]
            desired_q[i] = np.clip(desired_q[i], low_limit, high_limit)

        # 4. Apply combined action using your PID controller
        self._apply_pid_control(desired_q) # This method should exist in your env

        # 5. Simulate for one step
        mujoco.mj_step(self.model, self.data)

        # 6. Calculate reward (CRITICAL for RL learning!)
        reward, info = self._calculate_reward(bc_action_unscaled, rl_residual_unscaled, desired_q)

        # 7. Check if episode is done (e.g., time limit, failure)
        done = self._check_done()

        # 8. Get next observation for the RL agent
        next_obs = self._get_obs()

        # ... (render if necessary)

        return next_obs, reward, done, info
    def _get_tcp_pos(self):
        # Use the tcp_site_id initialized in __init__
        return self.data.site_xpos[self.tcp_site_id]
    # def _get_tcp_pos(self):
    #     tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "TCP_point")
    #     if tcp_site_id == -1:
    #         raise ValueError("TCP_point site not found in MuJoCo model. Check your XML.")
    #     return self.data.site_xpos[tcp_site_id]

    def _get_obs(self):
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()
        # string contacts
        contacts = contact.detect_bow_string_contact(self.model, self.data)
        contact_vec = [float(contacts[s]) for s in ('A','D','G','C')]
        
        raw_obs_dict = {
        'q_base': self.data.qpos[0],
        'q_shoulder': self.data.qpos[1],
        'q_elbow': self.data.qpos[2],
        'q_wrist1': self.data.qpos[3],
        'q_wrist2': self.data.qpos[4],
        'q_wrist3': self.data.qpos[5],
        'TCP_pose_x': self.data.site_xpos[self.tcp_site_id][0],
        'TCP_pose_y': self.data.site_xpos[self.tcp_site_id][1],
        'TCP_pose_z': self.data.site_xpos[self.tcp_site_id][2],
        'TCP_pose_rx': self.data.site_xmat[self.tcp_site_id][0], # Assuming you extract rotation correctly
        'TCP_pose_ry': self.data.site_xmat[self.tcp_site_id][1],
        'TCP_pose_rz': self.data.site_xmat[self.tcp_site_id][2],
        'time_elapsed_sec': self.data.time, # Or self.current_time if you prefer
        'remaining_duration_sec': self.total_duration - self.data.time, # Adjust if your duration logic is different
        'current_note_number': self._get_current_note_number(), # Ensure this returns a number
        'current_string': self._get_current_string(), # Ensure this returns a string like 'A', 'D', 'G', 'C'
        'event_label': self._get_current_event_label(), # Ensure this returns a string like 'START a_bow', 'END d_bow'
        'event_flag': self._get_current_event_flag() # Ensure this returns a number
        }

        print(f"raw_obs_dict from _get_obs(): {raw_obs_dict}") # <--- ADD THIS LINE

        # Ensure Preprocessor is imported or defined
        processed_obs = Preprocessor.preprocess_observation(
            raw_obs_dict, self.bc_scalers, self.bc_encoders, Preprocessor.INPUT_FEATURE_COLS
        )

        print(f"Shape of processed_obs from _get_obs(): {processed_obs.shape}") # <--- ADD THIS LINE

        return processed_obs
        #return np.concatenate([qpos, qvel, contact_vec])

    def _compute_reward(self):
        """
        Calculates the reward for the current step, incorporating residual penalty
        and existing task-specific rewards.

        Args:
            bc_action (np.ndarray): The joint positions predicted by the BC model (unscaled).
            rl_residual (np.ndarray): The raw residual action from the RL agent (unscaled).
            final_q_command (np.ndarray): The combined (BC + residual) target joint positions.
        """
        r = 0.0 # Initialize total reward
        info = {} # Dictionary to store debug/logging info

        # --- 1. Residual Penalty (NEW for Residual RL) ---
        # Penalize the magnitude of the RL agent's residual action.
        # This encourages the RL agent to make small, precise corrections,
        # relying primarily on the BC policy.
        residual_magnitude = np.linalg.norm(rl_residual)
        r -= self.residual_penalty * residual_magnitude
        info['residual_penalty_term'] = self.residual_penalty * residual_magnitude
        info['residual_magnitude'] = residual_magnitude


        # --- 2. Existing Bow String Position Error ---
        # (Adapted from your original _compute_reward)
        # Ensure self.note_sequence and current_idx are properly managed/updated in step() or reset()
        # You'll need to update self.current_idx in your step() method to track musical progression.
        # This might involve finding the current note based on `self.data.time`.
        if not self.note_sequence: # Handle case where note_sequence might be empty
            dist = 0.0
            info['bow_string_dist'] = 0.0
        else:
            # You might need a more robust way to get `self.current_idx` if it's not time-synced.
            # E.g., self.current_idx = self._get_current_note_idx_from_time()
            idx = min(self.current_idx, len(self.note_sequence) - 1)
            
            tcp = self._get_tcp_pos() # Ensure _get_tcp_pos() is defined and returns 3D numpy array
            raw_s = self.note_sequence[idx].get('string', '')

            if '-' in raw_s:
                target_str = raw_s.split('-')[1]
            else:
                target_str = raw_s

            fid, tid = self.string_sites.get(target_str, (None, None))
            if fid is None or tid is None:
                dist = 0.0
            else:
                p1 = self.data.site_xpos[fid]
                p2 = self.data.site_xpos[tid]
                line_vec = p2 - p1
                norm = np.linalg.norm(line_vec)
                if norm > 1e-6:
                    dist = np.linalg.norm(np.cross(tcp - p1, tcp - p2)) / norm
                else:
                    dist = 0.0
            info['bow_string_dist'] = dist
            r -= dist # Penalize distance from string

        # --- 3. Contact Penalty ---
        # (From your original _compute_reward)
        collision, _, _ = contact.detect_collision(self.model, self.data)
        if collision:
            r -= self.contact_penalty
            info['contact_penalty_applied'] = True
        else:
            info['contact_penalty_applied'] = False

        # --- 4. Torque Change Penalty ---
        # (From your original _compute_reward)
        # Note: self.data.ctrl contains the *applied* control signals, which should be
        # the result of your PID control acting on `final_q_command`.
        # Ensure self.prev_torque is initialized (e.g., to np.zeros(6)) in __init__
        delta_tau = self.data.ctrl[:6] - self.prev_torque
        torque_change_magnitude = np.linalg.norm(delta_tau)
        r -= self.torque_penalty * torque_change_magnitude
        self.prev_torque = self.data.ctrl[:6].copy()
        info['torque_penalty_term'] = self.torque_penalty * torque_change_magnitude
        info['torque_change_magnitude'] = torque_change_magnitude

        # --- 5. Bow-String Alignment Penalty ---
        # (From your original _compute_reward)
        if fid is not None and tid is not None:
            p1 = self.data.site_xpos[fid]
            p2 = self.data.site_xpos[tid]
            string_vec = p2 - p1
            norm_string_vec = np.linalg.norm(string_vec)
            if norm_string_vec > 1e-8:
                string_vec /= norm_string_vec
            else:
                string_vec[:] = 1.0  # fallback default unit vector

            tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_link')
            bow_vec = self.data.xmat[tcp_id].reshape((3, 3))[:, 0] # Assuming x-axis of TCP frame is bow direction
            norm_bow_vec = np.linalg.norm(bow_vec)
            if norm_bow_vec > 1e-8:
                 bow_vec /= norm_bow_vec
            else:
                 bow_vec[:] = 1.0 # fallback default unit vector

            dot = np.clip(np.dot(bow_vec, string_vec), -1.0, 1.0)
            angle_error = np.abs(np.pi / 2 - np.arccos(dot)) # Ideal angle is 90 degrees (pi/2 radians)
            r -= 0.5 * angle_error # Penalize deviation from 90 degrees
            info['alignment_angle_error'] = angle_error
        else:
            info['alignment_angle_error'] = 0.0 # No penalty if no target string

        return r, info
    def _check_done(self):
        # Example done conditions:
        # 1. Episode time limit reached:
        # if self.data.time >= self.episode_duration_limit: return True
        # 2. Robot out of bounds / crashed:
        # if self._is_robot_crashed(): return True
        # 3. All notes played:
        # if self.current_note_idx >= len(self.note_sequence): return True
        return False # Placeholder

    def render(self, mode: str = "human"):
        if mode != "human":
            return None      # Gym API requires this guard

        if self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model,
                                                            self.data)
            except Exception as e:
                print(f"Failed to launch MuJoCo viewer: {e}")
                self.viewer = None

        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# --- geminiRunv2.py ---
import time, sys, importlib
sys.modules['numpy._core.numeric'] = importlib.import_module('numpy.core.numeric')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_trajectory import UR5eCelloTrajectoryEnv
from parsemidi import parse_midi
import pandas as pd

def extract_joint_angles(csv_filename):
    df = pd.read_csv(csv_filename)
    cols = ['q_base','q_shoulder','q_elbow','q_wrist1','q_wrist2','q_wrist3']
    return df[cols].values.tolist()


# if __name__ == '__main__':
#     traj = extract_joint_angles('path/to/log.csv')
#     notes = parse_midi('path/to/file.mid')
#     scene = 'path/to/scene.xml'
#     start = traj[0]
#     def mk():
#         return UR5eCelloTrajectoryEnv(
#             model_path=scene,
#             trajectory=traj,
#             note_sequence=notes,
#             action_scale=0.05,
#             residual_penalty=0.02,
#             contact_penalty=0.1,
#             torque_penalty=0.001,
#             kp=100, kd=2, ki=0.1,
#             start_joint_positions=start
#         )
#     env = DummyVecEnv([mk])
#     model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./tb_residual/')
#     model.learn(total_timesteps=200000)
#     model.save('ppo_residual_ur5e')
