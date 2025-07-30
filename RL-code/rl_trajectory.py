import gym
from gym import spaces
import mujoco
import contact
#from parsemidi import parse_midi 
import pandas as pd
import numpy as np
import time
import sys
import os
import mujoco.viewer 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Baseline-Runners')))
#import robot_runner  # baseline controller
import torch.nn as nn 
import torch
import pickle

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder 
from scipy.spatial.transform import Rotation as R # idk about this


# robot joints need to align with xml description
MUJOCO_JOINT_NAME_MAP = {
    'q_base': 'shoulder_pan_joint',
    'q_shoulder': 'shoulder_lift_joint',
    'q_elbow': 'elbow_joint',
    'q_wrist1': 'wrist_1_joint',
    'q_wrist2': 'wrist_2_joint',
    'q_wrist3': 'wrist_3_joint',
}

# BC Architecture (must align with BC code!!!)
class BehavioralCloningModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(BehavioralCloningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Corrected from hidden_dim to hidden_size
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_dim) # Corrected from hidden_dim to hidden_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x) 
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class Preprocessor:
    INPUT_FEATURE_COLS = [
        'q_base', 'q_shoulder', 'q_elbow', 'q_wrist1', 'q_wrist2', 'q_wrist3',
        'TCP_pose_x', 'TCP_pose_y', 'TCP_pose_z', 'TCP_pose_rx', 'TCP_pose_ry', 'TCP_pose_rz',
        'time_elapsed_sec', 'remaining_duration_sec', 'current_note_number',
        'current_string', 'event_label', 'event_flag', 'current_bowing'
    ]
    TARGET_COLS = [
        'q_base', 'q_shoulder', 'q_elbow', 'q_wrist1', 'q_wrist2', 'q_wrist3'
    ]

    CATEGORICAL_COLS = ['current_string', 'event_label', 'current_bowing']
    
    HIDDEN_SIZE = 256 # Should match what was used for BC training

    @staticmethod
    # no
    def _infer_bow_direction(event_label):
        if 'a_bow' in str(event_label).lower() or 'up' in str(event_label).lower():
            return 'up'
        elif 'd_bow' in str(event_label).lower() or 'down' in str(event_label).lower():
            return 'down'
        else:
            return 'none'

    @staticmethod
    def _infer_is_transition(event_flag, event_label):
        if 'TRANSITION' in str(event_label).upper():
            return 1
        if 1 <= event_flag <= 8:
            return 0 # regular note
        return 1 

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
                # Ensure the column exists and cast to float32
                if col in single_obs_df.columns:
                    # FIX for MinMaxScaler warning: Pass DataFrame slice (matching fitting)
                    processed_features.append(loaded_scalers[col].transform(single_obs_df[[col]].astype(np.float32)))
                else:
                    # Fallback: append a zero array if a numerical column is missing
                    print(f"Warning: Numerical column '{col}' not found in raw_obs_dict. Appending zeros.")
                    processed_features.append(np.zeros((1, 1), dtype=np.float32))
            else:
                # If no scaler, just get values and cast to float32
                # This doesn't involve a transformer, so .values is appropriate for NumPy output
                processed_features.append(single_obs_df[[col]].values.astype(np.float32))

        # current_string one-hot encoding
        # FIX for OneHotEncoder warning: Pass NumPy array (matching fitting)
        current_string_encoded = loaded_encoders['current_string'].transform(single_obs_df[['current_string']].values)
        processed_features.append(current_string_encoded.astype(np.float32))

        # bow_direction inference and one-hot encoding
        # This part is already correct as it's fitted and transformed with NumPy arrays
        bow_dir = Preprocessor._infer_bow_direction(single_obs_df['event_label'].iloc[0])
        bow_direction_encoded = loaded_encoders['bow_direction'].transform(np.array([[bow_dir]]))
        processed_features.append(bow_direction_encoded.astype(np.float32))

        # is_transition inference
        is_transition_val = Preprocessor._infer_is_transition(single_obs_df['event_flag'].iloc[0], single_obs_df['event_label'].iloc[0])
        processed_features.append(np.array([[is_transition_val]], dtype=np.float32))

        # Stack horizontally, flatten, and ensure C-contiguous float32 array for PyTorch compatibility
        # FIX for ValueError: Flatten the result to (23,)
        return np.ascontiguousarray(np.hstack(processed_features)).flatten()

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
        trajectory: list, # unused?
        note_sequence: list,
        render_mode=None,
        action_scale: float = 0.10,
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
        on_line_bonus: float = 1.0,
        dist_penalty_coeff: float = 100.0,
        perpendicularity_penalty_coeff: float = 150.0
    ):
        super().__init__()
        # --- MuJoCo setup ---
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = self.model.opt.timestep

        # --- RL hyperparameters ---
        self.residual_penalty = residual_penalty
        self.contact_penalty = contact_penalty
        self.torque_penalty = torque_penalty

        # --- PID gains ---
        self.kp, self.kd, self.ki = kp, kd, ki
        self.total_pid_error = np.zeros(6)

        # --- Musical notes & string mapping ---
        self.note_sequence = note_sequence

        # Calculate cumulative durations for note tracking
        self.cumulative_note_durations = [0.0]
        cumulative_sum = 0.0
        for note in self.note_sequence:
            cumulative_sum += note.get('duration', 0.5) 
            self.cumulative_note_durations.append(cumulative_sum)
        self.total_midi_duration = cumulative_sum
        self.total_duration = self.total_midi_duration # Use actual midi duration

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
                'current_string': 'A', 'current_bowing': 'down', 'event_label': 'START a_bow', 'event_flag': 1
            }
            
            
            bc_input_dim = Preprocessor.preprocess_observation(
                dummy_raw_obs_dict, self.bc_scalers, self.bc_encoders, Preprocessor.INPUT_FEATURE_COLS
            ).shape[0]
            print(f"Calculated bc_input_dim for observation_space: {bc_input_dim}")
            bc_output_dim = len(Preprocessor.TARGET_COLS)

            self.bc_model = BehavioralCloningModel(bc_input_dim, bc_output_dim, bc_hidden_size)
            self.bc_model.load_state_dict(torch.load(bc_policy_path))
            self.bc_model.eval() # evaluation mode

            print(f"Successfully loaded BC model and preprocessors.")

        except Exception as e:
            print(f"Error loading BC model/preprocessors: {e}. Running without BC baseline.")
            self.bc_model = None # Ensure it's None if loading fails

        # --- Action Space for RL Agent (Residuals) ---
        # The RL agent will predict small deltas for each joint.
        # Max change per step for each joint. 
        # This will be in RADIANS for joint angles.
        self.action_space_range = action_scale # Renamed to be more explicit for residual
        self.action_space = spaces.Box(
            low=-self.action_space_range,
            high=self.action_space_range,
            shape=(len(Preprocessor.TARGET_COLS),), # 6 joints for UR5e
            dtype=np.float32,
        )

        # --- Observation Space for RL Agent (Should be same as BC input) ---
        self.observation_space = spaces.Box(
            low=-np.inf, # Scaled values can be negative
            high=np.inf, # Scaled values can be positive
            shape=(bc_input_dim,), # This is the preprocessed feature vector size
            dtype=np.float32,
        )

        # --- Tracking ---
        self.prev_torque = np.zeros(6)
        self.current_time = 0.0
        self.start_positions = start_joint_positions

        # Store physical string endpoints (from 'fromto' in XML)
        self.string_physical_endpoints = {}
        for s in ('A', 'D', 'G', 'C'):
            top_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_string_physical_top')
            bottom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_string_physical_bottom')
            if top_id == -1 or bottom_id == -1:
                print(f"Warning: Physical top/bottom sites for {s} string not found. Ensure they are in XML.")
                self.string_physical_endpoints[s] = (None, None)
            else:
                self.string_physical_endpoints[s] = (top_id, bottom_id)
        print(f'String physical endpoints: {self.string_physical_endpoints}')

        # # Store optimal bowing region sites (from your _bow_poses data)
        # self.bow_region_tip_sites = {}
        # self.bow_region_frog_sites = {}
        # for s in ('A', 'D', 'G', 'C'):
        #     tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_bow_tip_region')
        #     frog_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_bow_frog_region')
        #     if tip_id == -1: print(f"Warning: {s}_bow_tip_region site not found.")
        #     if frog_id == -1: print(f"Warning: {s}_bow_frog_region site not found.")
        #     self.bow_region_tip_sites[s] = tip_id
        #     self.bow_region_frog_sites[s] = frog_id
        # print(f'Bowing region tip sites: {self.bow_region_tip_sites}')
        # print(f'Bowing region frog sites: {self.bow_region_frog_sites}')

        # For the bow_hair geom for orientation and contact point
        self.bow_hair_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bow_hair")
        if self.bow_hair_geom_id == -1:
            raise ValueError("bow_hair geom not found in XML!")
        print(f'Bow hair geom ID: {self.bow_hair_geom_id}')

        # Get sensor IDs for direct pressure feedback
        self.string_touch_sensor_ids = {
            'A': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'A_string_touch'),
            'D': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'D_string_touch'),
            'G': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'G_string_touch'),
            'C': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'C_string_touch'),
        }
        for s, sensor_id in self.string_touch_sensor_ids.items():
            if sensor_id == -1:
                print(f"Warning: {s}_string_touch sensor not found.")
        print(f'String touch sensor IDs: {self.string_touch_sensor_ids}')

        # initialize tcp_site_id (still useful for general TCP position)
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "TCP_point")
        if self.tcp_site_id == -1:
            site_names = [self.model.site(i).name for i in range(self.model.nsite)]
            print("Available sites in model:", site_names)
            raise ValueError("TCP_point site not found in MuJoCo model. Please check your XML model.")
        print(f'TCP site ID: {self.tcp_site_id}')

        # Pre-calculate target bowing lines for each string
        self.target_bow_lines = self._calculate_target_bowing_lines()

        self.on_line_bonus = on_line_bonus
        self.dist_penalty_coeff = dist_penalty_coeff
        self.perpendicularity_penalty_coeff = perpendicularity_penalty_coeff

        """
        self.string_sites = {}
        for s in ('A','D','G','C'):
            frog = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_frog')
            tip  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_tip')
            self.string_sites[s] = (frog, tip)
        print(f'String sites: {self.string_sites}')

        # initialize tcp_site_id
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "TCP_point")
        if self.tcp_site_id == -1:
            site_names = [self.model.site(i).name for i in range(self.model.nsite)]
            print("Available sites in model:", site_names)
            raise ValueError("TCP_point site not found in MuJoCo model. Please check your XML model.")
        print(f'TCP site ID: {self.tcp_site_id}')
       
       # should work better with the new waypoints
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
        """
        self.render_mode = render_mode
        self.viewer = None

        self.reset()
    def _calculate_target_bowing_lines(self):
        """
        Calculates the ideal bowing line segment for each string,
        ensuring it's perpendicular to the physical string and passes
        through the midpoint of the measured optimal bowing region.
        """
        target_lines = {}
        # Define your _bow_poses data as a class attribute or pass it
        # For simplicity, I'll hardcode it here, but ideally load it from a file
        # or pass it during environment creation.
        _bow_poses_data = {
            'A': {
                "tip_p": np.array([.473129539189, .413197423330, .256308427905]),
                "frog_p": np.array([.300717266074, .793568239540, .099710283103])
            },
            'D': {
                "tip_p": np.array([.340413993945, .280157415162, .176342071758]),
                "frog_p": np.array([.302785064368, .749849181019, .117254426008])
            },
            'G': {
                "tip_p": np.array([.162016291992, .201320984957, .059414774157]),
                "frog_p": np.array([.281203376642, .681662588607, .104672526365])
            },
            'C': {
                "tip_p": np.array([.079815569355, .285182178102, -.086654726588]),
                "frog_p": np.array([.256662516098, .610082591416, .062624387196])
            }
        }

        for s in ('A', 'D', 'G', 'C'):
            top_id, bottom_id = self.string_physical_endpoints[s]
            if top_id is None or bottom_id is None:
                print(f"Skipping target line calculation for {s} due to missing physical string endpoints.")
                target_lines[s] = (None, None, None) # start, end, normalized_direction
                continue

            # 1. Get physical string line (simulation_string_line)
            p_top = self.model.site(top_id).pos # Use model.site().pos for static XML positions
            p_bottom = self.model.site(bottom_id).pos
            
            simulation_string_vec = p_top - p_bottom
            norm_simulation_string_vec = np.linalg.norm(simulation_string_vec)
            if norm_simulation_string_vec < 1e-6:
                print(f"Warning: Physical string {s} has zero length. Cannot calculate perpendicular line.")
                target_lines[s] = (None, None, None)
                continue
            normalized_simulation_string_vec = simulation_string_vec / norm_simulation_string_vec

            # 2. Get initial measured bow stroke points
            bow_measured_tip_pos = _bow_poses_data[s]["tip_p"]
            bow_measured_frog_pos = _bow_poses_data[s]["frog_p"]

            # Calculate midpoint and initial direction of measured bow stroke
            bow_measured_midpoint = (bow_measured_tip_pos + bow_measured_frog_pos) / 2.0
            bow_measured_direction_vec = bow_measured_tip_pos - bow_measured_frog_pos
            bow_measured_length = np.linalg.norm(bow_measured_direction_vec)

            if bow_measured_length < 1e-6:
                print(f"Warning: Measured bow stroke for {s} has zero length. Cannot calculate perpendicular line.")
                target_lines[s] = (None, None, None)
                continue

            # 3. Adjust to be perpendicular to simulation_string_line
            # Project bow_measured_direction_vec onto the plane perpendicular to normalized_simulation_string_vec
            # Component parallel to string:
            parallel_component = np.dot(bow_measured_direction_vec, normalized_simulation_string_vec) * normalized_simulation_string_vec
            
            # Component perpendicular to string:
            perpendicular_component = bow_measured_direction_vec - parallel_component
            
            norm_perpendicular_component = np.linalg.norm(perpendicular_component)

            if norm_perpendicular_component < 1e-6:
                # This means the measured bow stroke was already perfectly parallel to the string.
                # In this rare case, we need to pick an arbitrary perpendicular direction.
                if np.linalg.norm(np.cross(normalized_simulation_string_vec, np.array([0,0,1]))) > 1e-6:
                    target_bow_stroke_direction_normalized = np.cross(normalized_simulation_string_vec, np.array([0,0,1]))
                elif np.linalg.norm(np.cross(normalized_simulation_string_vec, np.array([0,1,0]))) > 1e-6:
                    target_bow_stroke_direction_normalized = np.cross(normalized_simulation_string_vec, np.array([0,1,0]))
                else: # Fallback, highly unlikely
                    target_bow_stroke_direction_normalized = np.cross(normalized_simulation_string_vec, np.array([1,0,0]))
                target_bow_stroke_direction_normalized /= np.linalg.norm(target_bow_stroke_direction_normalized)
            else:
                target_bow_stroke_direction_normalized = perpendicular_component / norm_perpendicular_component

            # Scale to original measured length
            target_bow_stroke_vector = target_bow_stroke_direction_normalized * bow_measured_length

            # 4. Define target line for the given string
            target_bow_stroke_start = bow_measured_midpoint - target_bow_stroke_vector / 2.0
            target_bow_stroke_end = bow_measured_midpoint + target_bow_stroke_vector / 2.0

            target_lines[s] = (target_bow_stroke_start, target_bow_stroke_end, target_bow_stroke_direction_normalized)
            print(f"Calculated target bowing line for {s}: Start={target_bow_stroke_start}, End={target_bow_stroke_end}, Dir={target_bow_stroke_direction_normalized}")

        return target_lines
    
    def reset(self):
        # reset qpos/qvel
        if self.start_positions is not None:
            self.data.qpos[:6] = np.array(self.start_positions)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # reset trackers
        self.current_time = 0.0
        self.current_midi_note_idx = 0 
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
            joint_name_from_preprocessor = Preprocessor.TARGET_COLS[i]
            joint_id = self.model.joint(MUJOCO_JOINT_NAME_MAP.get(joint_name_from_preprocessor)).id # Adjust name mapping
            low_limit = self.model.jnt_range[joint_id, 0]
            high_limit = self.model.jnt_range[joint_id, 1]
            desired_q[i] = np.clip(desired_q[i], low_limit, high_limit)

        # 4. Apply combined action using your PID controller
        self._apply_pid_control(desired_q) # This method should exist in your env

        # 5. Simulate for one step
        mujoco.mj_step(self.model, self.data)

        # 6. Calculate reward (CRITICAL for RL learning!)
        reward, info = self._compute_reward(bc_action_unscaled, rl_residual_unscaled, desired_q) # Updated call

        # 7. Check if episode is done (e.g., time limit, failure)
        done = self._check_done()

        # 8. Get next observation for the RL agent
        next_obs = self._get_obs()

        return next_obs, reward, done, info
        
    def _get_current_note_info(self):
        """
        Determines the current active note from the note_sequence based on simulation time.
        Updates self.current_midi_note_idx.
        """
        # If we are past the end of the last note, return sentinel info
        if self.data.time >= self.total_midi_duration and len(self.note_sequence) > 0:
            # Ensure index points to the last note if we're exactly at the end, otherwise signal end.
            if self.data.time >= self.cumulative_note_durations[-1]: 
                 return {
                    'note_number': 0, # Placeholder for no active note
                    'string': 'None',
                    'bowing': 'none', # No bowing if no active note
                    'event_label': 'END_SEQUENCE',
                    'event_flag': 0,
                    'duration_sec': 0.0
                }

        # Advance current_midi_note_idx
        # Use a while loop to correctly jump over multiple short notes if simulation step is large
        while (self.current_midi_note_idx < len(self.note_sequence) - 1 and # Ensure not trying to go past last note
               self.data.time >= self.cumulative_note_durations[self.current_midi_note_idx + 1]):
            self.current_midi_note_idx += 1
        if self.current_midi_note_idx < len(self.note_sequence):
            current_note = self.note_sequence[self.current_midi_note_idx]
            # Construct dictionary for observation, using 'number' for note_number
            # and 'duration' for duration_sec, as per parse_midi output.
            note_info_for_obs = {
                'note_number': current_note.get('number', 0), # Corrected key from 'note' to 'number' as per parse_midi
                'string': current_note.get('string', 'None'), # 'string' is directly provided by parse_midi
                'bowing': current_note.get('bowing', 'none'), # Use 'bowing' key if available, else default to 'none'
                'duration_sec': current_note.get('duration', 0.5), # Use 'duration' key directly from parse_midi
                'event_label': current_note.get('event_label', ''), # Try to get 'event_label' if it exists
                'event_flag': current_note.get('event_flag', 0) # Try to get 'event_flag' if it exists
            }
            
            return note_info_for_obs
        else:
            # This case should ideally be caught by the first `if` block, but as a safeguard:
            return {
                'note_number': 0,
                'string': 'None',
                'bowing': 'none', # No bowing if no active note
                'event_label': 'END_SEQUENCE',
                'event_flag': 0,
                'duration_sec': 0.0
            }

    def _get_current_note_number(self):
        """Returns the MIDI note number of the current active note."""
        note_info = self._get_current_note_info()
        return note_info['note_number']

    def _get_current_string(self):
        """
        Returns the single-character string ('A', 'D', 'G', 'C') corresponding
        to the current active note. Handles 'X-string' format and note-to-string mapping.
        """
        note_info = self._get_current_note_info()
        raw_s = note_info['string']
        if '-' in raw_s:
            return raw_s.split('-')[1]
        elif raw_s == 'None': # Handle end of sequence or unmapped notes
            return 'None'
        return raw_s # Should be 'A', 'D', 'G', 'C' directly if no hyphen
    def _get_current_bowing(self):
        """
        Returns the bowing direction based on the current event label.
        Uses the _infer_bow_direction method to determine the bowing direction.
        """
        note_info = self._get_current_note_info()
        raw_b = note_info['bowing']
        if raw_b is None or raw_b == '':
            return 'none'
        return raw_b.lower()
    def _get_current_event_label(self):
        """
        Synthesizes an event label (e.g., 'START a_bow', 'END_SEQUENCE')
        based on the current note and simulation time.
        Prioritizes existing 'event_label' if present in note_info.
        """
        note_info = self._get_current_note_info()
        
        # If the note_info already has an event_label, use it (from original log or augmented MIDI)
        if 'event_label' in note_info and note_info['event_label'] not in ['', None]:
            return note_info['event_label']

        if note_info['note_number'] == 0 and note_info['string'] == 'None': 
            return 'END_SEQUENCE'
        
        current_string = self._get_current_string()
        if current_string in ['A', 'D', 'G', 'C']:
            # Simplification: assuming it's always a "START" event when a note is active.
            # If the BC model needs 'END' events, more complex logic is required
            # to detect the end of a note's duration.
            return f"START {current_string.lower()}_bow"
        
        return "TRANSITION" # Default if no specific note is active or recognized

    def _get_current_event_flag(self):
        """
        Synthesizes an event flag (e.g., 1 for START, 0 for END_SEQUENCE)
        based on the current note.
        Prioritizes existing 'event_flag' if present in note_info.
        """
        note_info = self._get_current_note_info()

        # If the note_info already has an event_flag, use it
        if 'event_flag' in note_info and note_info['event_flag'] != 0:
            return note_info['event_flag']

        if note_info['note_number'] == 0 and note_info['string'] == 'None': 
            return 0 # Or a specific flag for end of sequence
        
        # For a "START" event, a common flag might be 1 or 3 from typical log data.
        # Let's use 1 as a generic start flag.
        return 1

    def _get_tcp_pos(self):
        # Use the tcp_site_id initialized in __init__
        return self.data.site_xpos[self.tcp_site_id]
    def _get_tcp_xmat(self):
        """Returns the current rotation matrix of the TCP site."""
        return self.data.site_xmat[self.tcp_site_id].reshape(3,3)

    def _get_obs(self):
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()
        # string contacts
        contacts = contact.detect_bow_string_contact(self.model, self.data)
        contact_vec = [float(contacts[s]) for s in ('A','D','G','C')]
        
        # --- Rotation Conversion ---
        # Get the 3x3 rotation matrix for the TCP site
        tcp_xmat = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        # Convert the rotation matrix to Euler angles (radians)
        # Assuming 'xyz' convention matches your BC training data
        # If your data used a different convention (e.g., 'zyx'), change 'xyz' accordingly.
        # If your BC training data used degrees, change degrees=False to degrees=True.
        r = R.from_matrix(tcp_xmat)
        tcp_euler_angles = r.as_euler('xyz', degrees=False) 

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
            'TCP_pose_rx': tcp_euler_angles[0], # Use converted Euler angles
            'TCP_pose_ry': tcp_euler_angles[1], # Use converted Euler angles
            'TCP_pose_rz': tcp_euler_angles[2], # Use converted Euler angles
            'time_elapsed_sec': self.data.time, # Or self.current_time if you prefer
            'remaining_duration_sec': self.total_duration - self.data.time, # Adjust if your duration logic is different
            'current_note_number': self._get_current_note_number(), 
            'current_string': self._get_current_string(), 
            'current_bowing': self._get_current_bowing(),
            'event_label': self._get_current_event_label(), 
            'event_flag': self._get_current_event_flag() 
        }

        #print(f"raw_obs_dict from _get_obs(): {raw_obs_dict}") 

        # Ensure Preprocessor is imported or defined
        processed_obs = Preprocessor.preprocess_observation(
            raw_obs_dict, self.bc_scalers, self.bc_encoders, Preprocessor.INPUT_FEATURE_COLS
        )

        #print(f"Shape of processed_obs from _get_obs(): {processed_obs.shape}") 

        return processed_obs

    def _apply_pid_control(self, desired_q):
        """Applies PID control to achieve desired joint positions."""
        current_q = self.data.qpos[:6]
        current_q_vel = self.data.qvel[:6]

        error = desired_q - current_q
        self.total_pid_error += error * self.sim_dt # Integrate error

        # Proportional, Integral, Derivative components
        p_term = self.kp * error
        i_term = self.ki * self.total_pid_error
        d_term = -self.kd * current_q_vel # D-term acts on current velocity to dampen

        # Combine terms to get desired torques
        # Ensure that the torques are within the actuator's force range.
        # Check model.actuator_forcemax or similar for limits if needed.
        torques = p_term + i_term + d_term

        # Apply torques to actuators
        self.data.ctrl[:6] = torques

    def _compute_reward(self, bc_action, rl_residual, final_q_command):
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

        # --- Get current note info and TCP data ---
        idx = self.current_midi_note_idx
        string_crossing = False
        if not self.note_sequence or idx >= len(self.note_sequence): # Handle end of sequence
            target_str_key = 'None'
            # current_note_info = {'note_number': 0, 'string': 'None'} # Not strictly needed if bowing/pressure removed
        else:
            current_note_info = self.note_sequence[idx]
            raw_s = current_note_info.get('string', '')
            if '-' in raw_s:
                string_crossing = True
            target_str_key = raw_s.split('-')[1] if '-' in raw_s else raw_s # e.g., 'A', 'D', 'G', 'C'

        # Get TCP position and orientation
        tcp_pos = self._get_tcp_pos()
        tcp_xmat = self._get_tcp_xmat()

        # Bow string position error
         # --- Positional Alignment & Perpendicularity ---
        if not string_crossing:
            region_start_point, region_end_point, normalized_string_vec = self.target_bow_line_segments.get(target_str_key, (None, None, None))

            if region_start_point is not None and region_end_point is not None:
                # Calculate distance from TCP_pos to the TARGET BOWING REGION LINE SEGMENT
                line_segment_vec = region_end_point - region_start_point
                line_segment_length_sq = np.dot(line_segment_vec, line_segment_vec)

                dist_to_bowing_region = 0.0
                if line_segment_length_sq > 1e-9:
                    t = np.dot(tcp_pos - region_start_point, line_segment_vec) / line_segment_length_sq
                    t = np.clip(t, 0.0, 1.0)
                    closest_point_on_segment = region_start_point + t * line_segment_vec
                    dist_to_bowing_region = np.linalg.norm(tcp_pos - closest_point_on_segment)
                else:
                    dist_to_bowing_region = np.linalg.norm(tcp_pos - region_start_point)

                # Bonus for being on line, penalty for being off
                reward_pos = self.on_line_bonus - self.dist_penalty_coeff * dist_to_bowing_region
                r += reward_pos
                info['dist_to_bowing_region'] = dist_to_bowing_region
                info['reward_positional'] = reward_pos
            else: # If no target line defined for the current string
                info['dist_to_bowing_region'] = 0.0
                info['reward_positional'] = 0.0
                info['bow_string_perpendicularity_penalty'] = 0.0
                info['dot_product_bow_string_perp'] = 0.0
        else:
            # want to avoid colliding with strings during string crossing
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

        

        return r, info
    def _check_done(self):
        # Example done conditions:
        # 1. Episode time limit reached:
        if self.data.time >= self.total_duration: return True
        # 2. Robot out of bounds / crashed:
        # if self._is_robot_crashed(): return True
        # 3. All notes played:
        if self.current_midi_note_idx >= len(self.note_sequence): return True
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