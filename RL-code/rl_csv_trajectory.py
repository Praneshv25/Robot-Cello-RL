import gym
from gym import spaces
import numpy as np
import mujoco
import mujoco.viewer
import pandas as pd
import time
import os
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R 


# Assuming parsemidi.py is available
from parsemidi import parse_midi

class UR5eCelloTrajectoryEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30} 

    def __init__(
        self,
        model_path: str,
        baseline_csv_path: str, 
        note_sequence: list,      
        render_mode=None,
        action_scale: float = 0.05,
        residual_penalty: float = 0.01,
        contact_penalty: float = 0.1,
        torque_penalty: float = 0.001,
        perpendicularity_penalty: float = 0.0,
        timing_penalty: float = 0.0,
        kp: float = 100.0,
        kd: float = 2.0,
        ki: float = 0.1,
        start_joint_positions=None,
    ):
        super().__init__()
        # --- MuJoCo setup ---
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = self.model.opt.timestep
        self.viewer = None
        self.render_mode = render_mode

        self.kp, self.kd, self.ki = kp, kd, ki
        self.residual_penalty = residual_penalty
        self.contact_penalty = contact_penalty
        self.torque_penalty = torque_penalty
        self.perpendicularity_penalty = perpendicularity_penalty
        self.timing_penalty = timing_penalty

        # --- Baseline Trajectory Setup ---
        self.baseline_df = pd.read_csv(baseline_csv_path)
        self.trajectory_times = self.baseline_df['time_elapsed_sec'].values
        # instead of detailed logs 
        # --- Musical Note Sequence Setup ---
        # Ensure note times are in seconds and calculate duration_sec
        try:
            self.note_sequence = sorted([
                {**note,
                 'start_time_sec': note['start_time'] / 1000.0,
                 'end_time_sec': note['end_time'] / 1000.0,
                 'duration_sec': (note['end_time'] - note['start_time']) / 1000.0 # Re-calculate for robustness
                }
                for note in note_sequence
            ], key=lambda x: x['start_time_sec'])
        except KeyError as e:
            raise KeyError(f"Each note in 'note_sequence' must contain '{e}' keys. Please check your MIDI parsing output.")

        self.current_note_idx = 0
        self.num_notes = len(self.note_sequence)

        # --- NEW: Pre-process baseline_df to get start/end states for each note ---
        self.note_targets = []
        joint_cols = ['q_base', 'q_shoulder', 'q_elbow', 'q_wrist1', 'q_wrist2', 'q_wrist3']
        tcp_pose_cols = ['TCP_pose_x', 'TCP_pose_y', 'TCP_pose_z', 'TCP_pose_rx', 'TCP_pose_ry', 'TCP_pose_rz']
        
        # Create interpolation functions for the full baseline, but only to extract note start/end states accurately
        # These will be temporary or used for fetching initial/final states of notes.
        full_baseline_interp_fns = {}
        all_numeric_cols = joint_cols + tcp_pose_cols + [f'qd_{joint}' for joint in joint_cols]
        # Add remaining_duration_sec to all_numeric_cols if it exists and is used in observation
        if 'remaining_duration_sec' in self.baseline_df.columns:
            all_numeric_cols.append('remaining_duration_sec')


        for col in all_numeric_cols:
            if col in self.baseline_df.columns:
                full_baseline_interp_fns[col] = interp1d(
                    self.trajectory_times, self.baseline_df[col].values,
                    kind='linear', fill_value='extrapolate', bounds_error=False
                )
            else:
                print(f"Warning: Column '{col}' not found in baseline CSV for full interpolation setup.")

        for i, note in enumerate(self.note_sequence):
            note_start_time = note['start_time_sec']
            note_end_time = note['end_time_sec']

            # Use full_baseline_interp_fns to get precise start/end states for each note
            # This handles cases where note start/end times don't exactly match CSV timestamps
            start_q = np.array([full_baseline_interp_fns[f'q_{j}'](note_start_time) for j in joint_cols])
            end_q = np.array([full_baseline_interp_fns[f'q_{j}'](note_end_time) for j in joint_cols])
            
            start_qd = np.array([full_baseline_interp_fns[f'qd_{j}'](note_start_time) for j in joint_cols])
            end_qd = np.array([full_baseline_interp_fns[f'qd_{j}'](note_end_time) for j in joint_cols])

            start_tcp = np.array([full_baseline_interp_fns[f'TCP_pose_{dim}'](note_start_time) for dim in ['x', 'y', 'z', 'rx', 'ry', 'rz']])
            end_tcp = np.array([full_baseline_interp_fns[f'TCP_pose_{dim}'](note_end_time) for dim in ['x', 'y', 'z', 'rx', 'ry', 'rz']])
            
            # Identify if it's a string crossing (transition) note
            # Assuming 'is_transition' flag is added by your parsemidi or manually
            is_string_crossing = note.get('is_transition', False) 

            self.note_targets.append({
                'start_time_sec': note_start_time,
                'end_time_sec': note_end_time,
                'duration_sec': note['duration_sec'],
                'start_q': start_q,
                'end_q': end_q,
                'start_qd': start_qd,
                'end_qd': end_qd,
                'start_tcp': start_tcp,
                'end_tcp': end_tcp,
                'is_string_crossing': is_string_crossing,
                'original_note_data': note # Keep original note info for musical context in obs
            })


        # --- Initial MuJoCo State (using first note's start) ---
        if self.note_targets:
            initial_baseline_q = self.note_targets[0]['start_q']
            self.data.qpos[:6] = initial_baseline_q
        elif start_joint_positions is not None:
            start_joint_positions_np = np.array(start_joint_positions).flatten()
            if start_joint_positions_np.shape[0] != 6:
                raise ValueError(f"start_joint_positions must have 6 elements, but got {start_joint_positions_np.shape[0]}")
            self.data.qpos[:6] = start_joint_positions_np
        else:
            print("Warning: No note targets or start_joint_positions provided. Defaulting initial qpos to zeros.")
            self.data.qpos[:6] = np.zeros(6)


        # --- Action Space ---
        self.action_scale = action_scale
        self.action_space = spaces.Box(
            low=np.array([-self.action_scale] * 6),
            high=np.array([self.action_scale] * 6),
            dtype=np.float32
        )

        # --- Observation Space ---
        # Adjust observation space size if needed based on new features
        obs_dim = (
            6 + 6 + 6 + 6 + 1 + # Robot State (q, qdot, tcp_pose, tcp_lin_vel, tcp_ang_vel, contact_force)
            6 + 6 + 6 + # Baseline Reference (q_target, qdot_target, tcp_pose_target) - now note-based
            6 + 6 + # Error from Baseline (q_error, tcp_error)
            1 + 1 + 4 + 2 + 1 + 4 + 2 + # Musical Context (progress, time_rem, current_string, bow_dir, is_trans, next_string, next_bow_dir)
            1 # Add progress within string crossing (placeholder for now, will be 0-1)
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.last_q = self.data.qpos[:6].copy()
        self.q_error_integral = np.zeros(6) # Reset integral error for PID
        
        # State for string crossing
        self._string_crossing_progress = 0.0 # 0.0 to 1.0 within a string crossing
        self._string_crossing_stage = 0 # To track phases (e.g., lift, cross, descend)


    def _get_obs(self):
        sim_time = self.data.time # Current simulation time

        # Determine current note and progress
        current_note_data = None
        current_note_segment = None
        progress_in_current_note = 0.0
        time_remaining_in_note = 0.0
        
        # Advance current_note_idx if simulation time has passed the end of the current note
        while self.current_note_idx < self.num_notes and \
              sim_time >= self.note_targets[self.current_note_idx]['end_time_sec']:
            self.current_note_idx += 1

        if self.current_note_idx < self.num_notes:
            current_note_data = self.note_sequence[self.current_note_idx]
            current_note_segment = self.note_targets[self.current_note_idx]
            
            note_start_time = current_note_segment['start_time_sec']
            note_end_time = current_note_segment['end_time_sec']
            note_duration = current_note_segment['duration_sec']

            # Calculate progress within the current note
            if note_duration > 1e-6: # Avoid division by zero
                progress_in_current_note = (sim_time - note_start_time) / note_duration
                progress_in_current_note = np.clip(progress_in_current_note, 0.0, 1.0) # Clip to [0,1]
            else:
                progress_in_current_note = 1.0 # Instantaneous note

            time_remaining_in_note = note_end_time - sim_time
        else:
            # If all notes are finished, set to last note's info or default to zeros
            if self.num_notes > 0:
                current_note_data = self.note_sequence[self.num_notes - 1]
                current_note_segment = self.note_targets[self.num_notes - 1]
                progress_in_current_note = 1.0 # Already past the last note
                time_remaining_in_note = 0.0
            else:
                # Fallback if no notes at all
                current_note_data = {'start_time_sec': sim_time, 'end_time_sec': sim_time + 1.0, 'duration_sec': 1.0, 'string': '', 'note': '', 'bowing': '', 'is_transition': False}
                current_note_segment = {
                    'start_q': np.zeros(6), 'end_q': np.zeros(6),
                    'start_qd': np.zeros(6), 'end_qd': np.zeros(6),
                    'start_tcp': np.zeros(6), 'end_tcp': np.zeros(6),
                    'is_string_crossing': False
                }
                progress_in_current_note = 0.0
                time_remaining_in_note = 1.0

        # --- Determine Baseline Reference based on Note Type ---
        baseline_q = np.zeros(6)
        baseline_qdot = np.zeros(6)
        baseline_tcp_pose = np.zeros(6)
        string_crossing_progress_obs = 0.0 # Will be non-zero during string crossings

        if current_note_segment and not current_note_segment['is_string_crossing']:
            # Linear interpolation for regular notes
            start_q = current_note_segment['start_q']
            end_q = current_note_segment['end_q']
            start_qd = current_note_segment['start_qd']
            end_qd = current_note_segment['end_qd']
            start_tcp = current_note_segment['start_tcp']
            end_tcp = current_note_segment['end_tcp']

            baseline_q = start_q + (end_q - start_q) * progress_in_current_note
            baseline_qdot = start_qd + (end_qd - start_qd) * progress_in_current_note
            baseline_tcp_pose = start_tcp + (end_tcp - start_tcp) * progress_in_current_note
            string_crossing_progress_obs = 0.0

        elif current_note_segment and current_note_segment['is_string_crossing']:
            # --- Handle String Crossing (Non-Linear) ---
            # This is where the logic from the URScript's string_crossing method needs to be implemented.
            # This will NOT be a simple linear interpolation based on progress_in_current_note.
            # Instead, you'll calculate specific target waypoints/poses based on the current state and note context.
            
            # PLACEHOLDER: For now, it will default to the end_tcp of the segment if it's a string crossing.
            # You need to replace this with your detailed string crossing logic.
            # Example: Define stages (lift, cross, descend) and target poses for each stage.
            # For demonstration, we'll make it jump to the end pose, but this should be dynamic.
            baseline_q = current_note_segment['end_q'] # Or a calculated intermediate Q
            baseline_qdot = current_note_segment['end_qd'] # Or a calculated intermediate QD
            baseline_tcp_pose = current_note_segment['end_tcp'] # Or a calculated intermediate TCP pose

            # This _string_crossing_progress needs to be updated in the step function,
            # perhaps based on reaching certain sub-goals or specific time within the crossing.
            string_crossing_progress_obs = self._string_crossing_progress
            
            # --- IMPORTANT: IMPLEMENT YOUR CUSTOM STRING CROSSING LOGIC HERE ---
            # You will need to calculate baseline_q, baseline_qdot, baseline_tcp_pose
            # based on the start_bow_poses, end_bow_poses, next_dir concept from your URScript.
            # This is complex and might involve:
            # 1. Defining `start_bow_poses` and `end_bow_poses` for the string crossing event.
            # 2. Calculating `step1`, `target_pose` as described in your URScript.
            # 3. Determining which target pose (e.g., `step1` or `target_pose`) is the current "baseline" target
            #    based on `_string_crossing_stage` or `_string_crossing_progress`.
            # For now, it's a very simplistic placeholder.
            pass


        # Musical Context Features (from original note_data)
        current_string_one_hot = np.zeros(4) # A, D, G, C
        bow_direction_one_hot = np.zeros(2) # Up, Down
        is_transition = 0.0
        next_string_one_hot = np.zeros(4)
        next_bow_direction_one_hot = np.zeros(2)

        if current_note_data:
            string_map = {'A': 0, 'D': 1, 'G': 2, 'C': 3}
            if current_note_data['string'] in string_map:
                current_string_one_hot[string_map[current_note_data['string']]] = 1.0

            is_transition = 1.0 if current_note_data.get('is_transition', False) else 0.0
            
            if current_note_data.get('bowing') == 'up':
                bow_direction_one_hot[0] = 1.0
            elif current_note_data.get('bowing') == 'down':
                bow_direction_one_hot[1] = 1.0

            # Get next note data for context
            if (self.current_note_idx + 1) < self.num_notes:
                next_note_data = self.note_sequence[self.current_note_idx + 1]
                if next_note_data['string'] in string_map:
                    next_string_one_hot[string_map[next_note_data['string']]] = 1.0
                if next_note_data.get('bowing') == 'up':
                    next_bow_direction_one_hot[0] = 1.0
                elif next_note_data.get('bowing') == 'down':
                    next_bow_direction_one_hot[1] = 1.0

        # Robot State (from MuJoCo)
        actual_q = self.data.qpos[:6]
        actual_q_vel = self.data.qvel[:6]

        actual_tcp_pose_6d = np.zeros(6)
        actual_tcp_lin_vel = np.zeros(3)
        actual_tcp_ang_vel = np.zeros(3)
        contact_force_magnitudes = np.zeros(1)

        try:
            tcp_site_id = self.model.site_name2id('tool_tip')
            actual_tcp_pos = self.data.site_xpos[tcp_site_id]
            actual_tcp_rot_mat = self.data.site_xmat[tcp_site_id].reshape(3,3)
            # Convert actual_tcp_rot_mat to axis-angle representation (rx, ry, rz)
            r = R.from_matrix(actual_tcp_rot_mat)
            actual_tcp_pose_6d[:3] = actual_tcp_pos # x, y, z
            actual_tcp_pose_6d[3:] = r.as_rotvec() # axis-angle (rx, ry, rz)

            tcp_body_id = self.model.body_name2id('tool0')
            actual_tcp_lin_vel = self.data.body_cvel[tcp_body_id, :3]
            actual_tcp_ang_vel = self.data.body_cvel[tcp_body_id, 3:]

            # Contact force (if contact module is implemented and relevant)
            # if 'contact' in sys.modules and hasattr(contact, 'get_cello_bow_string_contact_force'):
            #     contact_info = contact.get_cello_bow_string_contact_force(self.model, self.data)
            #     if contact_info is not None:
            #         contact_force_magnitudes[0] = contact_info['force_magnitude']

        except Exception as e:
            # print(f"Warning: TCP pose/velocity calculation error: {e}. Defaulting to zeros.")
            pass # Keep zeros if calculation fails


        # Error from Baseline
        q_error = actual_q - baseline_q
        tcp_error = actual_tcp_pose_6d - baseline_tcp_pose

        # Concatenate all observation components
        obs = np.concatenate([
            actual_q,
            actual_q_vel,
            actual_tcp_pose_6d,
            actual_tcp_lin_vel,
            actual_tcp_ang_vel,
            contact_force_magnitudes,
            baseline_q,
            baseline_qdot,
            baseline_tcp_pose,
            q_error,
            tcp_error,
            np.array([progress_in_current_note]),
            np.array([time_remaining_in_note]),
            current_string_one_hot,
            bow_direction_one_hot,
            np.array([is_transition]),
            next_string_one_hot,
            next_bow_direction_one_hot,
            np.array([string_crossing_progress_obs]) # New observation feature
        ]).astype(np.float32)

        # --- Debugging NaN/Inf in Observation ---
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"CRITICAL ERROR: NaN/Inf detected in observation at time: {self.data.time}")
            print(f"Current Note Index: {self.current_note_idx}")
            print(f"Progress in Current Note: {progress_in_current_note}")
            print(f"Actual Q: {actual_q}, NaN: {np.any(np.isnan(actual_q))}")
            print(f"Baseline Q: {baseline_q}, NaN: {np.any(np.isnan(baseline_q))}")
            # ... add more specific prints for components that might be NaN/Inf
            raise ValueError("Observation contains NaN or Inf values!")
        # --- END Debugging ---

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.current_note_idx = 0
        self._string_crossing_progress = 0.0
        self._string_crossing_stage = 0

        if self.note_targets:
            initial_baseline_q = self.note_targets[0]['start_q']
            self.data.qpos[:6] = initial_baseline_q
        else:
            self.data.qpos[:6] = np.zeros(6) # Fallback if no notes

        self.data.qvel[:6] = np.zeros(6) # Start with zero velocity
        self.q_error_integral = np.zeros(6) # Reset integral error for PID
        
        mujoco.mj_forward(self.model, self.data) # Update kinematics

        observation = self._get_obs()
        info = {}
        return observation, info

    def _apply_pid_control(self, target_q):
        actual_q = self.data.qpos[:6]
        actual_q_vel = self.data.qvel[:6]

        q_error = target_q - actual_q
        self.q_error_integral += q_error * self.sim_dt
        
        # Calculate proportional, integral, and derivative terms
        p_term = self.kp * q_error
        i_term = self.ki * self.q_error_integral
        d_term = -self.kd * actual_q_vel # D-term acts against current velocity

        joint_torques = p_term + i_term + d_term
        self.data.ctrl[:6] = joint_torques # Apply torques to MuJoCo actuators
        return joint_torques # Return for potential debugging/logging


    def step(self, action):
        # Apply the residual action to the baseline control
        target_q_baseline = np.zeros(6) # Placeholder, will be filled based on current note
        
        # Recalculate accurate baseline target for current sim_time and note
        sim_time = self.data.time
        
        # Determine which note we are currently in for controlling the robot
        current_note_segment = None
        temp_current_note_idx = self.current_note_idx # Use a temporary index to find current segment
        
        while temp_current_note_idx < self.num_notes and \
              sim_time >= self.note_targets[temp_current_note_idx]['end_time_sec']:
            temp_current_note_idx += 1
        
        # Update self.current_note_idx for the environment's state
        self.current_note_idx = temp_current_note_idx

        progress_in_current_note = 0.0 # Re-calculate accurate progress for control/reward here

        if self.current_note_idx < self.num_notes:
            current_note_segment = self.note_targets[self.current_note_idx]
            note_start_time = current_note_segment['start_time_sec']
            note_duration = current_note_segment['duration_sec']
            if note_duration > 1e-6:
                progress_in_current_note = (sim_time - note_start_time) / note_duration
                progress_in_current_note = np.clip(progress_in_current_note, 0.0, 1.0)
            else:
                progress_in_current_note = 1.0 # Instantaneous note

            if not current_note_segment['is_string_crossing']:
                start_q = current_note_segment['start_q']
                end_q = current_note_segment['end_q']
                target_q_baseline = start_q + (end_q - start_q) * progress_in_current_note
                self._string_crossing_progress = 0.0 # Reset when not in crossing
                self._string_crossing_stage = 0
            else:
                # --- Handle String Crossing CONTROL Baseline ---
                # This needs to align with the string crossing targets defined in _get_obs
                # For control, the 'baseline' is the target position for the PID controller.
                # You'll use your URScript string_crossing logic here to determine the dynamic target_q_baseline
                # based on sim_time, current robot state, and the string crossing stage/progress.
                
                # Update string crossing progress for the obs
                if note_duration > 1e-6:
                    self._string_crossing_progress = (sim_time - note_start_time) / note_duration
                    self._string_crossing_progress = np.clip(self._string_crossing_progress, 0.0, 1.0)
                else:
                    self._string_crossing_progress = 1.0

                # PLACEHOLDER: For now, it will simply go to the end Q of the segment if it's a string crossing.
                # Replace with your actual calculated string crossing waypoint/target Q.
                target_q_baseline = current_note_segment['end_q'] # Or a calculated intermediate Q

                # Example of state machine for stages (requires more development)
                # if self._string_crossing_progress < 0.33:
                #     self._string_crossing_stage = 1 # Lift
                #     target_q_baseline = calculate_lift_q(current_q, current_note_segment['start_q'], ...)
                # elif self._string_crossing_progress < 0.66:
                #     self._string_crossing_stage = 2 # Cross
                #     target_q_baseline = calculate_cross_q(current_q, current_note_segment['end_q'], ...)
                # else:
                #     self._string_crossing_stage = 3 # Descend
                #     target_q_baseline = calculate_descend_q(current_q, current_note_segment['end_q'], ...)

        else: # Past all notes
            # If all notes are finished, hold the last known baseline end position
            if self.num_notes > 0:
                target_q_baseline = self.note_targets[self.num_notes - 1]['end_q']
            else:
                target_q_baseline = self.data.qpos[:6] # Hold current position if no notes
            self._string_crossing_progress = 0.0
            self._string_crossing_stage = 0


        action_np = np.asarray(action, dtype=np.float32)
        # Scale the action before adding to baseline
        scaled_action = action_np * self.action_scale 
        target_q_with_residual = target_q_baseline + scaled_action


        # Apply PID control or direct joint position control
        self._apply_pid_control(target_q_with_residual)

        # Step the MuJoCo simulation
        mujoco.mj_step(self.model, self.data)

        # --- Debugging NaN/Inf in MuJoCo State ---
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)) or \
           np.any(np.isnan(self.data.qvel)) or np.any(np.isinf(self.data.qvel)):
            print(f"CRITICAL ERROR: NaN/Inf detected in MuJoCo state (qpos/qvel) after step at time: {self.data.time}")
            print(f"qpos: {self.data.qpos[:6]}")
            print(f"qvel: {self.data.qvel[:6]}")
            # Immediately terminate episode and penalize heavily
            observation = self._get_obs() # Get final observation before terminating
            return observation, -1000, True, False, {'is_failure': True, 'reason': 'mujoco_state_nan_inf'}
        # --- END Debugging ---


        # Render if in human mode
        if self.render_mode == "human":
            self._render_frame()

        # --- Reward Calculation ---
        # Get current robot state for reward calculation
        current_q = self.data.qpos[:6]
        current_tcp_pose_6d = np.zeros(6)
        
        try:
            tcp_site_id = self.model.site_name2id('tool_tip')
            actual_tcp_pos = self.data.site_xpos[tcp_site_id]
            actual_tcp_rot_mat = self.data.site_xmat[tcp_site_id].reshape(3,3)
            r = R.from_matrix(actual_tcp_rot_mat)
            current_tcp_pose_6d[:3] = actual_tcp_pos
            current_tcp_pose_6d[3:] = r.as_rotvec()
        except Exception:
             pass # Handle error if site not found or conversion fails

        # The baseline_tcp_pose for reward calculation must be derived in the same way as in _get_obs
        # This can be obtained by calling _get_obs and extracting, or re-calculating.
        # For simplicity and consistency, let's derive it using the same logic as _get_obs
        # based on current_note_idx and progress_in_current_note (already calculated above)

        baseline_tcp_pose_for_reward = np.zeros(6)
        if current_note_segment: # Use the segment identified earlier in step
            if not current_note_segment['is_string_crossing']:
                start_tcp = current_note_segment['start_tcp']
                end_tcp = current_note_segment['end_tcp']
                baseline_tcp_pose_for_reward = start_tcp + (end_tcp - start_tcp) * progress_in_current_note
            else:
                # This must match the baseline_tcp_pose logic in _get_obs for string crossings
                baseline_tcp_pose_for_reward = current_note_segment['end_tcp'] # Placeholder

        tcp_error = current_tcp_pose_6d - baseline_tcp_pose_for_reward
        
        # Calculate contact forces for penalty (if implemented)
        contact_force_magnitude = 0.0
        # For example, if you have a `contact` module with a function:
        # contact_info = contact.get_cello_bow_string_contact_force(self.model, self.data)
        # if contact_info is not None:
        #     contact_force_magnitude = contact_info['force_magnitude']


        # Example Reward calculation (adjust as per your reward function)
        reward = -self.residual_penalty * np.sum(np.square(action_np)) \
                 -self.contact_penalty * contact_force_magnitude \
                 -0.1 * np.linalg.norm(tcp_error) # Example: penalty for TCP error


        # --- Determine Termination and Truncation ---
        terminated = False
        truncated = False
        info = {}

        # Episode termination conditions (when the episode ends due to "failure" or success)
        if np.linalg.norm(tcp_error) > 0.5: # Example threshold
            terminated = True
            reward -= 100 # Large penalty for failure
            info['is_failure'] = True

        # Check for contact (example, you need a robust contact detection)
        # Assuming you have a way to check if an unwanted contact occurred
        # if contact_force_magnitude > self.CONTACT_THRESHOLD and self.data.time > self.note_sequence[0]['start_time_sec']:
        #     terminated = True
        #     reward -= 50
        #     info['unwanted_contact'] = True

        # Episode truncation conditions (when the episode ends due to time limit or max steps)
        # The episode ends when all notes in the sequence are completed.
        if self.current_note_idx >= self.num_notes:
            # Add a small buffer to ensure the last note's duration is covered
            if self.num_notes > 0 and sim_time >= self.note_targets[self.num_notes - 1]['end_time_sec'] + 0.1: # 0.1s buffer
                truncated = True
                info['is_success'] = not terminated # If it finishes without terminating early
                info['terminal_observation'] = self._get_obs() # Optional: useful for value function if episode ends by truncation
            elif self.num_notes == 0: # Handle case with no notes
                 truncated = True
                 info['is_success'] = not terminated
                 info['terminal_observation'] = self._get_obs()

        # If an episode is terminated, it is generally not also truncated.
        if terminated:
            truncated = False

        observation = self._get_obs()

        # RETURN 5 VALUES
        return observation, reward, terminated, truncated, info

    def _render_frame(self):
        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# Helper function to extract joint angles and timestamps from CSV (might not be used directly by new env)
def extract_joint_angles_and_timestamps(csv_filename):
    df = pd.read_csv(csv_filename)
    joint_cols = ['q_base','q_shoulder','q_elbow','q_wrist1','q_wrist2','q_wrist3']
    return df[joint_cols].values.tolist(), df["time_elapsed_sec"].values.tolist()