import gym
from gym import spaces
import numpy as np
import mujoco
import mujoco.viewer
import pandas as pd
import time
import os
from scipy.interpolate import interp1d

from parsemidi import parse_midi

class UR5eCelloTrajectoryEnv(gym.Env):

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

        # MuJoCo Setup
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = self.model.opt.timestep

        self.viewer = None
        self.render_mode = render_mode

        # PID Control Params
        self.kp, self.kd, self.ki = kp, kd, ki
        self.total_pid_error = np.zeros(6)

        # RL Hyperparams
        self.residual_penalty = residual_penalty
        self.contact_penalty = contact_penalty
        self.torque_penalty = torque_penalty
        self.perpendicularity_penalty = perpendicularity_penalty
        self.timing_penalty = timing_penalty

        # Action Space
        # 6 joints for the UR5e, and actions are continuous residual adjustments
        self.action_scale = action_scale
        self.action_space = spaces.Box(
            low=np.array([-self.action_scale] * 6),
            high=np.array([self.action_scale] * 6),
            dtype=np.float32
        )

        # Baseline CSV
        self.baseline_df = pd.read_csv(baseline_csv_path)
        self.trajectory_times = self.baseline_df['time_elapsed_sec'].values

        # Numerical columns from baseline csv to interpolate
        numerical_cols = [
            'TCP_pose_x', 'TCP_pose_y', 'TCP_pose_z', 'TCP_pose_rx', 'TCP_pose_ry', 'TCP_pose_rz',
            'q_base', 'q_shoulder', 'q_elbow', 'q_wrist1', 'q_wrist2', 'q_wrist3',
            'qd_base', 'qd_shoulder', 'qd_elbow', 'qd_wrist1', 'qd_wrist2', 'qd_wrist3', 
            'remaining_duration_sec' 
        ]

        self.baseline_interp_fns = {}
        # Plan:
        # We have our note_sequence containing the list of notes and string crossings to be played 
        # Instead of interpolating based on time, which might differ betweem the csv data and our rl version,
        # We instead interpolate based on current note
        # Since the interpolation requires numerical x values, we convert the note sequence to a numerical format 
        # Then we break down into two forms of interpolation:
        # 1. Interpolation of the linear non-string crossing note events
        # 2. Interpolation of the string crossing events, which are not linear
        #     -- Note: for kind of string crossing events, base on the way that string crossings are calculated in the song.script string_crossing method:
        #   def string_crossing(start_bow_poses, end_bow_poses, next_dir):
                # local a_to_d_frog=[ 0.03885732, -0.01042583, -0.03909307, -3.27378675, -4.54905543, 2.23515599]
                # local a_to_d_mid=[2.55715648e-03, -8.51565664e-02, -1.08389974e-01, -3.21889652e+00, -4.49939765e+00,  2.21639600e+00]
                # local a_to_d_tip=[-0.03416493, -0.12050286, -0.17569119, -3.16488982, -4.45964209, 2.20089496]

                #     local tcp_pose = get_actual_tcp_pose()
                #     local start_p = [tcp_pose[0], tcp_pose[1], tcp_pose[2], tcp_pose[3], tcp_pose[4], tcp_pose[5]]

                #     local bow_len = norm(end_bow_poses.tip_p - end_bow_poses.frog_p)
                #     local direction_vector = (end_bow_poses.tip_p - end_bow_poses.frog_p) / bow_len

                #     local dist_from_tip = norm(end_bow_poses.tip_p - end_bow_poses.frog_p) * norm(start_p - start_bow_poses.frog_p) / norm(start_bow_poses.tip_p - start_bow_poses.frog_p)

                #     local out = [0.89583677, 0.04158029, 0.44243367, 0, 0, 0]
                #     local step1 = start_p + out * 0.03
                #     # local step2 = end_bow_poses.frog_p + direction_vector *  dist_from_tip * 0.25 #+ out * 0.04
                #     local target_pose = end_bow_poses.frog_p + direction_vector *  dist_from_tip
                #     movep(p[step1[0], step1[1], step1[2], start_p[3], start_p[4], start_p[5]])
                #     movep(p[step1[0], step1[1], step1[2], d_bow_poses.frog_p[3], d_bow_poses.frog_p[4], d_bow_poses.frog_p[5]])
                #     sync()
                #     # movej(p[step2[0], step2[1], step2[2], step2[3], step2[4], step2[5]])
                #     # sync()
                #     movep(p[target_pose[0], target_pose[1], target_pose[2], target_pose[3], target_pose[4], target_pose[5]])
                #     # sync()

                #   end
        for col in numerical_cols:
            if col in self.baseline_df.columns:
                self.baseline_interp_fns[col] = interp1d(
                    self.trajectory_times, self.baseline_df[col].values,
                    kind='linear', fill_value='extrapolate', bounds_error=False
                )
            else:
                print(f"Column '{col}' not found in baseline CSV")


        # --- Fix for ValueError: setting an array element with a sequence ---
        if start_joint_positions is not None:
            # Ensure start_joint_positions is a flat 1D numpy array
            start_joint_positions_np = np.array(start_joint_positions).flatten()
            if start_joint_positions_np.shape[0] != 6:
                raise ValueError(f"start_joint_positions must have 6 elements, but got {start_joint_positions_np.shape[0]}")
            self.data.qpos[:6] = start_joint_positions_np
        else:
            try:
                # This part already uses np.array, which is good
                initial_baseline_q = np.array([self.baseline_interp_fns[f'q_{joint}'](0.0) for joint in ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']])
                self.data.qpos[:6] = initial_baseline_q
            except KeyError as e:
                print(f"Error: Could not get initial joint positions from baseline CSV. Missing joint column: {e}. Defaulting to zeros.")
                self.data.qpos[:6] = np.zeros(6)
            except Exception as e:
                print(f"An unexpected error occurred getting initial qpos: {e}. Defaulting to zeros.")
                self.data.qpos[:6] = np.zeros(6)


        # --- Musical Note Sequence Setup ---
        # Note: 'start_time' and 'end_time' are in the format provided by the user (milliseconds)
        # We need to convert them to seconds for comparison with self.data.time
        try:
            # Convert times to seconds immediately upon loading
            self.note_sequence = sorted([
                {**note,
                 'start_time_sec': note['start_time'] / 1000.0,
                 'end_time_sec': note['end_time'] / 1000.0,
                 'duration_sec': note['duration'] # Assuming duration is already in seconds, or convert if needed
                }
                for note in note_sequence
            ], key=lambda x: x['start_time_sec'])
        except KeyError as e:
            raise KeyError(f"Each note in 'note_sequence' must contain '{e}' keys. Please check your MIDI parsing output.")

        self.current_note_idx = 0

        



        # --- Observation Space ---
        obs_dim = (
            6 + 6 + 6 + 6 + 1 + # Robot State (q, qdot, tcp_pose, tcp_lin_vel, tcp_ang_vel, contact_force)
            6 + 6 + 6 + # Baseline Reference (q_target, qdot_target, tcp_pose_target)
            6 + 6 + # Error from Baseline (q_error, tcp_error)
            1 + 1 + 4 + 2 + 1 + 4 + 2 # Musical Context (progress, time_rem, current_string, bow_dir, is_trans, next_string, next_bow_dir)
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.last_q = self.data.qpos[:6].copy()
         # --- Action Space (ADDED) ---
        # Assuming 6 joints for the UR5e, and actions are continuous residual adjustments
        # The 'action_scale' determines the magnitude of these adjustments.
        self.action_scale = action_scale
        self.action_space = spaces.Box(
            low=np.array([-self.action_scale] * 6),
            high=np.array([self.action_scale] * 6),
            dtype=np.float32
        )



    def _get_obs(self):
        # Time-based features
        sim_time = self.data.time # This is in seconds

        # Get baseline values for current time
        baseline_q = np.array([self.baseline_interp_fns[f'q_{joint}'](sim_time) for joint in ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']])
        baseline_qdot = np.array([self.baseline_interp_fns[f'qd_{joint}'](sim_time) for joint in ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']])
        baseline_tcp_pose = np.array([self.baseline_interp_fns[f'TCP_pose_{dim}'](sim_time) for dim in ['x', 'y', 'z', 'rx', 'ry', 'rz']])


        # Find current note/event in the sequence
        current_event = None
        next_event = None
        for i, event in enumerate(self.note_sequence):
            # Use 'start_time_sec' and 'end_time_sec' which are in seconds
            if event['start_time_sec'] <= sim_time < event['end_time_sec']:
                current_event = event
                self.current_note_idx = i
                if i + 1 < len(self.note_sequence):
                    next_event = self.note_sequence[i+1]
                break
        # Handle case where sim_time might be beyond the last note
        if current_event is None and self.note_sequence and sim_time >= self.note_sequence[-1]['end_time_sec']:
            current_event = self.note_sequence[-1] # Assume last note is current if past end
            self.current_note_idx = len(self.note_sequence) - 1
        elif current_event is None and self.note_sequence and sim_time < self.note_sequence[0]['start_time_sec']:
            # Before the first note, use the first note but adjust time context
            current_event = self.note_sequence[0]
            # Adjust sim_time for context so progress_in_current_note starts near 0 for the first note
            sim_time_for_progress = current_event['start_time_sec'] # Forces progress calculation to be 0 or small
            if len(self.note_sequence) > 1:
                next_event = self.note_sequence[1]
        else: # If note_sequence is empty or current_event is truly outside all notes
            sim_time_for_progress = sim_time # Use actual sim_time
            # Placeholder for context if no notes - ensure it matches expected keys
            current_event = {'start_time_sec': sim_time, 'end_time_sec': sim_time + 1.0, 'duration_sec': 1.0, 'string': '', 'note': '', 'bowing': '', 'is_transition': False}


        # Musical Context Features
        progress_in_current_note = (sim_time_for_progress - current_event['start_time_sec']) / max(1e-6, current_event['duration_sec'])
        time_remaining_in_note = current_event['end_time_sec'] - sim_time


        current_string_one_hot = np.zeros(4) # A, D, G, C
        bow_direction_one_hot = np.zeros(2) # Up, Down
        is_transition = 0.0
        next_string_one_hot = np.zeros(4)
        next_bow_direction_one_hot = np.zeros(2)

        if current_event:
            string_map = {'A': 0, 'D': 1, 'G': 2, 'C': 3}
            if current_event['string'] in string_map:
                current_string_one_hot[string_map[current_event['string']]] = 1.0

            # Determine is_transition based on 'note' field for 'transition'
            is_transition = 1.0 if current_event.get('note') == 'transition' else 0.0

            # *** NOTE: 'bowing' key is missing from your provided note_sequence example. ***
            # The observation space expects it. Please consider adding it to your MIDI parsing.
            # For now, it will default to zeros if not present.
            if current_event.get('bowing') == 'up':
                bow_direction_one_hot[0] = 1.0 # Up
            elif current_event.get('bowing') == 'down':
                bow_direction_one_hot[1] = 1.0 # Down

            if next_event:
                if next_event['string'] in string_map:
                    next_string_one_hot[string_map[next_event['string']]] = 1.0
                if next_event.get('bowing') == 'up':
                    next_bow_direction_one_hot[0] = 1.0
                elif next_event.get('bowing') == 'down':
                    next_bow_direction_one_hot[1] = 1.0

        # Robot State (from MuJoCo)
        actual_q = self.data.qpos[:6]
        actual_q_vel = self.data.qvel[:6]

        try:
            tcp_site_id = self.model.site_name2id('tool_tip')
            actual_tcp_pos = self.data.site_xpos[tcp_site_id]
            actual_tcp_rot_mat = self.data.site_xmat[tcp_site_id].reshape(3,3)
            # You'll need a function to convert actual_tcp_rot_mat to a 3-element orientation representation (e.g., Euler angles, axis-angle)
            # For accurate comparison with baseline_tcp_pose (rx,ry,rz from CSV)
            # Placeholder: assuming actual_tcp_pose_6d is [x,y,z,rx,ry,rz] where rx,ry,rz are axis-angle for consistency
            actual_tcp_pose_6d = np.zeros(6)
            actual_tcp_pose_6d[:3] = actual_tcp_pos # x, y, z
            # TODO: Implement conversion of actual_tcp_rot_mat to appropriate 3-element rotation (rx,ry,rz) for actual_tcp_pose_6d[3:]
            # E.g., from scipy.spatial.transform import Rotation as R
            # r = R.from_matrix(actual_tcp_rot_mat)
            # actual_tcp_pose_6d[3:] = r.as_rotvec() # This gives axis-angle (rx, ry, rz)

            tcp_body_id = self.model.body_name2id('tool0')
            actual_tcp_lin_vel = self.data.body_cvel[tcp_body_id, :3]
            actual_tcp_ang_vel = self.data.body_cvel[tcp_body_id, 3:]

            contact_force_magnitudes = np.zeros(1)
            for i in range(self.data.ncon):
                contact_info = self.data.contact[i]
                geom1_name = self.model.geom_id2name(contact_info.geom1)
                geom2_name = self.model.geom_id2name(contact_info.geom2)
                if ("bow" in geom1_name.lower() and "string" in geom2_name.lower()) or \
                   ("bow" in geom2_name.lower() and "string" in geom1_name.lower()):
                    contact_force_magnitudes[0] = np.linalg.norm(contact_info.force)
                    break

        except Exception as e:
            actual_tcp_pose_6d = np.zeros(6)
            actual_tcp_lin_vel = np.zeros(3)
            actual_tcp_ang_vel = np.zeros(3)
            contact_force_magnitudes = np.zeros(1)

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
            bow_direction_one_hot, # Will be zeros if 'bowing' key is missing
            np.array([is_transition]),
            next_string_one_hot,
            next_bow_direction_one_hot # Will be zeros if 'bowing' key is missing in next_event
        ]).astype(np.float32)

        return obs
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set initial joint positions based on the baseline's start
        initial_baseline_q = np.array([self.baseline_interp_fns[f'q_{joint}'](0.0) for joint in ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']])
        self.data.qpos[:6] = initial_baseline_q
        
        self.data.qvel[:6] = np.zeros(6) # Start with zero velocity
        self.q_error_integral = np.zeros(6) # Reset integral error for PID

        self.current_note_idx = 0
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

        # Apply torque/force control if using that in MuJoCo, or directly set desired velocity for a position controller
        # For position control (type="position" in actuator), just set target position
        # For velocity control (type="velocity"), set target velocity
        # For torque control (type="motor"), set torque
        # Assuming MuJoCo actuators are for torque control (type="motor" or "general")
        # Ensure your model has actuators defined for the UR5e joints that accept torques
        joint_torques = p_term + i_term + d_term
        return joint_torques

    def step(self, action):
        # Apply the residual action to the baseline control
        # Your existing PID control or direct joint control logic
        target_q_baseline = np.array([
            self.baseline_interp_fns[f'q_{joint}'](self.data.time)
            for joint in ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']
        ])
        
        # Add the action (residual) to the baseline target
        # Make sure action is properly scaled and within bounds if necessary
        action_np = np.asarray(action, dtype=np.float32) # Ensure action is a numpy array
        target_q_with_residual = target_q_baseline + action_np

        # Apply PID control or direct joint position control
        self._apply_pid_control(target_q_with_residual)

        # Step the MuJoCo simulation
        mujoco.mj_step(self.model, self.data)

        # Render if in human mode
        if self.render_mode == "human":
            self._render_frame()

        # --- Reward Calculation ---
        # Get current robot state for reward calculation
        current_q = self.data.qpos[:6]
        current_tcp_pose_6d = np.zeros(6) # Placeholder, implement actual TCP pose extraction
        # ... (extract actual_tcp_pose_6d from self.data) ...
        try:
            tcp_site_id = self.model.site_name2id('tool_tip')
            actual_tcp_pos = self.data.site_xpos[tcp_site_id]
            actual_tcp_rot_mat = self.data.site_xmat[tcp_site_id].reshape(3,3)
            # You'll need a function to convert actual_tcp_rot_mat to a 3-element orientation representation (e.g., Euler angles, axis-angle)
            # For accurate comparison with baseline_tcp_pose (rx,ry,rz from CSV)
            # Placeholder: assuming actual_tcp_pose_6d is [x,y,z,rx,ry,rz] where rx,ry,rz are axis-angle for consistency
            current_tcp_pose_6d[:3] = actual_tcp_pos # x, y, z
            # TODO: Implement conversion of actual_tcp_rot_mat to appropriate 3-element rotation (rx,ry,rz) for actual_tcp_pose_6d[3:]
            # E.g., from scipy.spatial.transform import Rotation as R
            # r = R.from_matrix(actual_tcp_rot_mat)
            # current_tcp_pose_6d[3:] = r.as_rotvec() # This gives axis-angle (rx, ry, rz)
        except Exception:
             pass # Handle error if site not found or conversion fails

        baseline_tcp_pose = np.array([self.baseline_interp_fns[f'TCP_pose_{dim}'](self.data.time) for dim in ['x', 'y', 'z', 'rx', 'ry', 'rz']])
        tcp_error = current_tcp_pose_6d - baseline_tcp_pose
        
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
        # Example: if TCP error is too high, or a critical contact occurs
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
        # The episode ends when the simulation time reaches the end of the baseline trajectory.
        # Check if current time is past the end of the baseline trajectory
        if self.data.time >= self.trajectory_times[-1]:
            truncated = True
            # You might give a small bonus for completing the trajectory if it wasn't a failure
            # if not terminated:
            #     reward += 10 # Bonus for reaching end
            info['is_success'] = not terminated # If it finishes without terminating early
            info['terminal_observation'] = self._get_obs() # Optional: useful for value function if episode ends by truncation

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

# Helper function to extract joint angles and timestamps from CSV
def extract_joint_angles_and_timestamps(csv_filename):
    df = pd.read_csv(csv_filename)
    joint_cols = ['q_base','q_shoulder','q_elbow','q_wrist1','q_wrist2','q_wrist3']
    return df[joint_cols].values.tolist(), df["time_elapsed_sec"].values.tolist()