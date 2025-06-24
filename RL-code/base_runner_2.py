import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import sys, os
import pandas as pd
import mujoco

sys.path.append("/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Data-Files/Big-Logs")
from ik_nn import IKNet
from robot_runner_detailed_logs import BOW_POSES as RAW_BOW_POSES, parse_midi
from rl_trajectory import UR5eCelloTrajectoryEnv


device = torch.device("cpu")
ik_net = IKNet().to(device)
ik_net.load_state_dict(torch.load("ik_net.pth", map_location=device))
ik_net.eval()

x_mean = np.load("x_scaler_mean.npy")
x_scale = np.load("x_scaler_scale.npy")
y_mean = np.load("y_scaler_mean.npy")
y_scale = np.load("y_scaler_scale.npy")

BOW_POSES = RAW_BOW_POSES


def calculate_point_on_string(from_point, to_point, distance):
    # vector to_point to from_point by subtracting from_point from to_point
    direction_vector = np.array(from_point[:3]) - np.array(to_point[:3])
    # normalize this vector by dividing from its length
    length = np.linalg.norm(direction_vector)
    if length == 0:
        return from_point
    direction_vector /= length
    # multiply the normalized vector by the given distance
    offset_vector = direction_vector * distance
    return offset_vector

def calculate_rotations(from_rot, to_rot, duration):
    """
    calculate the interpolated rotation between two rotation values given a number of total steps and the current step,
    uses SLERP
    """
    
    # calculate shortest path between from_rot and to_rot using SLERP
    r1 = R.from_euler('xyz', from_rot)
    r2 = R.from_euler('xyz', to_rot)
    dot = np.dot(r1.as_quat(), r2.as_quat())
    if dot < 0.0:
        r2 = -r2
        dot = -dot  # Ensure dot product is non-negative for SLERP  
    if dot > 0.9995:
        # If the angles are very close, use linear interpolation
        interp_rot = r1.as_quat() + (r2.as_quat() - r1.as_quat()) * duration
        interp_rot /= np.linalg.norm(interp_rot)
    else: 
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * duration
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        interp_rot = s0 * r1.as_quat() + s1 * r2.as_quat()
        interp_rot /= np.linalg.norm(interp_rot)
    # interp_rot is a quaternion [x, y, z, w] 
    interp_euler = R.from_quat(interp_rot).as_euler('xyz', degrees=False)
    return interp_euler.tolist()

def generate_trajectory(events, sim_dt=0.01):
    trajectory_cartesian = []
    detailed_log = []
    start_time = time.time()
    timestamp_robot = 0.0

    def add_log_entry(pose, label_info):
        nonlocal timestamp_robot
        xyz = pose[:3]
        rpy = pose[3:]
        try:
            rotvec = R.from_euler('xyz', rpy).as_rotvec()
            input_data = np.concatenate((xyz, rotvec))
            input_scaled = (input_data - x_mean) / x_scale
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_scaled = ik_net(input_tensor).cpu().numpy().flatten()
                pred_q = pred_scaled * y_scale + y_mean
        except Exception as e:
            print(f"❌ Error during IK prediction: {e}")
            return

        log_entry = {
            'timestamp_robot': timestamp_robot,
            'time_elapsed_sec': time.time() - start_time,
            'event_label': label_info.get('event_label', ''),
            'current_event_type': label_info.get('current_event_type', ''),
            'bow_direction': label_info.get('bow_direction', ''),
            'current_string': label_info.get('current_string', ''),
            'remaining_duration_sec': label_info.get('remaining_duration_sec', 0.0),
            'calc_TCP_pose_x': xyz[0],
            'calc_TCP_pose_y': xyz[1],
            'calc_TCP_pose_z': xyz[2],
            'calc_TCP_pose_rx': rpy[0],
            'calc_TCP_pose_ry': rpy[1],
            'calc_TCP_pose_rz': rpy[2],
            'pred_q_base': pred_q[0],
            'pred_q_shoulder': pred_q[1],
            'pred_q_elbow': pred_q[2],
            'pred_q_wrist1': pred_q[3],
            'pred_q_wrist2': pred_q[4],
            'pred_q_wrist3': pred_q[5]
        }
        detailed_log.append(log_entry)
        timestamp_robot += sim_dt
        trajectory_cartesian.append(pose)

    last_pose = None
    event_idx = 0
    for e in events:
        if e['is_transition']:
            # --- String crossing logic ---
            from_str, to_str = e['string'].split("-")
            from_frog = BOW_POSES[from_str]['frog']
            from_tip  = BOW_POSES[from_str]['tip']
            to_frog   = BOW_POSES[to_str]['frog']
            to_tip    = BOW_POSES[to_str]['tip']

            # Compute current fractional progress along old string
            if last_pose is None:
                start_frac = 0.0 if e['bowing'] == 'down' else 1.0
            else:
                from_start = np.array(from_frog[:3])
                from_end   = np.array(from_tip[:3])
                cur_pos    = np.array(last_pose[:3])
                total_dist = np.linalg.norm(from_end - from_start)
                cur_dist   = np.linalg.norm(cur_pos - from_start)
                start_frac = np.clip(cur_dist / total_dist, 0.0, 1.0)

            # Interpolate pose on the new string using that same fraction
            to_pos = (1 - start_frac) * np.array(to_frog[:3]) + start_frac * np.array(to_tip[:3])
            to_rot = calculate_rotations(to_frog[3:], to_tip[3:], start_frac)
            target_pose = list(to_pos) + to_rot

            for _ in range(int(0.2 / sim_dt)):
                add_log_entry(target_pose, {
                    'event_label': e.get('event', ''),
                    'current_event_type': 'string_cross',
                    'bow_direction': e.get('bowing', ''),
                    'current_string': e['string'],
                    'remaining_duration_sec': 0.2
                })
                last_pose = target_pose
            continue

        # --- Bowing motion logic ---
        frog_pose = BOW_POSES[e['string']]['frog']
        tip_pose  = BOW_POSES[e['string']]['tip']
        start_pose = frog_pose if e['bowing'] == 'down' else tip_pose
        end_pose   = tip_pose if e['bowing'] == 'down' else frog_pose

        if last_pose is not None:
            start_xyz = np.array(last_pose[:3])
            start_rot = last_pose[3:]
        else:
            start_xyz = np.array(start_pose[:3])
            start_rot = start_pose[3:]

        end_xyz = np.array(end_pose[:3])
        end_rot = end_pose[3:]

        total_steps = int(e['duration_sec'] / sim_dt)
        for step in range(total_steps):
            alpha = (step + 1) / total_steps
            xyz = (1 - alpha) * start_xyz + alpha * end_xyz
            rpy = calculate_rotations(start_rot, end_rot, alpha)
            pose = list(xyz) + rpy
            
            add_log_entry(pose, {
                'event_label': e.get('event', ''),
                'current_event_type': 'bowing',
                'bow_direction': e.get('bowing', ''),
                'current_string': e.get('string', ''),
                'remaining_duration_sec': e.get('duration_sec', 0.0)
            })
            last_pose = pose
        event_idx += 1

    df = pd.DataFrame(detailed_log)
    df.to_csv("nn_predicted_log.csv", index=False)
    print("✅ Saved log to nn_predicted_log.csv")
    return [[entry[f'pred_q_{j}'] for j in ['base','shoulder','elbow','wrist1','wrist2','wrist3']] for entry in detailed_log]

def simulate_trajectory(trajectory, model_path, sim_dt=0.01):
    if len(trajectory) == 0:
        print("❌ No valid joint-space trajectory was generated.")
        return
    env = UR5eCelloTrajectoryEnv(
        model_path=model_path,
        trajectory=trajectory,
        note_sequence=[],
        render_mode='human',
        action_scale=0.0,
        residual_penalty=0.0,
        contact_penalty=0.0,
        torque_penalty=0.0,
        kp=0.0, kd=0.0, ki=0.0,
        start_joint_positions=trajectory[0]
    )

    for joint_angles in trajectory:
        env.data.qpos[:6] = joint_angles
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.render()
        time.sleep(sim_dt)

    env.close()
    print("✅ Simulation finished.")

from scipy.spatial.transform import Rotation as R




def main():
    midi_file = "/Users/skamanski/Documents/GitHub/Robot-Cello/midi_robot_pipeline/midi_files/allegro.mid"
    model_path = "/Users/skamanski/Documents/GitHub/Robot-Cello/MuJoCo_RL_UR5/env_experiment/universal_robots_ur5e/scene.xml"
    events = parse_midi(midi_file)
    print(events)
    bowing_directions = [event["bowing"] for event in events]
    # bowing_directions = 
    # controller = MJ_Controller(model_path, viewer=False)
    # print("⏳ Precomputing trajectory...")
    trajectory = generate_trajectory(events)
    # print(f"✅ Trajectory generated with {len(trajectory)} poses.")
    simulate_trajectory(trajectory, model_path)

if __name__ == "__main__":
    main()
