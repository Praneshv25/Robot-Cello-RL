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

sys.path.append(os.path.dirname(os.path.abspath(
    "/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/UR5_Sim/MuJoCo_RL_UR5/gym_grasper/controller/MujocoController.py"
)))
from MujocoController import MJ_Controller

# === Load bowing direction ===
with open("/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Pieces-Bowings/allegro_bowings.txt", "r") as f:
    content = f.read()
    bowing_directions = [x.strip() == "True" for x in content.strip().replace("[", "").replace("]", "").split(",") if x.strip()]

device = torch.device("cpu")
ik_net = IKNet().to(device)
ik_net.load_state_dict(torch.load("ik_net.pth", map_location=device))
ik_net.eval()

x_mean = np.load("x_scaler_mean.npy")
x_scale = np.load("x_scaler_scale.npy")
y_mean = np.load("y_scaler_mean.npy")
y_scale = np.load("y_scaler_scale.npy")

BOW_POSES = RAW_BOW_POSES

def interpolate_pose(p1, p2, alpha):
    """Linearly interpolate between two 6D poses."""
    pos = (1 - alpha) * np.array(p1[:3]) + alpha * np.array(p2[:3])
    rot = (1 - alpha) * np.array(p1[3:]) + alpha * np.array(p2[3:])
    return list(pos) + list(rot)

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
    bow_idx = 0
    event_idx = 0
    print(events.__len__(), "events to process")
    for e in events:
        print(f"Processing event: {e}")
        if e['is_transition']:
            parts = e['string'].split("-")
            if len(parts) != 2:
                continue
            from_str, to_str = parts
            from_frog = BOW_POSES[from_str]['frog']
            from_tip  = BOW_POSES[from_str]['tip']
            to_frog   = BOW_POSES[to_str]['frog']
            to_tip    = BOW_POSES[to_str]['tip']

            # Determine relative progress along from_str string
            if last_pose is None:
                if bowing_directions[event_idx] == 'down':
                    start_frac = 0.0
                else:
                    start_frac = 1.0
            else:
                from_start = np.array(from_frog[:3])
                from_end   = np.array(from_tip[:3])
                total_dist = np.linalg.norm(from_end - from_start)
                cur_dist   = np.linalg.norm(np.array(last_pose[:3]) - from_start)
                start_frac = np.clip(cur_dist / total_dist, 0.0, 1.0)
            # finds a point which is start_frac along the string to_frog to to_tip
            target_pose = interpolate_pose_quat(to_frog, to_tip, start_frac)
            for _ in range(int(0.2 / sim_dt)):
                add_log_entry(target_pose, {
                    'event_label': e.get('event', ''),
                    'current_event_type': 'string_cross',
                    'bow_direction': '',
                    'current_string': e['string'],
                    'remaining_duration_sec': 0.2
                })
                last_pose = target_pose
            event_idx += 1
            continue

        if event_idx >= len(bowing_directions):
            print(f"⚠️  Bowing direction index {event_idx} out of range. Skipping remaining events.")
            break

        string = e['string']
        bowing = bowing_directions[event_idx]
        event_idx += 1
        frog_pose = BOW_POSES[string]['frog']
        tip_pose  = BOW_POSES[string]['tip']

        start_pose = frog_pose if bowing else tip_pose
        end_pose   = tip_pose if bowing else frog_pose

        if last_pose is not None:
            start_pose = last_pose

        total_steps = int(e['duration_sec'] / sim_dt)
        for step in range(total_steps):
            alpha = (step + 1) / total_steps
            if step == 0:
                pose = start_pose
            else: 
                pose = interpolate_pose_quat(start_pose, end_pose, alpha)
            bow_dir = ''
            if bowing:
                bow_dir = 'down' 
            else:
                bow_dir = 'up'
            add_log_entry(pose, {
                'event_label': e.get('event', ''),
                'current_event_type': 'bowing',
                'bow_direction': bow_dir,
                'current_string': string,
                'remaining_duration_sec': e.get('duration_sec', 0.0)
            })
            last_pose = pose

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

# Convert Euler angles to quaternions and interpolate using slerp
def interpolate_pose_quat(p1, p2, alpha):
    """
    Linearly interpolate position, SLERP orientation (quaternion).
    p1, p2: 6D poses [x, y, z, rx, ry, rz] where r* are Euler angles.
    """
    pos1, pos2 = np.array(p1[:3]), np.array(p2[:3])
    r1, r2 = R.from_euler('xyz', p1[3:]), R.from_euler('xyz', p2[3:])
    q1, q2 = r1.as_quat(), r2.as_quat()  # [x, y, z, w]

    # SLERP
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        interp_quat = q1 + alpha * (q2 - q1)
        interp_quat /= np.linalg.norm(interp_quat)
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * alpha
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        interp_quat = s0 * q1 + s1 * q2

    interp_rot = R.from_quat(interp_quat)
    interp_euler = interp_rot.as_euler('xyz', degrees=False)
    interp_pos = (1 - alpha) * pos1 + alpha * pos2

    return list(interp_pos) + list(interp_euler)



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
