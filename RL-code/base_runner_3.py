import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch
import sys, os
import pandas as pd
import mujoco
import math 
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


def calculate_point_on_string(from_point, to_point, frac_of_bow):
    # vector to_point to from_point by subtracting from_point from to_point
    direction_vector = np.array(to_point) - np.array(from_point)
    # normalize this vector by dividing from its length
    length = np.linalg.norm(direction_vector)
    if length == 0:
        return from_point
    direction_vector /= length
    # this point dist is a fraction of the total length from frog to tip
    this_point_dist = frac_of_bow * length
    # multiply the normalized vector by the given distance

    offset_vector = np.array(from_point) + np.array(direction_vector) * this_point_dist
    return offset_vector

def calculate_rotations(starting_rotation, ending_rotation, sim_dt, current_duration):
    """
    calculate the interpolated rotation between two rotation values for each simulation step
    time 0 will be starting_rotation, time 3.0 will be ending_rotation
    """
    # fix so we calculate the rotation that is event_duration + curr_time away from the curr_rotation and return this calculated rotation
    print(starting_rotation)
    print(ending_rotation)
    #r_start = R.from_euler('xyz', starting_rotation) 
    #r_end = R.from_euler('xyz', ending_rotation)
    print("Current duration for key times definitiion")
    print(current_duration)
    if current_duration >= 3.0:
        # instead of using slerp we just have our ending_rotation returned 
        return [ending_rotation]
    
    # Updated key_times to use current_duration as the start time for interpolation
    key_times = [current_duration, 3.0] 
    rotations = R.from_euler('xyz', [starting_rotation, ending_rotation])
    slerp = Slerp(key_times, rotations)
    times = np.arange(current_duration, 3.0, sim_dt)
    print(times)
    interpolated = slerp(times)
    return interpolated.as_euler('xyz', degrees=False).tolist()

def calculate_rotations_sx(starting_rotation, ending_rotation, sim_dt):
    print(starting_rotation)
    print(ending_rotation)
    #r_start = R.from_euler('xyz', starting_rotation) 
    #r_end = R.from_euler('xyz', ending_rotation)

    key_times = [0, 0.2]
    rotations = R.from_euler('xyz', [starting_rotation, ending_rotation])
    slerp = Slerp(key_times, rotations)
    times = np.arange(0.0, 0.2, sim_dt)
    interpolated = slerp(times)
    return interpolated.as_euler('xyz', degrees=False).tolist()

def generate_trajectory(events, sim_dt=0.01):
    trajectory_cartesian = []
    detailed_log = []
    start_time = time.time()
    timestamp_robot = 0.0

    def add_log_entry(pose, label_info):
        nonlocal timestamp_robot
        xyz = pose[:3]
        rpy = pose[3:]
        if len(rpy) != 3:
            print(f"⚠️ Skipping invalid pose due to malformed RPY: {rpy}")
            return
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
    dist_from_frog = 0.0 
    dist_from_tip = 0.0
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
                if e['bowing'] == 'down':
                    dist_from_tip = 3.0
                    start_frac_xyz = 0.0
                    cur_pos_xyz = np.array(from_frog[:3])
                else: 
                    dist_from_frog = 3.0
                    start_frac_xyz = 1.0
                    cur_pos_xyz = np.array(from_tip[:3])
            else:
                from_frog_xyz = np.array(from_frog[:3])
                from_frog_rxryrz = np.array(from_frog[3:])
                to_frog_rxryrz = np.array(to_frog[3:])
                from_tip_xyz   = np.array(from_tip[:3])
                from_tip_rxryrz = np.array(from_tip[3:])
                to_tip_rxryrz = np.array(to_tip[3:])
                cur_pos_xyz    = np.array(last_pose[:3])
                cur_pos_rxryrz = np.array(last_pose[3:])
                total_from_dist_xyz = np.linalg.norm(from_frog_xyz - from_tip_xyz)
                if e['bowing'] == 'down':
                    cur_dist_xyz   = np.linalg.norm(cur_pos_xyz - from_frog_xyz)
                else:
                    # want positive distance 
                    cur_dist_xyz = np.linalg.norm(from_tip_xyz - cur_pos_xyz)
                # fraction along the bow towards the target 
                start_frac_xyz = np.clip(cur_dist_xyz / total_from_dist_xyz, 0.0, 1.0)

            # Interpolate pose on the new string using that same fraction
            if e['bowing'] == 'down':
                to_pos = calculate_point_on_string(to_frog[:3], to_tip[:3], start_frac_xyz)
                # to_pos = (1 - start_frac_xyz) * np.array(to_frog[:3]) + start_frac_xyz * np.array(to_tip[:3])
                # two rotations to interpolate between are from_frog and to_tip, where the current rotation is last_pose
                # dist_from_tip represents the duration at last_pose (duration to go from frog rotation to last_pose rotation)
                # goal is to figure out rotation between last_pose and to_tip which results 
                if last_pose is None: 
                    rotations = calculate_rotations_sx(from_frog_rxryrz, to_frog_rxryrz, sim_dt)
                else:
                    total_entries = 3.0 / 0.01 # 300
                    print("Start fraction xyz")
                    print(start_frac_xyz)
                    curr_steps = round(start_frac_xyz * total_entries)
                    target_rotations = calculate_rotations(to_frog_rxryrz, to_tip_rxryrz, sim_dt, 3.0)
                    if curr_steps < len(target_rotations):
                        target_rxryrz = target_rotations[curr_steps]
                    else:
                        target_rxryrz = to_tip_rxryrz
                    rotations = calculate_rotations_sx(cur_pos_rxryrz, target_rxryrz, sim_dt)
            else:
                # fix to account for the fact that going from tip to frog is sort of "negative"
                to_pos = calculate_point_on_string(to_tip[:3], to_frog[:3], start_frac_xyz)
                #to_pos = (1 - start_frac_xyz) * np.array(to_tip[:3]) + start_frac_xyz * np.array(to_frog[:3])
                if last_pose is None:
                    rotations = calculate_rotations_sx(from_tip_rxryrz, to_tip_rxryrz, sim_dt)
                else: 
                    total_entries = 3.0 / 0.01
                    curr_steps = round(start_frac_xyz * total_entries)
                    print("Current steps")
                    print(curr_steps)
                    print("Start frac xyz")
                    print(start_frac_xyz)
                    print("Length of rotations: ")
                    print(len(calculate_rotations(to_tip_rxryrz, to_frog_rxryrz, sim_dt, 3.0)))
                    target_rotations = calculate_rotations(to_tip_rxryrz, to_frog_rxryrz, sim_dt, 3.0)
                    if curr_steps < len(target_rotations):
                        target_rxryrz = target_rotations[int(curr_steps)]
                    else:
                        target_rxryrz = to_frog_rxryrz
                    rotations = calculate_rotations_sx(cur_pos_rxryrz, target_rxryrz, sim_dt)
            # to_pos is the xyz position we need to get to; we can access rotations[step] to get the desired rotation 
            #target_pose = list(to_pos) + to_rot
            total_steps = int(0.2 / sim_dt)
            pose_fencepost = None
            for step in range(total_steps):
                frac = step / total_steps
                xyz = calculate_point_on_string(cur_pos_xyz, to_pos, frac)
                #xyz = (1 - (step/total_steps)) * np.array(cur_pos_xyz) + (step/total_steps) * np.array(to_pos)
                print("Rotations")
                print(rotations)
                rpy = rotations[step]
                pose = list(xyz) + rpy
                remaining_duration_sec = e['duration_sec'] - step * 0.01
                add_log_entry(pose, {
                    'event_label': e.get('event', ''),
                    'current_event_type': 'string_cross',
                    'bow_direction': e.get('bowing', ''),
                    'current_string': e['string'],
                    'remaining_duration_sec': remaining_duration_sec
                })
                pose_fencepost = pose
                print(pose_fencepost)
            last_pose = pose_fencepost
            
        else: # straight bowing
            string = e['string']
            frog = BOW_POSES[string]['frog']
            tip  = BOW_POSES[string]['tip']
            total_dist = np.linalg.norm(np.array(tip[:3]) - np.array(frog[:3]))
            # calculate number of steps for this event 
            total_steps = int(e['duration_sec'] / sim_dt)
            # calculate cap number of steps (3.0 duration, time for full span from start to end pos)
            full_bow_steps = int(3.0 / sim_dt)
            # calculate duration until start of this event
            event_rotations = None
            if last_pose is None: # this is the first pose, we will be at the frog or tip
                curr_dur = 0
                curr_dist = 0
                # fraction of bow to use is the number of steps needed for this note over number of steps needed to play full bow
                # since we are starting at either the tip or the frog, there is no risk thath we will have to shorten the duration of this note
                # (unless the duration of this note is somehow greater than 3)
                frac_of_bow = total_steps / full_bow_steps
                if frac_of_bow > 1:
                    # we maximally need to use the full bow
                    frac_of_bow = 1
                if e['bowing'] == 'down':
                    target_pose = calculate_point_on_string(frog[:3], tip[:3], frac_of_bow)
                    #target_pose = (1 - frac_of_bow) * np.array(frog[:3]) + frac_of_bow * np.array(tip[:3])
                    event_rotations = calculate_rotations(frog[3:], tip[3:], sim_dt, 0)
                    print(event_rotations)
                    print("frog tipppp")
                else: 
                    #target_pose = frac_of_bow * np.array(frog[:3]) + (1 - frac_of_bow) * np.array(tip[:3])
                    target_pose = calculate_point_on_string(tip[:3], frog[:3], frac_of_bow)
                    event_rotations = calculate_rotations(tip[3:], frog[3:], sim_dt, 0)
                    print(event_rotations)
                    print("tip froggg")
            else: # not necessarily at frog or tip
                if e['bowing'] == 'down':
                    # duration to get from frog to last_pos
                    # calculate distance between frog an last_pos (xyz only)
                    curr_dist = np.linalg.norm(np.array(last_pose[:3]) - np.array(frog[:3]))
                    # distance is a fraction of the total distance from frog to tip, mult by 3 to get duration
                    curr_dur = 3 * (curr_dist / total_dist)
                    dur_left = 3 - curr_dur
                    if e['duration_sec'] >= dur_left:
                        # we end at the tip
                        target_pose = tip[:3]
                    else:
                        # we end some fraction from the tip 
                        reduced_bow_steps = int(dur_left / sim_dt)
                        reduced_frac_of_bow = total_steps / reduced_bow_steps
                        target_pose = calculate_point_on_string(last_pose[:3], tip[:3], reduced_frac_of_bow)
                        # target_pose = (1 - reduced_frac_of_bow) * np.array(last_pose[:3]) + (reduced_frac_of_bow) * np.array(tip[:3])
                    if last_pose is None:
                        event_rotations =  calculate_rotations(frog[3:], tip[3:], sim_dt, 0)
                        print(event_rotations)
                        print("frog tip event rote")
                    else:
                        print("Last pose")
                        print(last_pose)
                        print("Tip")
                        print(tip)
                        event_rotations = calculate_rotations(last_pose[3:], tip[3:], sim_dt, curr_dur)
                        print(curr_dur)
                        print(event_rotations)
                        print("last pose tip event rot")
                else:
                    # duration to get from tip to last_pos (xyz only)
                    # calculate distance between tip and last_pos
                    curr_dist = np.linalg.norm(np.array(tip[:3]) - np.array(last_pose[:3]))
                    # distance is a fraction of the total distance from frog to tip, mult by 3 to get duration
                    curr_dur = 3 * (curr_dist / total_dist)
                    dur_left = 3 - curr_dur
                    if e['duration_sec'] > dur_left:
                        # we end at the tip
                        target_pose = frog[:3]
                    else:
                        # we end some fraction from the tip 
                        reduced_bow_steps = int(dur_left / sim_dt)
                        reduced_frac_of_bow = total_steps / reduced_bow_steps
                        target_pose = calculate_point_on_string(last_pose[:3], frog[:3], reduced_frac_of_bow)
                        # target_pose = (1 - reduced_frac_of_bow) * np.array(last_pose[:3]) + (reduced_frac_of_bow) * np.array(frog[:3])
                    if last_pose is None:
                        event_rotations = calculate_rotations(tip[3:], frog[3:], sim_dt, 0)
                        print(event_rotations)
                        print("tip frog event rotation")
                    else:
                        event_rotations = calculate_rotations(last_pose[3:], frog[3:], sim_dt, curr_dur)
                        print("last pose event rotation")
                        print(last_pose[3:])
                        print(event_rotations)
            pose_fencepost = None
            for step in range(total_steps):
                # iterate through all simulation steps for the current note event 
                frac_towards_target = step / total_steps
                # first event and first step, curr position will be at the frog or tip
                if step == 0 and last_pose is None:
                    if e['bowing'] == 'down':
                        # at frog of curr string
                        curr_pos = frog
                        xyz = frog[:3]
                        rpy = frog[3:]
                    else:
                        # at tip of curr string
                        curr_pos = tip
                        xyz = tip[:3]
                        rpy = tip[3:]
                elif step == 0 and last_pose is not None:
                # first step of this event, starting position should be at last_pose
                    curr_pos = last_pose
                    xyz = last_pose[:3]
                    rpy = last_pose[3:]
                elif step != 0 and last_pose is None: 
                    if e['bowing'] == 'down':
                        xyz = calculate_point_on_string(frog[:3], target_pose, frac_towards_target)
                        #xyz = (1 - frac_towards_target) * np.array(frog[:3]) + frac_towards_target * np.array(target_pose)
                    else:
                        xyz = calculate_point_on_string(tip[:3], target_pose, frac_towards_target)
                        #xyz = (1 - frac_towards_target) * np.array(target_pose) + frac_towards_target * np.array(tip[:3])
                    print()
                    if step < len(event_rotations): # Changed from <= to <
                        rpy = event_rotations[step]
                    else:
                        rpy = event_rotations[-1]
                elif step != 0 and last_pose is not None:
                # not the first step, will be some ways from last_pose towards the target_pose
                    print("Step:")
                    print(step)
                    print("Last pose:")
                    print(last_pose)
                    if last_pose is None: 
                        print(f"💥 Invalid pose: last_pose={last_pose}, target_pose={target_pose}")
                        continue  # skip this step
                    if e['bowing'] == 'down':
                        xyz = calculate_point_on_string(last_pose[:3], target_pose, frac_towards_target)
                        #xyz = (1 - frac_towards_target) * np.array(last_pose[:3]) + frac_towards_target * np.array(target_pose)
                    else:
                        xyz = calculate_point_on_string(last_pose[:3], target_pose, frac_towards_target)
                        #xyz = (1 - frac_towards_target) * np.array(target_pose)+ frac_towards_target * np.array(last_pose[:3])
                    print("Event rotations length")
                    print(len(event_rotations))
                    print(event_rotations)
                    if step < len(event_rotations): # Changed from <= to <
                        rpy = event_rotations[step]
                    else:
                        rpy = event_rotations[-1]

                pose = list(xyz) + list(rpy)
                print("pose")
                print(pose)
                remaining_duration_sec = e['duration_sec'] - step * 0.01
                add_log_entry(pose, {
                    'event_label': e.get('event', ''),
                    'current_event_type': 'bowing',
                    'bow_direction': e.get('bowing', ''),
                    'current_string': e.get('string', ''),
                    'remaining_duration_sec': remaining_duration_sec
                })
                pose_fencepost = pose
                print("pose_fencepost")
                print(pose_fencepost)
            last_pose = pose_fencepost
            curr_dur = 0
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