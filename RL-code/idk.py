import pandas as pd
import numpy as np
import torch
import time
import mujoco
import os
import sys
from scipy.spatial.transform import Rotation as R

# Import your model and environment
sys.path.append("/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Data-Files/Big-Logs")
from ik_nn import IKNet  # Your trained model definition
from rl_trajectory import UR5eCelloTrajectoryEnv

# Load neural network
device = torch.device("cpu")
ik_net = IKNet().to(device)
ik_net.load_state_dict(torch.load("ik_net.pth", map_location=device))
ik_net.eval()

# Load normalization parameters
x_mean = np.load("x_scaler_mean.npy")
x_scale = np.load("x_scaler_scale.npy")
y_mean = np.load("y_scaler_mean.npy")
y_scale = np.load("y_scaler_scale.npy")

# Load log CSV
log_path = "/Users/skamanski/Documents/GitHub/Robot-Cello/biglogs/minuet_no_2v2-log-detailed.csv"
log_df = pd.read_csv(log_path)

# Rename columns to match expected input
log_df = log_df.rename(columns={
    'TCP_pose_x': 'x', 'TCP_pose_y': 'y', 'TCP_pose_z': 'z',
    'TCP_pose_rx': 'rx', 'TCP_pose_ry': 'ry', 'TCP_pose_rz': 'rz'
})

required_fields = ['x', 'y', 'z', 'rx', 'ry', 'rz']
predicted_joint_angles = []
true_joint_angles = []
errors = []

# Predict joint angles
for i, row in log_df.iterrows():
    try:
        if not all(field in row and pd.notna(row[field]) for field in required_fields):
            raise ValueError("Missing or NaN in input fields")

        input_pose_np = np.array([
            row['x'], row['y'], row['z'],
            row['rx'], row['ry'], row['rz']
        ])

        input_pose_scaled = (input_pose_np - x_mean) / x_scale
        input_pose = torch.tensor(input_pose_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = ik_net(input_pose).cpu().numpy().flatten()

        pred = pred_scaled * y_scale + y_mean
        predicted_joint_angles.append(pred.tolist())

        true_joint = [
            row['q_base'], row['q_shoulder'], row['q_elbow'],
            row['q_wrist1'], row['q_wrist2'], row['q_wrist3']
        ]
        true_joint_angles.append(true_joint)
        errors.append(np.linalg.norm(pred - np.array(true_joint)))

    except Exception as e:
        print(f"Error processing row {i}: {e}")

# Only keep matching number of rows
min_len = min(len(predicted_joint_angles), len(true_joint_angles))
predicted_joint_angles = predicted_joint_angles[:min_len]
true_joint_angles = true_joint_angles[:min_len]
print("Predicted joint angles:")
print(predicted_joint_angles[:30])
print("True joint angles:")
print(true_joint_angles[:30])
print(f"✅ Average prediction error: {np.mean(errors):.4f} radians")

# Simulation
def simulate_dual_trajectories(pred_traj, true_traj, model_path, sim_dt=0.01):
    if not pred_traj or not true_traj:
        print("❌ Missing one or both joint-space trajectories.")
        return

    # Load two independent simulation environments
    env_pred = UR5eCelloTrajectoryEnv(
        model_path=model_path,
        trajectory=pred_traj,
        note_sequence=[],
        render_mode='human',
        action_scale=0.0,
        residual_penalty=0.0,
        contact_penalty=0.0,
        torque_penalty=0.0,
        kp=0.0, kd=0.0, ki=0.0,
        start_joint_positions=pred_traj[0]
    )

    env_true = UR5eCelloTrajectoryEnv(
        model_path=model_path,
        trajectory=true_traj,
        note_sequence=[],
        render_mode='human',
        action_scale=0.0,
        residual_penalty=0.0,
        contact_penalty=0.0,
        torque_penalty=0.0,
        kp=0.0, kd=0.0, ki=0.0,
        start_joint_positions=true_traj[0]
    )

    max_steps = min(len(pred_traj), len(true_traj))

    for i in range(max_steps):
        env_pred.data.qpos[:6] = pred_traj[i]
        env_pred.data.qvel[:] = 0.0
        mujoco.mj_forward(env_pred.model, env_pred.data)

        env_true.data.qpos[:6] = true_traj[i]
        env_true.data.qvel[:] = 0.0
        mujoco.mj_forward(env_true.model, env_true.data)

        env_pred.render()
        env_true.render()
        time.sleep(sim_dt)

    env_pred.close()
    env_true.close()
    print("✅ Dual simulation finished.")

# Run simulation
model_path = "/Users/skamanski/Documents/GitHub/Robot-Cello/MuJoCo_RL_UR5/env_experiment/universal_robots_ur5e/scene.xml"
simulate_dual_trajectories(predicted_joint_angles, true_joint_angles, model_path)
