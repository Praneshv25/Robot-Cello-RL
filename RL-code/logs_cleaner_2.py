import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# --- 1. Define Waypoints ---
# Convert your struct-like data into a more usable format, e.g., dictionaries
# Each pose is [x, y, z, rx, ry, rz]

bow_poses = {
    'A_tip': [.473129539189, .413197423330, .256308427905, -1.460522581833, -2.310115543652, 1.445803824327],
    'A_frog': [.300717266074, .793568239540, .099710283103, -1.543522183454, -2.354885618328, 1.346770272474],
    'D_tip': [.340413993945, .280157415162, .176342071758, -1.614553612482, -2.044810993523, 1.042279199535],
    'D_frog': [.302785064368, .749849181019, .117254426008, -1.664082298752, -2.084265434693, 1.037965163360],
    'G_tip': [.162016291992, .201320984957, .059414774157, -1.929772636560, -1.931323067217, .555055912517],
    'G_frog': [.281203376642, .681662588607, .104672526365, -1.812194031755, -1.940153681829, .493747597283],
    'C_tip': [.079815569355, .285182178102, -.086654726588, -1.819646014269, -1.658258006768, .180930717120],
    'C_frog': [.256662516098, .610082591416, .062624387196, -1.743236422252, -1.524514092756, .163823228357]
}

# Separate note types for easier lookup
note_mapping = {
    'A_tip': 'A', 'A_frog': 'A',
    'D_tip': 'D', 'D_frog': 'D',
    'G_tip': 'G', 'G_frog': 'G',
    'C_tip': 'C', 'C_frog': 'C',
}


# --- 2. Create Sample RTDE Data (Replace with your actual data loading) ---
# In a real scenario, you would load your RTDE log file into a DataFrame.
data = {
    'timestamp_robot': np.arange(0, 100000000000, 0.1),
    'TCP_pose_x': np.linspace(bow_poses['A_tip'][0], bow_poses['A_frog'][0], 100) + np.random.normal(0, 0.01, 100),
    'TCP_pose_y': np.linspace(bow_poses['A_tip'][1], bow_poses['A_frog'][1], 100) + np.random.normal(0, 0.01, 100),
    'TCP_pose_z': np.linspace(bow_poses['A_tip'][2], bow_poses['A_frog'][2], 100) + np.random.normal(0, 0.01, 100),
    'TCP_pose_rx': np.linspace(bow_poses['A_tip'][3], bow_poses['A_frog'][3], 100) + np.random.normal(0, 0.01, 100),
    'TCP_pose_ry': np.linspace(bow_poses['A_tip'][4], bow_poses['A_frog'][4], 100) + np.random.normal(0, 0.01, 100),
    'TCP_pose_rz': np.linspace(bow_poses['A_tip'][5], bow_poses['A_frog'][5], 100) + np.random.normal(0, 0.01, 100),
    'q_base': np.zeros(100), 'q_shoulder': np.zeros(100), 'q_elbow': np.zeros(100),
    'q_wrist1': np.zeros(100), 'q_wrist2': np.zeros(100), 'q_wrist3': np.zeros(100),
    'TCP_force_x': np.zeros(100), 'TCP_force_y': np.zeros(100), 'TCP_force_z': np.zeros(100),
    'TCP_force_rx': np.zeros(100), 'TCP_force_ry': np.zeros(100), 'TCP_force_rz': np.zeros(100),
}
df = pd.DataFrame(data)

# Add a few frames that are closer to 'D' to show detection
df.loc[50:55, 'TCP_pose_x'] = np.linspace(bow_poses['D_tip'][0], bow_poses['D_frog'][0], 6)
df.loc[50:55, 'TCP_pose_y'] = np.linspace(bow_poses['D_tip'][1], bow_poses['D_frog'][1], 6)
df.loc[50:55, 'TCP_pose_z'] = np.linspace(bow_poses['D_tip'][2], bow_poses['D_frog'][2], 6)
df.loc[50:55, 'TCP_pose_rx'] = np.linspace(bow_poses['D_tip'][3], bow_poses['D_frog'][3], 6)
df.loc[50:55, 'TCP_pose_ry'] = np.linspace(bow_poses['D_tip'][4], bow_poses['D_frog'][4], 6)
df.loc[50:55, 'TCP_pose_rz'] = np.linspace(bow_poses['D_tip'][5], bow_poses['D_frog'][5], 6)


# --- 3. Define a distance function (Example: Weighted Euclidean) ---
def pose_distance(current_pose, target_pose, pos_weight=1.0, ori_weight=0.5):
    """
    Calculates a weighted Euclidean distance between two 6D poses.
    current_pose, target_pose: lists/arrays of [x, y, z, rx, ry, rz]
    pos_weight: weighting for position (m)
    ori_weight: weighting for orientation (rad)
    """
    pos_dist = euclidean(current_pose[:3], target_pose[:3]) # meters
    ori_dist = euclidean(current_pose[3:], target_pose[3:]) # radians

    # A simple way to combine: square, weight, sum, then sqrt
    return np.sqrt(pos_weight * pos_dist**2 + ori_weight * ori_dist**2)

# --- 4. Apply the prediction logic ---
predicted_notes = []
for index, row in df.iterrows():
    current_tcp_pose = [
        row['TCP_pose_x'], row['TCP_pose_y'], row['TCP_pose_z'],
        row['TCP_pose_rx'], row['TCP_pose_ry'], row['TCP_pose_rz']
    ]

    min_distance = float('inf')
    closest_waypoint_name = 'Unknown'

    for wp_name, wp_pose in bow_poses.items():
        dist = pose_distance(current_tcp_pose, wp_pose, pos_weight=1.0, ori_weight=0.1) # Adjust weights as needed
        if dist < min_distance:
            min_distance = dist
            closest_waypoint_name = wp_name

    # Map the closest waypoint (e.g., 'A_tip') to its note ('A')
    predicted_notes.append(note_mapping.get(closest_waypoint_name, 'Unknown'))

df['predicted_note'] = predicted_notes

# --- 5. Optional: Post-processing for smoother predictions ---
# This is a simple smoothing for demonstration. More advanced methods may be needed.
# For example, using a rolling window to pick the most frequent note.
window_size = 5 # Number of samples to consider for smoothing
df['predicted_note_smoothed'] = df['predicted_note'].rolling(window=window_size, min_periods=1).apply(lambda x: x.mode()[0], raw=False)

print(df[['robot timestamp', 'TCP_pose_x', 'TCP_pose_y', 'TCP_pose_z', 'predicted_note', 'predicted_note_smoothed']].head(10))
print(df[['robot timestamp', 'TCP_pose_x', 'TCP_pose_y', 'TCP_pose_z', 'predicted_note', 'predicted_note_smoothed']].tail(10))
print(df['predicted_note_smoothed'].value_counts())