import numpy as np
from scipy.spatial.transform import Rotation as R

def extract_z_axis_from_pose(pose):
    """
    Extracts the Z-axis (third column of rotation matrix) from an axis-angle pose.
    pose: [x, y, z, rx, ry, rz]
    Returns: unit vector for Z-axis
    """
    rotvec = np.array(pose[3:6])
    rotation = R.from_rotvec(rotvec)
    rot_matrix = rotation.as_matrix()  # 3x3
    z_axis = rot_matrix[:, 2]          # Z-axis = 3rd column
    return z_axis / np.linalg.norm(z_axis)

def average_normals(poses):
    """
    Takes a list of 6D poses and returns the average Z-axis normal.
    """
    normals = np.array([extract_z_axis_from_pose(p) for p in poses])
    mean_normal = np.mean(normals, axis=0)
    return mean_normal / np.linalg.norm(mean_normal)

# === Your cello string bow poses ===

a_tip = [.473129539189, .413197423330, .256308427905, -1.460522581833, -2.310115543652, 1.445803824327]
a_frog = [.300717266074, .793568239540, .099710283103, -1.543522183454, -2.354885618328, 1.346770272474]

d_tip = [.340413993945, .280157415162, .176342071758, -1.614553612482, -2.044810993523, 1.042279199535]
d_frog = [.302785064368, .749849181019, .117254426008, -1.664082298752, -2.084265434693, 1.037965163360]

g_tip = [.162016291992, .201320984957, .059414774157, -1.929772636560, -1.931323067217, .555055912517]
g_frog = [.281203376642, .681662588607, .104672526365, -1.812194031755, -1.940153681829, .493747597283]

c_tip = [.079815569355, .285182178102, -.086654726588, -1.819646014269, -1.658258006768, .180930717120]
c_frog = [.256662516098, .610082591416, .062624387196, -1.743236422252, -1.524514092756, .163823228357]

# === Combine all poses ===
poses = [a_tip, a_frog, d_tip, d_frog, g_tip, g_frog, c_tip, c_frog]

# === Compute average cello body normal ===
cello_normal = average_normals(poses)
print("Estimated cello body normal:", cello_normal)
