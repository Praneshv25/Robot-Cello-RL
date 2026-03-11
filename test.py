import time
import numpy as np

def play_note_with_rl(rtde_c, string_poses, note_duration, direction, rl_model):
    t = 0.0
    start_pose = rtde_c.getActualTCPPose()
    
    tip = np.array(string_poses['tip_p'])
    frog = np.array(string_poses['frog_p'])
    vector = tip - frog
    full_dist = np.linalg.norm(vector)
    unit_vector = vector / full_dist
    
    while t < note_duration:
        start_loop = time.time()
        
        scale = get_ease_scale(t, note_duration)
        linear_dist = scale * full_dist
        
        if direction == "down":
             target_xyz = frog[:3] + (unit_vector[:3] * linear_dist)
        else:
             target_xyz = tip[:3] - (unit_vector[:3] * linear_dist)
             
        baseline_cmd = list(target_xyz) + list(frog[3:]) 

        obs = get_observation()
        pressure_offset = rl_model.predict(obs) 
        
        baseline_cmd[2] += pressure_offset 

        rtde_c.servoL(baseline_cmd, time=0.002, lookahead_time=0.1, gain=300)
        
        t += 0.002
