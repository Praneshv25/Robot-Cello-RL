import rtde_control
import rtde_receive
import numpy as np
# from ur_rtde import rtde_control
# from ur_rtde import rtde_receive

# from rtde_control import RTDEControlInterface as RTDEControl
# from rtde_receive import RTDERecieveInterface as RTDEReceive
import time

ROBOT_IP = "10.165.11.242"
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
# rtde_c = RTDEControl(ROBOT_IP)
# rtde_r = RTDEReceive(ROBOT_IP)

try:

    #actual_q = rtde_r.getActualQ()
    current_pos = rtde_r.getActualTCPPose()
    print(current_pos)

    up_pose = list(current_pos)
    up_pose[2] += .10
    print("Moving up 10cm")

    #rtde_c.moveL(up_pose, speed = 0.1, acceleration = 0.2)

    tcp_pose = rtde_r.getActualTCPPose()
    tcp_pose[2] += .20
    j_pos = rtde_c.getInverseKinematics(tcp_pose)
    #rtde_c.servoJ(j_pos, time = 0.1, lookahead_time = 0.05, gain = 300.0)
    """
    for i in range(len(tcp_pose)):
        tcp_pose[i] += -.25

    print(tcp_pose)
        """
    

    #rtde_c.servoL(tcp_pose, .25 ,1.2 , 1, 0.1, 1000.0)

    rtde_c.servoJ(j_pos, time = 0.1, lookahead_time = 0.05, gain = 300.0)

    print("movement complete")

except Exception as e:
    print(f"An error has occured: {e}")
finally:
    rtde_c.stopScript()

