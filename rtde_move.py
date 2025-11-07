import rtde_control
import rtde_receive
import time

# Setup robot IP
ROBOT_HOST = 'localhost' # Using localhost for simulation, replace with actual robot IP if needed

try:
    # Initialize control and receive interfaces
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

    # Get current joint angles
    initial_q = rtde_r.getActualQ()
    print(f"Initial joint angles: {initial_q}")

    # Define a new target joint position (example: small move on the first joint)
    target_q = initial_q[:]
    target_q[0] += 0.1  # Small increment to the first joint

    print(f"Target joint angles: {target_q}")

    # Parameters for moveJ
    velocity = 0.5
    acceleration = 1.5
    blend = 0.0

    # Move to the target position
    print("Moving to target...")
    rtde_c.moveJ(target_q, velocity, acceleration)
    print("Move complete.")

    # Wait a moment to observe the new position
    time.sleep(2)

    # Get final joint angles to verify the move
    final_q = rtde_r.getActualQ()
    print(f"Final joint angles: {final_q}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Disconnect
    if 'rtde_c' in locals() and rtde_c.isConnected():
        rtde_c.disconnect()
        print("Disconnected from the robot.")
