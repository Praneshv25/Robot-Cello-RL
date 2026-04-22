import socket
import threading
import time
import logging
import sys

sys.path.append('/home/skamanski/Downloads/rtde-2.7.2-release/rtde-2.7.2') # Adjust this path if needed
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

# === CONFIG ===
ROBOT_IP = "192.168.0.119"  # <- Replace with IP of your URSim or robot
URSCRIPT_PORT = 30002
RTDE_PORT = 30004
CONFIG_XML = "control_loop_configuration.xml"  # XML file must be in same directory


# === URScript to send ===
urscript = """
def myProgram():
  textmsg("Hello from Python!")
  sleep(1.0)
  textmsg("Finished.")
end

myProgram()
"""


# === Function: Send URScript to robot ===
def send_urscript():
    print("📤 Sending URScript...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ROBOT_IP, URSCRIPT_PORT))
            s.sendall(urscript.encode("utf-8"))
        print("✅ URScript sent successfully.")
    except Exception as e:
        print(f"❌ Error sending URScript: {e}")


# === Function: Log RTDE data ===
def log_rtde_data():
    try:
        conf = rtde_config.ConfigFile(CONFIG_XML)
        # Specify the recipe name from your XML file that contains the desired outputs
        # Make sure your XML recipe includes: "timestamp", "output_int_register_0", "actual_TCP_pose", "actual_q", "actual_TCP_force"
        output_names, output_types = conf.get_recipe("state") # Adjust recipe name if needed

        con = rtde.RTDE(ROBOT_IP, RTDE_PORT)
        if not con.connect():
            print("❌ RTDE: TCP socket opened but protocol handshake failed.")
            print("   • Make sure Polyscope is in Remote Control mode.")
            print("   • Make sure a program is running (your rtde_control_loop.urp).")
            return
        print("✅ RTDE handshake completed.")

    except FileNotFoundError:
        print(f"❌ Error: RTDE Configuration file not found at {CONFIG_XML}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Error: Recipe 'state' (or your chosen name) not found or missing fields in {CONFIG_XML}. Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error setting up RTDE configuration: {e}")
        sys.exit(1)
    con.get_controller_version()
    con.send_output_setup(output_names, output_types)
    if not con.send_start():
        print("❌ Failed to start RTDE data exchange.")
        return

    print("📡 Logging RTDE data...\n")
    for i in range(50):  # ~5 seconds at 10Hz
        state = con.receive()
        if state is None:
            print("⚠️ No data received.")
            continue

        print(f"[{i}] TCP Pose: {state.actual_TCP_pose}, Joint Angles: {state.actual_q}")

        time.sleep(0.1)

    con.send_pause()
    con.disconnect()
    print("✅ RTDE logging complete.")

 


# === Main ===
if __name__ == "__main__":
    # Start RTDE logging in a thread
    rtde_thread = threading.Thread(target=log_rtde_data)
    rtde_thread.start()

    # Give RTDE a moment to start
    time.sleep(0.5)

    # Send URScript while RTDE is logging
    send_urscript()

    # Wait for RTDE to finish
    rtde_thread.join()
