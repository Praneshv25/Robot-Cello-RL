import rtde_control
import rtde_receive
import time
import sounddevice as sd
import numpy as np
import librosa

# --- Robot Setup ---
ROBOT_HOST = 'localhost'  # Use 'localhost' for simulation
rtde_c = None
rtde_r = None

# --- Audio Setup ---
SAMPLERATE = 44100
BUFFER_SIZE = 2048

# Pitch thresholds (in Hz)
LOW_PITCH_THRESHOLD = 200.0
HIGH_PITCH_THRESHOLD = 600.0

# Movement distance (in meters)
MOVE_DISTANCE = 0.05  # 5 cm (approx 2 inches)

def audio_callback(indata, frames, time, status):
    """This function is called for each audio block."""
    if status:
        print(status)
    
    # Use librosa to detect pitch
    pitch, _, _ = librosa.pyin(y=indata[:, 0], fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    
    # Get the average pitch from the buffer
    avg_pitch = np.nanmean(pitch)

    if not np.isnan(avg_pitch) and avg_pitch > 0:
        print(f"Detected pitch: {avg_pitch:.2f} Hz")
        move_robot(avg_pitch)

def move_robot(pitch):
    """Moves the robot based on the detected pitch."""
    if not rtde_c or not rtde_c.isConnected():
        print("Robot not connected.")
        return

    try:
        current_pose = rtde_r.getActualTCPPose()
        target_pose = current_pose[:]

        if pitch > HIGH_PITCH_THRESHOLD:
            print("High pitch detected! Moving right.")
            target_pose[1] -= MOVE_DISTANCE  # Move right (adjust axis if needed)
            rtde_c.moveL(target_pose, 0.25, 0.5)
        elif pitch < LOW_PITCH_THRESHOLD:
            print("Low pitch detected! Moving left.")
            target_pose[1] += MOVE_DISTANCE  # Move left (adjust axis if needed)
            rtde_c.moveL(target_pose, 0.25, 0.5)
            
    except Exception as e:
        print(f"Error moving robot: {e}")


def main():
    global rtde_c, rtde_r
    try:
        print("Connecting to robot...")
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)
        print("Connected to robot.")

        # Start listening to the microphone
        print("Listening for audio...")
        with sd.InputStream(callback=audio_callback, device=None, channels=1, samplerate=SAMPLERATE, blocksize=BUFFER_SIZE):
            print("Script is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nScript stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rtde_c and rtde_c.isConnected():
            rtde_c.disconnect()
            print("Disconnected from the robot.")

if __name__ == "__main__":
    main()
