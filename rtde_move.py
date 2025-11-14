import rtde_control
import rtde_receive
import time
import sounddevice as sd
import numpy as np
import librosa

# --- Robot Setup ---
ROBOT_HOST = '10.165.11.242'  # Use 'localhost' for simulation
rtde_c = None
rtde_r = None

# --- Audio Setup ---
SAMPLERATE = 44100
BUFFER_SIZE = 2048

# Note frequencies (in Hz) with tolerance
NOTE_FREQUENCIES = {
    'C': 261.63,  # C4
    'D': 293.66,  # D4
    'G': 392.00,  # G4
    'A': 440.00,  # A4
}

FREQUENCY_TOLERANCE = 20.0  # Hz tolerance for note detection

# Movement speed (in m/s for velocity control)
MOVE_SPEED = 0.05  # 5 cm/s
ACCELERATION = 0.5  # m/s^2

# Track the current movement state
current_note = None

def detect_note(pitch):
    """Detects which note (A, D, G, C) is being played based on pitch."""
    if np.isnan(pitch) or pitch <= 0:
        return None
    
    for note, freq in NOTE_FREQUENCIES.items():
        if abs(pitch - freq) < FREQUENCY_TOLERANCE:
            return note
    return None

def audio_callback(indata, frames, time, status):
    """This function is called for each audio block."""
    global current_note
    
    if status:
        print(status)
    
    # Use librosa to detect pitch
    try:
        pitch, _, _ = librosa.pyin(
            y=indata[:, 0], 
            sr=SAMPLERATE,
            fmin=librosa.note_to_hz('C3'), 
            fmax=librosa.note_to_hz('C6')
        )
        
        # Get the average pitch from the buffer
        avg_pitch = np.nanmean(pitch)
        
        # Detect which note is playing
        detected_note = detect_note(avg_pitch)
        
        if detected_note:
            print(f"Detected pitch: {avg_pitch:.2f} Hz -> Note: {detected_note}")
        
        # Update robot movement if note changed
        if detected_note != current_note:
            current_note = detected_note
            move_robot(detected_note)
            
    except Exception as e:
        print(f"Error in audio processing: {e}")

def move_robot(note):
    """Moves the robot based on the detected note using velocity control."""
    if not rtde_c or not rtde_c.isConnected():
        print("Robot not connected.")
        return

    try:
        # Create velocity vector [vx, vy, vz, wx, wy, wz]
        # All velocities are in m/s for linear and rad/s for angular
        velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        if note == 'A':
            print("Note A detected! Moving UP.")
            velocity[2] = MOVE_SPEED  # Move up (positive Z)
        elif note == 'D':
            print("Note D detected! Moving LEFT.")
            velocity[1] = MOVE_SPEED  # Move left (positive Y)
        elif note == 'G':
            print("Note G detected! Moving RIGHT.")
            velocity[1] = -MOVE_SPEED  # Move right (negative Y)
        elif note == 'C':
            print("Note C detected! Moving DOWN.")
            velocity[2] = -MOVE_SPEED  # Move down (negative Z)
        else:
            print("No recognized note. Stopping.")
        
        # Use speedL for velocity-based control
        # speedL continues until a new command is given
        rtde_c.speedL(velocity, ACCELERATION)
            
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
            # This loop keeps the main thread alive while the sounddevice callback
            # runs in a background thread.
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nScript stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop the robot before disconnecting
        if rtde_c and rtde_c.isConnected():
            try:
                print("Stopping robot movement...")
                rtde_c.speedStop()  # Stop any velocity-based movement
            except:
                pass
            rtde_c.disconnect()
            print("Disconnected from the robot.")

if __name__ == "__main__":
    main()
