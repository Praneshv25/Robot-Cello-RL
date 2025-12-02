import rtde_control
import rtde_receive
import time
import sounddevice as sd
import numpy as np
import queue
import threading

# --- Robot Setup ---
ROBOT_HOST = '10.165.11.242'  # Use 'localhost' for simulation
rtde_c = None
rtde_r = None

# --- Audio Setup ---
SAMPLERATE = 44100
BUFFER_SIZE = 4096  # Larger buffer = less frequent callbacks

# Note frequencies (in Hz) for octaves 3-8
# Each octave maps to a joint (octave 3 → joint 0, octave 4 → joint 1, ..., octave 8 → joint 5)
NOTE_FREQUENCIES = {
    # Octave 3 -> Joint 0
    'C3': 130.81,
    'D3': 146.83,
    'G3': 196.00,
    'A3': 220.00,
    
    # Octave 4 -> Joint 1
    'C4': 261.63,
    'D4': 293.66,
    'G4': 392.00,
    'A4': 440.00,
    
    # Octave 5 -> Joint 2
    'C5': 523.25,
    'D5': 587.33,
    'G5': 783.99,
    'A5': 880.00,
    
    # Octave 6 -> Joint 3
    'C6': 1046.50,
    'D6': 1174.66,
    'G6': 1567.98,
    'A6': 1760.00,
    
    # Octave 7 -> Joint 4
    'C7': 2093.00,
    'D7': 2349.32,
    'G7': 3135.96,
    'A7': 3520.00,
    
    # Octave 8 -> Joint 5
    'C8': 4186.01,
    'D8': 4698.63,
    'G8': 6271.93,
    'A8': 7040.00,
}

FREQUENCY_TOLERANCE = 30.0  # Hz tolerance for note detection

# Movement parameters for joints
# C: 45 degrees counter-clockwise (negative)
# G: 45 degrees clockwise (positive)
# D: 15 degrees counter-clockwise (negative)
# A: 15 degrees clockwise (positive)
JOINT_MOVEMENTS = {
    'C': -np.radians(45),  # 45 degrees counter-clockwise
    'G': np.radians(45),   # 45 degrees clockwise
    'D': -np.radians(15),  # 15 degrees counter-clockwise
    'A': np.radians(15),   # 15 degrees clockwise
}

# Map octaves to joints
OCTAVE_TO_JOINT = {
    3: 0,  # Octave 3 -> Joint 0 (base)
    4: 1,  # Octave 4 -> Joint 1 (shoulder)
    5: 2,  # Octave 5 -> Joint 2 (elbow)
    6: 3,  # Octave 6 -> Joint 3 (wrist 1)
    7: 4,  # Octave 7 -> Joint 4 (wrist 2)
    8: 5,  # Octave 8 -> Joint 5 (wrist 3)
}

MOVE_SPEED = 0.5      # Speed for joint movement (rad/s)
MOVE_ACCELERATION = 1.0  # Acceleration (rad/s^2)

# Track the current movement state
current_note = None

# Queue for audio processing (small queue to keep processing current)
audio_queue = queue.Queue(maxsize=2)

# Threading flag
processing_active = True

def detect_pitch_fft(audio_data):
    """Detect pitch using FFT (Fast Fourier Transform) - fast and cross-platform."""
    # Apply window to reduce spectral leakage
    windowed = audio_data * np.hanning(len(audio_data))
    
    # Compute FFT
    fft = np.fft.rfft(windowed)
    magnitude = np.abs(fft)
    
    # Find the peak frequency
    peak_index = np.argmax(magnitude)
    
    # Convert index to frequency
    frequency = peak_index * SAMPLERATE / len(audio_data)
    
    # Return both frequency and magnitude for debugging
    return frequency, magnitude[peak_index]

def detect_note(pitch):
    """Detects which note is being played based on pitch."""
    if pitch <= 0:
        return None
    
    for note, freq in NOTE_FREQUENCIES.items():
        if abs(pitch - freq) < FREQUENCY_TOLERANCE:
            return note
    return None

def audio_callback(indata, frames, time, status):
    """This function is called for each audio block - just queues data, returns immediately."""
    if status:
        print(status)
    
    # Just queue the data without processing - this returns immediately
    # Drop frames if queue is full (keeps processing current data)
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass  # Skip this frame if queue is full

def audio_processing_thread():
    """Separate thread that processes audio without blocking RTDE control."""
    global current_note, processing_active
    
    frame_count = 0
    last_command_time = 0
    COMMAND_DELAY = 0.3  # Minimum 300ms between robot commands
    PROCESS_EVERY_N = 3  # Only process every 3rd frame to reduce load
    
    while processing_active:
        try:
            # Get audio data from queue with timeout
            indata = audio_queue.get(timeout=0.1)
            
            frame_count += 1
            
            # Skip some frames to reduce processing load
            if frame_count % PROCESS_EVERY_N != 0:
                continue
            
            # Extract audio samples
            samples = indata[:, 0]
            
            # Detect pitch using FFT
            detected_pitch, magnitude = detect_pitch_fft(samples)
            
            # Determine if we have a valid note
            # Expanded range to cover octaves 3-8
            if 100 < detected_pitch < 8000:
                detected_note = detect_note(detected_pitch)
            else:
                detected_note = None
            
            # Update robot movement if note changed AND enough time has passed
            current_time = time.time()
            if detected_note != current_note and (current_time - last_command_time) >= COMMAND_DELAY:
                # Print only on changes
                if detected_note:
                    # Extract note name and octave
                    note_name = detected_note[0]  # C, D, G, or A
                    octave = int(detected_note[1])  # 3-8
                    joint_num = OCTAVE_TO_JOINT[octave]
                    movement = np.degrees(JOINT_MOVEMENTS[note_name])
                    print(f"🎵 Note {detected_note} ({detected_pitch:.1f} Hz) → Joint {joint_num} moving {movement:.1f}°")
                else:
                    print(f"🔇 No note detected")
                
                current_note = detected_note
                move_robot(detected_note)
                last_command_time = current_time
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in audio processing: {e}")

def move_robot(note):
    """Moves the robot joint based on the detected note."""
    if not rtde_c or not rtde_c.isConnected():
        return

    if note is None:
        return

    try:
        # Parse the note to get note name and octave
        note_name = note[0]  # C, D, G, or A
        octave = int(note[1])  # 3-8
        
        # Get the joint to move and the movement amount
        joint_index = OCTAVE_TO_JOINT[octave]
        movement = JOINT_MOVEMENTS[note_name]
        
        # Get current joint positions
        current_joints = rtde_r.getActualQ()
        target_joints = list(current_joints)
        
        # Update the target joint position
        target_joints[joint_index] += movement
        
        # Use moveJ for joint-based control (asynchronous)
        rtde_c.moveJ(target_joints, MOVE_SPEED, MOVE_ACCELERATION, asynchronous=True)
            
    except Exception as e:
        print(f"Robot error: {e}")


def main():
    global rtde_c, rtde_r, processing_active
    
    # Start the audio processing thread
    processing_thread = threading.Thread(target=audio_processing_thread, daemon=True)
    processing_thread.start()
    print("Audio processing thread started.")
    
    try:
        print("Connecting to robot...")
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)
        print("Connected to robot.")
        
        # Print joint mapping information
        print("\n=== Joint Mapping ===")
        for octave in range(3, 9):
            joint = OCTAVE_TO_JOINT[octave]
            print(f"Octave {octave} → Joint {joint}")
        
        print("\n=== Movement per Note ===")
        for note, movement in JOINT_MOVEMENTS.items():
            print(f"{note}: {np.degrees(movement):.1f}° ({'clockwise' if movement > 0 else 'counter-clockwise'})")
        print()

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
        # Stop processing thread
        processing_active = False
        print("Stopping audio processing thread...")
        processing_thread.join(timeout=2)
        
        # Disconnect from robot
        if rtde_c and rtde_c.isConnected():
            try:
                print("Stopping robot...")
                rtde_c.stopJ()  # Stop any joint movement
            except:
                pass
            rtde_c.disconnect()
            print("Disconnected from the robot.")

if __name__ == "__main__":
    main()

