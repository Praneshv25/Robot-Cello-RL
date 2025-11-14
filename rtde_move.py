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
BUFFER_SIZE = 2048  # Buffer size for audio capture

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
    
    # Check if the signal is strong enough (simple volume threshold)
    if magnitude[peak_index] < 100:  # Adjust threshold as needed
        return 0.0
    
    # Only return frequencies in the musical range
    if 200 < frequency < 600:
        return frequency
    return 0.0

def detect_note(pitch):
    """Detects which note (A, D, G, C) is being played based on pitch."""
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
    
    while processing_active:
        try:
            # Get audio data from queue with timeout
            indata = audio_queue.get(timeout=0.1)
            
            # Extract audio samples
            samples = indata[:, 0]
            
            # Detect pitch using FFT
            detected_pitch = detect_pitch_fft(samples)
            
            # Detect which note is playing
            detected_note = detect_note(detected_pitch)
            
            if detected_note:
                print(f"Detected pitch: {detected_pitch:.2f} Hz -> Note: {detected_note}")
            
            # Update robot movement if note changed
            if detected_note != current_note:
                current_note = detected_note
                move_robot(detected_note)
                
        except queue.Empty:
            continue
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
