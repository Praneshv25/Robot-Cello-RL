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

# Note frequencies (in Hz) with tolerance
NOTE_FREQUENCIES = {
    'C': 261.63,  # C4
    'D': 293.66,  # D4
    'G': 392.00,  # G4
    'A': 440.00,  # A4
}

FREQUENCY_TOLERANCE = 20.0  # Hz tolerance for note detection

# Movement parameters
MOVE_DISTANCE = 0.02  # 2 cm per note detection
MOVE_SPEED = 0.1  # Speed for moveL (m/s)
MOVE_ACCELERATION = 0.5  # Acceleration (m/s^2)

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
    
    frame_count = 0
    last_command_time = 0
    COMMAND_DELAY = 0.3  # Minimum 300ms between robot commands for moveL
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
            
            # Determine if we have a valid note (simplified - no extra calculations)
            if 200 < detected_pitch < 600:
                detected_note = detect_note(detected_pitch)
            else:
                detected_note = None
            
            # Update robot movement if note changed AND enough time has passed
            current_time = time.time()
            if detected_note != current_note and (current_time - last_command_time) >= COMMAND_DELAY:
                # Print only on changes
                if detected_note:
                    print(f"🎵 Note {detected_note} ({detected_pitch:.1f} Hz)")
                else:
                    print(f"🔇 Stopped")
                
                current_note = detected_note
                move_robot(detected_note)
                last_command_time = current_time
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in audio processing: {e}")

def move_robot(note):
    """Moves the robot based on the detected note using position-based control."""
    if not rtde_c or not rtde_c.isConnected():
        return

    try:
        # Get current position
        current_pose = rtde_r.getActualTCPPose()
        target_pose = current_pose[:]
        
        if note == 'A':
            target_pose[2] += MOVE_DISTANCE  # Move up (positive Z)
        elif note == 'D':
            target_pose[1] += MOVE_DISTANCE  # Move left (positive Y)
        elif note == 'G':
            target_pose[1] -= MOVE_DISTANCE  # Move right (negative Y)
        elif note == 'C':
            target_pose[2] -= MOVE_DISTANCE  # Move down (negative Z)
        else:
            # No note detected, no movement needed
            return
        
        # Use moveL for position-based control (asynchronous)
        rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCELERATION, asynchronous=True)
            
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
                rtde_c.stopL()  # Stop any linear movement
            except:
                pass
            rtde_c.disconnect()
            print("Disconnected from the robot.")

if __name__ == "__main__":
    main()
