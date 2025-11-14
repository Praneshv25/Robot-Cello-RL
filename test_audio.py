import sounddevice as sd
import numpy as np
import queue
import threading
import time

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

FREQUENCY_TOLERANCE = 20.0

# Queue for audio processing
audio_queue = queue.Queue(maxsize=2)
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

def audio_callback(indata, frames, time_info, status):
    """This function is called for each audio block - just queues data."""
    if status:
        print(status)
    
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass

def audio_processing_thread():
    """Separate thread that processes audio."""
    global processing_active
    
    print("\n" + "="*60)
    print("AUDIO TEST - Listening for notes A, D, G, C")
    print("="*60)
    print("\nNote mapping:")
    print("  A (440 Hz) → UP")
    print("  D (294 Hz) → LEFT")
    print("  G (392 Hz) → RIGHT")
    print("  C (262 Hz) → DOWN")
    print("\nPlay or sing these notes to test detection...")
    print("="*60 + "\n")
    
    last_note = None
    frame_count = 0
    
    while processing_active:
        try:
            indata = audio_queue.get(timeout=0.1)
            samples = indata[:, 0]
            frame_count += 1
            
            # Detect pitch
            detected_pitch, magnitude = detect_pitch_fft(samples)
            
            # Calculate RMS volume for reference
            rms = np.sqrt(np.mean(samples**2))
            
            # Always print what we're detecting (with volume info)
            note_str = ""
            if 200 < detected_pitch < 600:
                detected_note = detect_note(detected_pitch)
                if detected_note:
                    note_str = f"→ Note {detected_note} ✓"
                else:
                    note_str = f"(no match)"
            else:
                detected_note = None
                note_str = "(out of range)"
            
            # Print every frame so we see it's working
            print(f"[{frame_count:4d}] Freq: {detected_pitch:6.1f} Hz  |  Mag: {magnitude:8.1f}  |  Vol: {rms:5.3f}  |  {note_str}")
            
            # Print direction changes
            if detected_note != last_note:
                if detected_note:
                    direction = {
                        'A': '↑ UP',
                        'D': '← LEFT', 
                        'G': '→ RIGHT',
                        'C': '↓ DOWN'
                    }[detected_note]
                    print(f"\n>>> 🎵 ROBOT ACTION: Moving {direction}\n")
                elif last_note is not None:
                    print(f"\n>>> 🔇 ROBOT ACTION: STOPPED\n")
                
                last_note = detected_note
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error: {e}")

def main():
    global processing_active
    
    # Check available audio devices
    print("Available audio devices:")
    print(sd.query_devices())
    print("\n")
    
    # Start processing thread
    processing_thread = threading.Thread(target=audio_processing_thread, daemon=True)
    processing_thread.start()
    
    try:
        # Start audio stream
        with sd.InputStream(
            callback=audio_callback, 
            device=None,  # Use default device
            channels=1, 
            samplerate=SAMPLERATE, 
            blocksize=BUFFER_SIZE
        ):
            print("Press Ctrl+C to stop...\n")
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n\nTest stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTip: If you see a device error, check the device list above")
        print("     and specify a device number in sd.InputStream(device=X)")
    finally:
        processing_active = False
        processing_thread.join(timeout=2)

if __name__ == "__main__":
    main()

