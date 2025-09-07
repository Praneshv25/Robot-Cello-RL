import librosa, soundfile as sf
import os, random

# load audio
y, sr = librosa.load("Robot/twinkle-fast.mp3", sr=None)  # sr = sample rate
length = len(y)

os.makedirs("Robot-segment", exist_ok=True)

start = 0
index = 0
while start < length:
    chunk_len = random.randint(5, 10) * sr  # samples
    end = min(start + chunk_len, length)
    sf.write(f"Robot-segment/output4_{index}.wav", y[start:end], sr)  # saves as wav
    start = end
    index += 1

print("✅ Done! Files are in 'Robot-segment' folder.")
