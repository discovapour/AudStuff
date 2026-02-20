import os
import numpy as np
import librosa

folder = "C:/Users/PCUSER/Videos/Audiio/TraCKS"

print("Scanning folder:", folder)
print("-" * 50)

file_count = 0
bpms = []

for root, _, files in os.walk(folder):
    for f in files:
        if f.lower().endswith((".mp3", ".wav", ".flac", ".m4a")):
            file_count += 1
            path = os.path.join(root, f)

            print(f"\nProcessing: {f}")
            print("Path:", path)

            try:
                # Load audio
                y, sr = librosa.load(path, mono=True)
                print("Loaded OK")
                print("Sample rate:", sr)
                print("Duration (seconds):", round(len(y)/sr, 2))

                if len(y) < sr * 5:
                    print("⚠ Skipping (too short for beat tracking)")
                    continue

                # Use stronger beat estimation
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

                print("Raw tempo:", tempo)

                if tempo < 40:
                    print("⚠ Tempo too low — likely detection failure")
                    continue

                # Half-time correction
                if tempo < 90:
                    tempo *= 2
                    print("Adjusted tempo (x2):", tempo)

                bpms.append(float(tempo))

            except Exception as e:
                print("ERROR loading file:", e)

print("\nTotal audio files found:", file_count)
print("Valid BPM values:", bpms)

if bpms:
    print("Average BPM:", round(np.mean(bpms), 2))
else:

    print("No BPM detected.")
