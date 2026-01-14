import numpy as np
import soundfile as sf

# Configuration consistent with EnCodec (24kHz)
sr = 24000
duration = 480.0

# Generate white noise (random values from -1 to 1)
noise = np.random.uniform(-1, 1, int(sr * duration))
sf.write("noise.wav", noise, sr)

print("The file noise.wav has been generated!")
