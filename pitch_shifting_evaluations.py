import librosa
import numpy as np
import soundfile as sf
import pyrubberband as pyrb

# Load audio file
y, sr = librosa.load("./data/tracks/Kodaline-Brother.wav")

y = y[len(y)//10:len(y)//6]

# # Define the amount of pitch shift you want (in semitones)
# semitones = 1

# # Compute the constant-Q transform
# CQT = librosa.cqt(y, sr=sr)

# # Compute the phase shift for the desired pitch shift
# bin_shift = int(round(semitones * CQT.shape[1] / 12))
# phase_shift = 2 * np.pi * bin_shift * np.arange(CQT.shape[0]) / CQT.shape[0]
# phase_shift = np.exp(-1j * phase_shift)

# # Apply the phase shift to the CQT coefficients
# CQT_shifted = CQT * phase_shift[:, np.newaxis]

# # Convert the shifted CQT back to the time domain
# y_shifted = librosa.icqt(CQT_shifted, sr=sr)

# y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-0.5)

for i in range(1,20):
    step_size = 0.2*i
    y_shifted_rubberband = pyrb.pitch_shift(y, sr, n_steps=step_size)
    sf.write(f'example_audio_shifted_{step_size}.wav', y_shifted_rubberband, samplerate=sr)


# Write shifted audio to file
# sf.write('example_audio_shifted_b.wav', y_shifted, samplerate=sr)