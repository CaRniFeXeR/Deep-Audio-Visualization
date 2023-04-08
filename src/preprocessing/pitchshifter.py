from pathlib import Path
from typing import List
import pyrubberband as pyrb
import librosa
from src.datastructures.pitchshiftconfig import PitchShiftConfig
import numpy as np
from scipy.io import wavfile
class PitchShifter:

    def __init__(self, config: PitchShiftConfig) -> None:
        self.config = config

    def range_pitch(self, wavefile: Path) -> List[Path]:
        """
        Shifts the pitch of a given wavefile in a given range and saves the results in a folder
        """
        y, sr = librosa.load(str(wavefile), sr=44100)
        l_y = len(y)
        # remove the first and last 10% of the signal to avoid artifacts
        y = y[int(round(l_y*self.config.percent_offsets[0])):-int(round(l_y*self.config.percent_offsets[1]))]

        result = []
        outfolder = self.config.outputlocation / wavefile.stem
        outfolder.mkdir(exist_ok=True, parents=True)
        for i in np.arange(self.config.min_pitch, self.config.max_pitch, self.config.step_size):
            n_steps = round(i,1)
            y_shifted_rubberband = self.shift_pitch(y, sr, n_steps=n_steps)
            output_file = Path(wavefile.stem + f'_ps{n_steps:.2f}.wav')
            # sf.write(str(outfolder /output_file), y_shifted_rubberband, samplerate=sr)
            wavfile.write(str(outfolder /output_file), sr, y_shifted_rubberband.astype(y.dtype))
            print("generated: " + str(output_file))
            result.append(outfolder /output_file)

        return result

    def shift_pitch(self, y, sr, n_steps: float):

        y_shifted_rubberband = pyrb.pitch_shift(y, sr, n_steps=n_steps)

        return y_shifted_rubberband
