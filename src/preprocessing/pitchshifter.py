from pathlib import Path
from typing import List
import soundfile as sf
import pyrubberband as pyrb
import librosa
from src.datastructures.pitchshiftconfig import PitchShiftConfig
import numpy as np

class PitchShifter:

    def __init__(self, config: PitchShiftConfig) -> None:
        self.config = config

    def range_pitch(self, wavefile: Path) -> List[Path]:
        """
        Shifts the pitch of a given wavefile in a given range and saves the results in a folder
        """
        y, sr = librosa.load(str(wavefile))
        result = []
        outfolder = self.config.outputlocation / wavefile.stem
        outfolder.mkdir(exist_ok=True, parents=True)
        for i in np.arange(self.config.min_pitch, self.config.max_pitch, self.config.step_size):
            n_steps = round(i,1)
            y_shifted_rubberband = self.shift_pitch(y, sr, n_steps=n_steps)
            output_file = Path(wavefile.stem + f'_ps{n_steps:.2f}.wav')
            sf.write(str(outfolder /output_file), y_shifted_rubberband, samplerate=sr)
            print("generated: " + str(output_file))
            result.append(output_file)

        return result

    def shift_pitch(self, y, sr, n_steps: float):

        y_shifted_rubberband = pyrb.pitch_shift(y, sr, n_steps=n_steps)

        return y_shifted_rubberband
