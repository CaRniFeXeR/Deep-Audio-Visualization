import numpy as np
from librosa.feature import melspectrogram


def convert_spectogram_to_mel(Sxx: np.ndarray, given_power_spec: bool = False):

    # if not given_power_spec:
    #     Sxx = Sxx ** 2

    mel_Sxx = melspectrogram(S=Sxx)
    np.abs(mel_Sxx)
    return mel_Sxx
