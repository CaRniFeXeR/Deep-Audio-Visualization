import numpy as np


def smooth_sequence(seq: np.ndarray, n_elements_smoothing: int) -> np.ndarray:
    smoothed = seq.copy()
    if n_elements_smoothing > 1:
        for i in range(n_elements_smoothing):
            current_seq = seq[i:-(n_elements_smoothing - i)]
            smoothed[:-n_elements_smoothing] += current_seq

        smoothed = smoothed / n_elements_smoothing
    return smoothed
