from dataclasses import dataclass

import numpy as np

@dataclass
class TrackFeatures:
    F : np.ndarray
    T : np.ndarray
    S_mag : np.ndarray
    dt : float
    frame_height : int
    frame_width : int
    time_resolution : float
    centroids : np.ndarray = None

    def get_normalized_magnitudes(self) -> np.ndarray:
        mag_min = np.min(self.S_mag)
        nag_max = np.max(self.S_mag)
        return (self.S_mag-mag_min)/(nag_max - mag_min)
