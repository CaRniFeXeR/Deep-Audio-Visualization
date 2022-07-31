from dataclasses import dataclass

import numpy as np

@dataclass
class TrackFeatures:
    F : np.ndarray
    T : np.ndarray
    S_mag : np.ndarray
    dt : float
    img_height : int
    img_width : int
    time_resolution : float