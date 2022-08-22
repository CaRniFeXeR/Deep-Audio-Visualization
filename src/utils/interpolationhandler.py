from typing import Tuple
import numpy as np
from scipy import interpolate


def smooth_curve(orginal_points: np.ndarray, s = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    interpolates a given array of Nd points using splines
    """
    tck, u = interpolate.splprep(orginal_points, s=s)
    interpolated = interpolate.splev(np.linspace(0, 1, len(orginal_points)*2), tck)
    return interpolated
