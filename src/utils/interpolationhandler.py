from typing import List, Tuple
import numpy as np
from scipy import interpolate


def spine_interpolate(orginal_points: List[np.ndarray], s=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    interpolates a given array of Nd points using splines
    """
    tck, u = interpolate.splprep(orginal_points, s=s, k=3)
    # interpolated = interpolate.splev(np.linspace(0, 1, len(orginal_points[0])*1), tck)
    u_extended = []
    for i in range(len(u) - 1):
        u_extended.append(u[i])
        u_extended.append((u[i] + u[i+1]) / 2)
    interpolated = interpolate.splev(u_extended, tck)
    return interpolated