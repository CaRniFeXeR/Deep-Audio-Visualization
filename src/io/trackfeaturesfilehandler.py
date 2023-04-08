from pathlib import Path
from typing import Dict

import numpy as np
import re
from src.datastructures.trackfeatures import TrackFeatures


regex = r'[-+]?\d*\.\d+|\d+'

class TrackFeaturesFileHandler:

    def save_track_features(self, file_location: Path, tf: TrackFeatures):
        np.savez(file_location, F=tf.F, T=tf.T, S_mag=tf.S_mag, dt=tf.dt, frame_width=tf.frame_width, frame_height=tf.frame_height, time_resolution=tf.time_resolution, centroids=tf.centroids)

    def load_track_features(self, file_location: Path) -> TrackFeatures:
        loaded_dict = dict(np.load(file_location, allow_pickle=True))
        return TrackFeatures(**loaded_dict)

    def load_pitch_shifted_track_features(self, folder: Path) -> Dict[str,TrackFeatures]:
        """
        Loads all track features from a folder that have been pitch shifted.
        """
        result_dict = {}
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                for file in subfolder.iterdir():
                    if file.suffix == ".npz":
                        pitch_level = re.search(regex, subfolder.name).group()
                        result_dict[pitch_level] = self.load_track_features(file)
        
        return result_dict