from pathlib import Path

import numpy as np

from src.datastructures.trackfeatures import TrackFeatures


class TrackFeaturesFileHandler:

    def save_track_features(self, file_location: Path, tf: TrackFeatures):
        np.savez(file_location, F=tf.F, T=tf.T, S_mag=tf.S_mag, dt=tf.dt, frame_width=tf.frame_width, frame_height=tf.frame_height, time_resolution=tf.time_resolution, centroids=tf.centroids)

    def load_track_features(self, file_location: Path) -> TrackFeatures:
        loaded_dict = np.load(file_location)
        return TrackFeatures(**loaded_dict)
