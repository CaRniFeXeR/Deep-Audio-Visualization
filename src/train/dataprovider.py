import random
from typing import Tuple
import torch
from ..datastructures.trackfeatures import TrackFeatures


class DataProvider:

    def __init__(self, tf: TrackFeatures, batch_size: int, prediction_length: int, batch_size_seq: int = 8) -> None:
        self.tf = tf
        self.bs = batch_size
        self.bs_seq = batch_size_seq
        self.l = prediction_length
        self._generate_idxs()
        self.prepare_iter()

    def _generate_idxs(self):
        self.S_mag_norm = self.tf.get_normalized_magnitudes()
        self.data_idxes = list(range(0, self.S_mag_norm.shape[1] - self.tf.frame_width))

    def _get_window_tensor(self, w_start: int) -> torch.Tensor:
        w_end = w_start + self.tf.frame_width
        if w_end <= self.S_mag_norm.shape[1]:
            return torch.from_numpy(self.S_mag_norm[:, w_start:w_end]).to(device="cuda")
        else:
            None

    def _get_avg_centroid(self, w_start: int) -> torch.Tensor:
        w_end = w_start + int(self.tf.frame_width / 2)
        if w_end <= self.tf.centroids.shape[-1]:
            return torch.mean(torch.from_numpy(self.tf.centroids[w_start:w_end])).to(device="cuda")
        else:
            None

    def prepare_iter(self):
        self.current_iter_idxes = self.data_idxes.copy()
        random.shuffle(self.current_iter_idxes)
        self.prediction_idxes = self.data_idxes.copy()
        random.shuffle(self.prediction_idxes)

    def get_next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_tensors_input = []
        batch_tensors_centroid = []
        for i in range(self.bs):

            if len(self.current_iter_idxes) < 1:
                raise StopIteration()
            w_start = self.current_iter_idxes.pop()
            input_tensor = self._get_window_tensor(w_start)
            if input_tensor is not None:
                batch_tensors_input.append(input_tensor)

            if self.tf.centroids is not None:
                centroid_tensor = self._get_avg_centroid(w_start)
                if centroid_tensor is not None:
                    batch_tensors_centroid.append(centroid_tensor)

        batch_tensor_input = torch.stack(batch_tensors_input)
        batch_tensor_centroid = None if self.tf.centroids is None else torch.stack(batch_tensors_centroid)
        return batch_tensor_input, batch_tensor_centroid

    def get_next_prediction_seq(self) -> torch.Tensor:
        """
        returns l-long sequences of the track data

        result shape (bs_seq,l,latent)
        """

        batch_tensors = []
        for i in range(self.bs_seq):
            if len(self.prediction_idxes) < 1:
                return None

            current_idx = self.prediction_idxes.pop()

            if current_idx < self.l or current_idx - self.l + self.tf.frame_width > self.S_mag_norm.shape[1]:
                continue  # skip current loop if l windows are out of img bounds

            sequence_tensors = []
            for i in range(current_idx - self.l, current_idx):
                current_window = self._get_window_tensor(i)
                sequence_tensors.append(current_window)

            sequence_tensor = torch.stack(sequence_tensors)
            batch_tensors.append(sequence_tensor)

        batch_tensor = torch.stack(batch_tensors)
        return batch_tensor

    def __iter__(self):
        self.prepare_iter()
        return self

    def __next__(self):
        return self.get_next_batch()
