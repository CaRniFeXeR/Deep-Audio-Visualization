import random
import torch
from ..datastructures.trackfeatures import TrackFeatures


class DataProvider:

    def __init__(self, tf: TrackFeatures, batch_size: int, prediction_length: int) -> None:
        self.tf = tf
        self.bs = batch_size
        self.l = prediction_length
        self._generate_idxs()
        self.prepare_iter()

    def _generate_idxs(self):
        self.S_mag_norm = self.tf.get_normalized_magnitudes()
        self.data_idxes = list(range(0, self.S_mag_norm.shape[1] - self.tf.img_width))

    def _get_window_tensor(self, w_start: int) -> torch.Tensor:
        w_end = w_start + self.tf.img_width
        if w_end <= self.S_mag_norm.shape[1]:
            return torch.from_numpy(self.S_mag_norm[:, w_start:w_end]).to(device="cuda")
        else:
            None

    def prepare_iter(self):
        self.current_iter_idxes = self.data_idxes.copy()
        random.shuffle(self.current_iter_idxes)
        self.prediction_idxes = self.data_idxes.copy()
        random.shuffle(self.prediction_idxes)

    def get_next_batch(self) -> torch.Tensor:
        batch_tensors = []
        for i in range(self.bs):

            if len(self.current_iter_idxes) < 1:
                raise StopIteration()
            w_start = self.current_iter_idxes.pop()
            input_tensor = self._get_window_tensor(w_start)
            if input_tensor is not None:
                batch_tensors.append(input_tensor)

        batch_tensor = torch.stack(batch_tensors)
        return batch_tensor

    def get_next_prediction_seq(self) -> torch.Tensor:
        
        batch_tensors = []
        for i in range(8):
            if len(self.prediction_idxes) < 1:
                return None

            current_idx = self.prediction_idxes.pop()

            if current_idx < self.l or current_idx - self.l + self.tf.img_width > self.S_mag_norm.shape[1]:
                return None #return none if l windows are out of img bounds

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
