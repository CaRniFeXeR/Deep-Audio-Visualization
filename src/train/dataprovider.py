import random
import torch
from ..datastructures.trackfeatures import TrackFeatures


class DataProvider:

    def __init__(self, tf : TrackFeatures, batch_size : int) -> None:
        self.tf = tf
        self.bs = batch_size
        self._generate_idxs()
        self.prepare_iter()
    
    def _generate_idxs(self):
        self.S_mag_norm = self.tf.get_normalized_magnitudes()
        self.data_idxes = list(range(0, self.S_mag_norm.shape[1] - self.tf.img_width))
    
    def prepare_iter(self):
        self.current_iter_idxes = self.data_idxes.copy()
        random.shuffle(self.current_iter_idxes)

    def get_next_batch(self) -> torch.Tensor:
        batch_tensors = []
        for i in range(self.bs):

            if len(self.current_iter_idxes) < 1:
                raise StopIteration()
            w_start = self.current_iter_idxes.pop()
            w_end = w_start + self.tf.img_width
            if w_end <= self.S_mag_norm.shape[1]:
                    input_tensor = torch.from_numpy(self.S_mag_norm[:, w_start:w_end]).to(device="cuda")
                    batch_tensors.append(input_tensor)
        
        batch_tensor = torch.stack(batch_tensors)
        return batch_tensor

    def __iter__(self):
        self.prepare_iter()
        return self
    
    def __next__(self):
        return self.get_next_batch()




