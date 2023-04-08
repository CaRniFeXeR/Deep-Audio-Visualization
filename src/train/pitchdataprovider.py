import random
from typing import Tuple, Dict

import torch

from src.datastructures.trackfeatures import TrackFeatures


class PitchDataProvider:
    """
    This class is responsible for providing the data for the pitch detection model.
    Samples different pitch shifted versions of the same audio file and provides them to the model.
    """

    def __init__(self, tf_dict: Dict[str,TrackFeatures], batch_size: int, pitch_tuple_size: int) -> None:
        self.tf_dict = tf_dict
        self.bs = batch_size
        self.pitch_tuple_size = pitch_tuple_size
        self._generate_idxs()
        self.prepare_iter()
    
    def __iter__(self):
        self.prepare_iter()
        return self

    def __next__(self):
        return self.get_next_batch()
    
    def _generate_idxs(self):
        self.S_mag_dict = {}
        for key, tf in self.tf_dict.items():
            S_mag_norm = tf.get_normalized_magnitudes()
            self.S_mag_dict[key] = S_mag_norm
    
        self.data_idxes = list(range(0, S_mag_norm.shape[1] - tf.frame_width))
        self.pitch_levels = list(self.tf_dict.keys())
    
    def _get_window_tensor(self, w_start: int, pitch_level : str) -> torch.Tensor:
        tf = self.tf_dict[pitch_level]
        S_mag_norm = self.S_mag_dict[pitch_level]

        w_end = w_start + tf.frame_width
        if w_end <= S_mag_norm.shape[1]:
            return torch.from_numpy(S_mag_norm[:, w_start:w_end]).to(device="cuda")
        else:
            None
    
    def prepare_iter(self):
        self.current_iter_idxes = self.data_idxes.copy()
        random.shuffle(self.current_iter_idxes)
       
    
    def get_next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (batch_size, pair, frame_width)
        """
        # TODO consider triple for later

        batch_tensors_input = []
        batch_tensors_pitch_level = []
        for i in range(self.bs):

            if len(self.current_iter_idxes) < 1:
                raise StopIteration()
            w_start = self.current_iter_idxes.pop()
            tensor_pitches = []
            tensor_pitch_levels = []
            for pitch_level in random.sample(self.pitch_levels, k= self.pitch_tuple_size):
                # samples k times the same window for different pitch levels
                input_tensor = self._get_window_tensor(w_start, pitch_level)
                if input_tensor is not None:
                    tensor_pitches.append(input_tensor)
                    tensor_pitch_levels.append(torch.tensor(float(pitch_level), device="cuda"))
            
            if len(tensor_pitches) > 0:
                batch_tensors_input.append(torch.stack(tensor_pitches))
                batch_tensors_pitch_level.append(torch.stack(tensor_pitch_levels))


        batch_tensor_input = torch.stack(batch_tensors_input)
        batch_tensors_pitch_level = torch.stack(batch_tensors_pitch_level)
        return batch_tensor_input, batch_tensors_pitch_level