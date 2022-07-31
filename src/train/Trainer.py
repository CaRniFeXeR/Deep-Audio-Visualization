from pathlib import Path
import numpy as np

import torch
from src.io.trackfeaturesfilehandler import TrackFeaturesFileHandler
from ..datastructures.trainconfig import TrainConfig
from ..model.audiomodel import AudioModel


class Trainer:

    def __init__(self, config : TrainConfig) -> None:
        self.config = config
        self.tf = TrackFeaturesFileHandler().load_track_features(self.config.track_features_location / Path("vars.npz"))
        self.config.modelconfig.encoderconfig.features_in_dim = self.tf.img_height
        self.config.modelconfig.decoderconfig.output_width = self.tf.img_height
        self.model = AudioModel(self.config.modelconfig)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.trainparams.learning_rate)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.loss_fn = torch.nn.MSELoss()

    def train(self):
        
        self.model.to(device = "cuda")
        def normalize0_1(A):
            return (A-np.min(A))/(np.max(A) - np.min(A))

        S_mag = normalize0_1(self.tf.S_mag) #e.g. shape = (94, 1757)
        

        for e in range(self.config.trainparams.n_epochs):
            print(f"training in epoch '{e}'")
            for window_idx in range(S_mag.shape[1] // self.tf.img_width):
                self.optimizer.zero_grad()
                w_start = window_idx*self.tf.img_width
                w_end = (window_idx+1)*self.tf.img_width
                input_tensor = torch.from_numpy(S_mag[:,w_start:w_end]).to(device="cuda").unsqueeze(dim = 0)
                predicted = self.model(input_tensor)
                loss = self.loss_fn(predicted, input_tensor)
                print(f" loss {loss.detach().cpu().numpy()}")
                loss.backward()
                self.optimizer.step()
        
        
                