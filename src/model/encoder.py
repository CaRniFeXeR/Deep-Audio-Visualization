import math
import torch

from src.datastructures.encoderconfig import EncoderConfig


class Encoder(torch.nn.Module):

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        n_pooling = 0
        layers.append(torch.nn.Conv1d(self.config.features_in_dim, 64, kernel_size=3, stride=1, padding=1)) #-2
        layers.append(torch.nn.BatchNorm1d(64)) 
        layers.append(torch.nn.LeakyReLU())
        if self.config.frame_width_in > 10:
            layers.append(torch.nn.MaxPool1d(2)) #/2
            n_pooling+=1
        layers.append(torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)) #-2
        layers.append(torch.nn.BatchNorm1d(128))
        layers.append(torch.nn.LeakyReLU())
        if self.config.frame_width_in > 21:
            layers.append(torch.nn.MaxPool1d(2))
            n_pooling+=1
        layers.append(torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)) # -2
        layers.append(torch.nn.BatchNorm1d(256))
        layers.append(torch.nn.LeakyReLU())
        if self.config.frame_width_in > 43:
            layers.append(torch.nn.MaxPool1d(2))
            n_pooling+=1
        layers.append(torch.nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)) # -2
        layers.append(torch.nn.BatchNorm1d(256))
        layers.append(torch.nn.LeakyReLU())
        if self.config.frame_width_in > 87:
            layers.append(torch.nn.MaxPool1d(2))
            n_pooling+=1
        layers.append(torch.nn.Conv1d(256, 128, kernel_size=1, stride=1))
        layers.append(torch.nn.BatchNorm1d(128))
        layers.append(torch.nn.LeakyReLU())
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(int(math.floor(self.config.frame_width_in  * 0.5**n_pooling)) * 128, self.config.latent_dim))
        layers.append(self.config.final_activation_fn)

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            # print(data.shape)
            # print(layer)
            data = layer(data)
        return data
