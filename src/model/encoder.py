import torch

from src.datastructures.encoderconfig import EncoderConfig


class Encoder(torch.Module):

    def __init__(self, config : EncoderConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        layers.append(torch.nn.Conv1d(self.config.features_in_dim, 128,kernel_size= 3, stride=1))
        layers.append(torch.nn.BatchNorm1d(128))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool1d(2))
        layers.append(torch.nn.Conv1d(64, 256,kernel_size= 3, stride=1))
        layers.append(torch.nn.BatchNorm1d(256))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool1d(2))
        layers.append(torch.nn.Conv1d(128, 256,kernel_size= 3, stride=1))
        layers.append(torch.nn.BatchNorm1d(256))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool1d(2))
        layers.append(torch.nn.Conv1d(128, 128,kernel_size= 1, stride=1))
        layers.append(torch.nn.BatchNorm1d(128))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(1152,3))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, data : torch.Tensor) -> torch.Tensor:
        return self.model(data)