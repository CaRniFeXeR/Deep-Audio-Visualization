import torch

from src.datastructures.encoderconfig import EncoderConfig


class Encoder(torch.nn.Module):

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        layers.append(torch.nn.Conv1d(self.config.features_in_dim, 64, kernel_size=3, stride=1))
        layers.append(torch.nn.BatchNorm1d(64))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool1d(2))
        layers.append(torch.nn.Conv1d(64, 128, kernel_size=3, stride=1))
        layers.append(torch.nn.BatchNorm1d(128))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool1d(2))
        layers.append(torch.nn.Conv1d(128, 256, kernel_size=3, stride=1))
        layers.append(torch.nn.BatchNorm1d(256))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool1d(2))
        layers.append(torch.nn.Conv1d(256, 128, kernel_size=1, stride=1))
        layers.append(torch.nn.BatchNorm1d(128))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(1152, 3))

        # self.model = torch.nn.Sequential(*layers)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            data = layer(data)
        return data #self.model(data)
