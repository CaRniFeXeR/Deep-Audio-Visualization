import torch
import torchaudio

from src.datastructures.decoderconfig import DecoderConfig


class Decoder(torch.Module):

    def __init__(self, config : DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.lin_latent = torch.nn.Linear(3, 352)
        #todo add reshape
        layers = []
        layers.append(torch.nn.ConvTranspose1d(32,64, kernel_size=3))
        layers.append(torch.nn.BatchNorm1d(64))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.ConvTranspose1d(64, 128, kernel_size=3))
        layers.append(torch.nn.BatchNorm1d(128))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.ConvTranspose1d(128, self.config.output_width, kernel_size=1))
        layers.append(torch.nn.ReLU())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, data : torch.Tensor) -> torch.Tensor:
        data = torch.relu(self.lin_latent(data)).view((11,32))
        return self.model(data)