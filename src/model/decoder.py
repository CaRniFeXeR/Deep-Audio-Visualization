import torch
import torchaudio

from src.datastructures.decoderconfig import DecoderConfig


class Decoder(torch.nn.Module):

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.lin_latent = torch.nn.Linear(3, 352)
        # todo add reshape
        layers = []
        layers.append(torch.nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1))
        layers.append(torch.nn.BatchNorm1d(64))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1))
        layers.append(torch.nn.BatchNorm1d(128))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.ConvTranspose1d(128, self.config.output_width,kernel_size = 3, stride=2, padding=1, output_padding=1))
        layers.append(torch.nn.ReLU())
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, embedded_points: torch.Tensor) -> torch.Tensor:
        batch_size = embedded_points.shape[0]
        data = torch.relu(self.lin_latent(embedded_points)).view((batch_size, 32, 11))
        for layer in self.layers:
            # print(layer)
            # print(data.shape)
            data = layer(data)

        return data
