import torch

from src.datastructures.decoderconfig import DecoderConfig


class Decoder(torch.nn.Module):

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        assert self.config.latent_dim is not None
        self.lin_latent = torch.nn.Linear(self.config.latent_dim, 5 * 32)
        stride = 2
        layers = []
        layers.append(torch.nn.ConvTranspose1d(32, 64, kernel_size=3, stride=stride, padding=1, output_padding=0, dilation=2))
        layers.append(torch.nn.BatchNorm1d(64))
        layers.append(torch.nn.LeakyReLU())
        if self.config.output_length < 12:
            stride = 1
        layers.append(torch.nn.ConvTranspose1d(64, 128, kernel_size=3, stride=stride, padding=1, output_padding=stride - 1))
        layers.append(torch.nn.BatchNorm1d(128))
        layers.append(torch.nn.LeakyReLU())
        if self.config.output_length < 23:
            stride = 1
        layers.append(torch.nn.ConvTranspose1d(128, 128, kernel_size=3, stride=stride, padding=1, output_padding=stride - 1))
        layers.append(torch.nn.BatchNorm1d(128))
        layers.append(torch.nn.LeakyReLU())
        if self.config.output_length < 45:
            stride = 1
        layers.append(torch.nn.ConvTranspose1d(128, self.config.output_dim, kernel_size=3, stride=stride, padding=1, output_padding=stride - 1))
        layers.append(torch.nn.LeakyReLU())
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, embedded_points: torch.Tensor) -> torch.Tensor:
        batch_size = embedded_points.shape[0]
        data = self.lin_latent(embedded_points).view((batch_size, 32, 5))
        for layer in self.layers:
            # print(layer)
            # print(data.shape)
            data = layer(data)

        return data
