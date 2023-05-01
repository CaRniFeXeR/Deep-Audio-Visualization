

import math
import torch
import torch.nn as nn
from ..datastructures.encoderconfig import EncoderConfig
from .encoder import Encoder


class VarationalEncoder(Encoder):
    
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__(config)
        self.latent_prj_sigma = nn.Linear(int(math.floor(self.config.frame_width_in  * 0.5**self.n_pooling)) * 128, self.config.latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl_loss = 0


    def forward(self, data: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            data = layer(data)

        mu = self.latent_prj(data)
        sigma = torch.exp(self.latent_prj_sigma(data))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z