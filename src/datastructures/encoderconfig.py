from dataclasses import dataclass

import torch

@dataclass
class EncoderConfig:
    n_layers : int = 2
    features_in_dim : int = 32
    frame_width_in : int = 22
    latent_dim : int = 3
    final_activation_fn : torch.nn.Module = torch.nn.LeakyReLU()
    use_variational_encoder : bool = False
