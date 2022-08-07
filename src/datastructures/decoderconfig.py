from dataclasses import dataclass


@dataclass
class DecoderConfig:
    n_layers : int = 2
    output_dim : int = 88
    output_length : int = 22
    latent_dim : int = None