from dataclasses import dataclass


@dataclass
class DecoderConfig:
    n_layers : int = 2
    output_width : int = 88
    latent_dim : int = None