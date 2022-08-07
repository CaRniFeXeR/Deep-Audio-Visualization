from dataclasses import dataclass

@dataclass
class EncoderConfig:
    n_layers : int = 2
    features_in_dim : int = 32
    frame_width_in : int = 22
    latent_dim : int = 3