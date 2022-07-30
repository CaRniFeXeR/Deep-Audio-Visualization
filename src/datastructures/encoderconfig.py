from dataclasses import dataclass

@dataclass
class EncoderConfig:
    n_layers : int = 2
    features_in_dim : int = 32