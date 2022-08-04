from dataclasses import dataclass

@dataclass
class SeqPredictorConfig:
    n_layers : int = 2
    latent_dim : int = None