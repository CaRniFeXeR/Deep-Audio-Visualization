from typing import Tuple
import torch
from .varational_encoder import VarationalEncoder
from .decoder import Decoder
from .encoder import Encoder
from .sequence_predictor import SequencePredictor

from ..datastructures.audiomodelconfig import AudioModelConfig


class AudioModel(torch.nn.Module):

    def __init__(self, config : AudioModelConfig) -> None:
        super().__init__()
        self.config = config
        self.config.decoderconfig.latent_dim = self.config.encoderconfig.latent_dim
        self.config.seqpredictorconfig.latent_dim = self.config.encoderconfig.latent_dim
        
        self.encoder = VarationalEncoder(self.config.encoderconfig) if self.config.encoderconfig.use_variational_encoder else Encoder(self.config.encoderconfig)
        self.decoder = Decoder(self.config.decoderconfig)
        if self.config.enable_prediction_head:
            self.sequence_predictor = SequencePredictor(self.config.seqpredictorconfig)
        else:
            self.sequence_predictor = None
   

    def embed_track_window(self, x : torch.Tensor) -> torch.Tensor:
        return self.encoder(x)    
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        encoded = self.embed_track_window(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def seq_prediction_forward(self, x : torch.Tensor) -> torch.Tensor:
        assert self.sequence_predictor is not None
        
        return self.sequence_predictor(x)



