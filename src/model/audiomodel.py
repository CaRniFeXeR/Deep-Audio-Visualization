from typing import Tuple
import torch
from src.model.decoder import Decoder

from src.model.encoder import Encoder
from src.model.sequence_predictor import SequencePredictor

from ..datastructures.audiomodelconfig import AudioModelConfig


class AudioModel(torch.nn.Module):

    def __init__(self, config : AudioModelConfig) -> None:
        super().__init__()
        self.config = config
        self.config.decoderconfig.latent_dim = self.config.encoderconfig.latent_dim
        self.config.seqpredictorconfig.latent_dim = self.config.encoderconfig.latent_dim
        
        self.encoder = Encoder(self.config.encoderconfig)
        self.decoder = Decoder(self.config.decoderconfig)
        if self.config.enable_prediction_head:
            self.sequence_predictor = SequencePredictor(self.config.seqpredictorconfig)
   

    def embed_track_window(self, x : torch.Tensor) -> torch.Tensor:
        return self.encoder(x)    
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        encoded = self.embed_track_window(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def seq_prediction_forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.sequence_predictor(x)



