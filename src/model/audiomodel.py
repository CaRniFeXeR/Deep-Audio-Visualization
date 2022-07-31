import torch
from src.datastructures.predictionheadconfig import SeqPredictorConfig
from src.model.decoder import Decoder

from src.model.encoder import Encoder
from src.model.sequence_predictor import SequencePredictor

from ..datastructures.audiomodelconfig import AudioModelConfig


class AudioModel(torch.nn.Module):

    def __init__(self, config : AudioModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(self.config.encoderconfig)
        self.decoder = Decoder(self.config.decoderconfig)
        if self.config.enable_prediction_head:
            self.sequence_predictor = SequencePredictor(self.config.SeqPredictorConfig)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

