from dataclasses import dataclass

from src.datastructures.predictionheadconfig import SeqPredictorConfig

from .decoderconfig import DecoderConfig
from .encoderconfig import EncoderConfig

@dataclass
class AudioModelConfig:
    decoderconfig : DecoderConfig
    encoderconfig : EncoderConfig
    seqpredictorconfig : SeqPredictorConfig
    enable_prediction_head : bool = True