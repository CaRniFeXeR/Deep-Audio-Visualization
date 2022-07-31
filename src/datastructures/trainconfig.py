from dataclasses import dataclass
from pathlib import Path

from ..datastructures.audiomodelconfig import AudioModelConfig
from ..datastructures.trainparams import Trainparams
from ..datastructures.wandbconfig import WandbConfig

@dataclass
class TrainConfig:
    modelconfig : AudioModelConfig
    trainparams : Trainparams
    track_features_location : Path
    wandbconfig : WandbConfig