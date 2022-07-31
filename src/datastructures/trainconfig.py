from dataclasses import dataclass
from pathlib import Path

from ..datastructures.modelstorageconfig import ModelStorageConfig
from ..datastructures.audiomodelconfig import AudioModelConfig
from ..datastructures.trainparams import Trainparams
from ..datastructures.wandbconfig import WandbConfig

@dataclass
class TrainConfig:
    modelconfig : AudioModelConfig
    modelstorageconfig : ModelStorageConfig
    trainparams : Trainparams
    track_features_location : Path
    wandbconfig : WandbConfig