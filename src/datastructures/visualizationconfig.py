from dataclasses import dataclass
from pathlib import Path

from ..datastructures.audiomodelconfig import AudioModelConfig
from ..datastructures.modelstorageconfig import ModelStorageConfig

@dataclass
class VisualizationConfig:
    modelconfig : AudioModelConfig
    modelstorageconfig : ModelStorageConfig
    track_features_location : Path
    movie_out_location : Path