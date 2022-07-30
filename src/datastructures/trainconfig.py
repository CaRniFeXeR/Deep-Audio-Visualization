from dataclasses import dataclass

from src.datastructures.audiomodelconfig import AudioModelConfig

@dataclass
class TrainConfig:
    modelconfig : AudioModelConfig