from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeatureExtractionConfig:
    outputlocation: Path
    window_size: int = 1024
    percent_overlap: float = 0.0
    secs_per_spectogram: float = 2.0
