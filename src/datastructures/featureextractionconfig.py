from dataclasses import dataclass
from pathlib import Path

@dataclass
class FeatureExtractionConfig:
    window_size = 1024
    percent_overlap = 0.0
    secs_per_spectogram = 2
    outputlocation : Path