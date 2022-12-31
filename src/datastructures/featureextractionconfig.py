from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeatureExtractionConfig:
    outputlocation: Path
    window_size: int = 1024
    percent_overlap: float = 0.0
    secs_per_spectrum: float = 2.0
    use_mel_spec : bool = False
    use_db_scale : bool = False
    calculate_centroids : bool = True
