from dataclasses import dataclass
from pathlib import Path


@dataclass
class PitchShiftConfig:
    outputlocation: Path
    min_pitch: float = -2.0
    max_pitch: float = 2.0
    step_size: float = 0.1
    percent_offsets: tuple = (0.1, 0.1)
