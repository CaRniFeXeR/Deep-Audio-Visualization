from dataclasses import dataclass
from pathlib import Path


@dataclass
class PitchShiftConfig:
    outputlocation: Path
    min_pitch : float = -1.5
    max_pitch : float = 2.0
    step_size : float = 0.1
