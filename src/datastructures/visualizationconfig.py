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
    n_frames : int
    track_audio_location : Path = None
    dark_mode : bool = True
    plot_bins : bool = False
    show_grid : bool = False
    multi_pitch_feature_location : Path = None
    feqbin_offset_intensity : int = 110
    feqbin_linewidth_intensity : int = 12
    pooling_kernel_size : int = 10
    embed_seq_smooth_window_size : int = 4