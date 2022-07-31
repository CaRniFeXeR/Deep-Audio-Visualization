from pathlib import Path
from cfgparser.json_config_parser import JSONConfigParser

from src.visualization.embeddingvisualizer import EmbeddingVisualizer


config = JSONConfigParser().parse_config_from_file(Path("./configs/visualization_config.json"))
visualizer = EmbeddingVisualizer(config)
visualizer.render_video_track_trajectory()