from pathlib import Path
from cfgparser.json_config_parser import JSONConfigParser

from src.visualization.embeddingvisualizer import EmbeddingVisualizer


# config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_brother05s.json"))
# visualizer = EmbeddingVisualizer(config)
# visualizer.render_video_track_trajectory()
# config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_brother2s.json"))
# visualizer = EmbeddingVisualizer(config)
# visualizer.render_video_track_trajectory()
# config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_brother4s.json"))
# visualizer = EmbeddingVisualizer(config)
# visualizer.render_video_track_trajectory()
# config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_brother8s.json"))
# visualizer = EmbeddingVisualizer(config)
# visualizer.render_video_track_trajectory()

# config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_brother2s_new.json"))
# config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_tango2s.json"))
# config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_intheend4s.json"))
# config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_22703.json"))
# config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_lazygen.json"))
config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_brother2s_mel_db.json"))
config = JSONConfigParser().parse_config_from_file(Path("./configs/vis/visualization_config_brother2s_mel_centroid.json"))
visualizer = EmbeddingVisualizer(config)
visualizer.render_video_track_trajectory()  
