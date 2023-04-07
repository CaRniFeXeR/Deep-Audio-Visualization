from distutils.command.config import config
from pathlib import Path
from cfgparser.json_config_parser import JSONConfigParser

from src.preprocessing.featureexetractor import FeatureExtractor

config = JSONConfigParser().parse_config_from_file(Path("./configs/extract_features_pitch_shifted_config.json"))
# config.secs_per_spectrum = 4.0
featureExtactor = FeatureExtractor(config)
featureExtactor.generate_pitchshifted_features(Path("./data/tracks/Kodaline-Brother.wav"))
# featureExtactor.extract_features(Path("./data/tracks/bust_tut_mir_leid/bust_tut mir leid dass ichs schon kenn.wav"))
# featureExtactor.extract_features(Path("./data/tracks/PJ_lazygen.wav"))
# featureExtactor.extract_features(Path("./data/tracks/InTheEnd_remixed.wav"))