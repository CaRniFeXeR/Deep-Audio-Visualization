from distutils.command.config import config
from pathlib import Path
from cfgparser.json_config_parser import JSONConfigParser

from src.preprocessing.featureexetractor import FeatureExtractor

config = JSONConfigParser().parse_config_from_file(Path("./configs/extract_features_config.json"))
featureExtactor = FeatureExtractor(config)
featureExtactor.extract_features(Path("./tracks/Majulah.wav"))