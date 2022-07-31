from pathlib import Path
from cfgparser.json_config_parser import JSONConfigParser

from src.train.Trainer import Trainer


config = JSONConfigParser().parse_config_from_file(Path("./configs/train_track_config.json"))
trainer = Trainer(config)
trainer.train()