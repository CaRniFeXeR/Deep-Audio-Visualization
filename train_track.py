from pathlib import Path
from cfgparser.json_config_parser import JSONConfigParser

from src.train.Trainer import Trainer


# config = JSONConfigParser().parse_config_from_file(Path("./configs/train/train_track_config_bust4s.json"))
# config = JSONConfigParser().parse_config_from_file(Path("./configs/train/train_track_config_paulsj_lazy.json"))
# config = JSONConfigParser().parse_config_from_file(Path("./configs/train/train_track_config_kodaline2s_seq_cosine.json"))
config = JSONConfigParser().parse_config_from_file(Path("./configs/train/train_track_config_kodaline2s_mel_new.json"))
trainer = Trainer(config)
trainer.train()