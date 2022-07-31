from pathlib import Path
import torch

from .dataprovider import DataProvider
from ..datastructures.visualizationconfig import VisualizationConfig
from ..visualization.embeddingvisualizer import EmbeddingVisualizer
from ..io.modelfilehandler import ModelFileHandler
from ..io.trackfeaturesfilehandler import TrackFeaturesFileHandler
from .WandbLogger import WandbLogger
from ..datastructures.trainconfig import TrainConfig
from ..model.audiomodel import AudioModel


class Trainer:

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.tf = TrackFeaturesFileHandler().load_track_features(self.config.track_features_location / Path("vars.npz"))
        self.config.modelconfig.encoderconfig.features_in_dim = self.tf.img_height
        self.config.modelconfig.decoderconfig.output_width = self.tf.img_height
        self.model = AudioModel(self.config.modelconfig)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.trainparams.learning_rate)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.storageHandler = ModelFileHandler(self.config.modelstorageconfig)
        self.logger = WandbLogger(self.config.wandbconfig, {**self.config.trainparams.__dict__})
        self.visualizer = EmbeddingVisualizer(VisualizationConfig(self.config.modelconfig, self.config.modelstorageconfig, self.config.track_features_location, None), self.model)
        # self.loss_fn = torch.nn.MSELoss()

    def train(self):
        self.logger.watch_model(self.model)
        self.model.to(device="cuda")

        S_mag_norm = self.tf.get_normalized_magnitudes()  # e.g. shape = (94, 1757)
        trajectory_plot = self.visualizer.plot_whole_track_trajectory()
        self.logger.log_figure_as_img("trajectory_init", trajectory_plot)

        dataprovider = DataProvider(self.tf, self.config.trainparams.batch_size)

        for e in range(self.config.trainparams.n_epochs):
            print(f"training in epoch '{e}'")
            for input_tensor in dataprovider:
                predicted = self.model(input_tensor)
                loss = self.loss_fn(predicted, input_tensor)
                loss.backward()
                self.optimizer.step()
                loss_np = loss.detach().cpu().numpy()
                print(f" loss {loss_np}")
                self.logger.log({"rec_loss": loss_np}, commit=False)

            trajectory_plot = self.visualizer.plot_whole_track_trajectory()
            self.logger.log_figure_as_img("trajectory", trajectory_plot)

        print("\n\nfinished training ... \n\n")
        self.storageHandler.save_model_state_to_file(self.model)
        self.logger.log_figure_as_img("trajectory_finished", trajectory_plot)
