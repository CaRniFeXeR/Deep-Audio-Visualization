from pathlib import Path
from matplotlib import pyplot as plt
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
        self.storageHandler = ModelFileHandler(self.config.modelstorageconfig)
        self.logger = WandbLogger(self.config.wandbconfig, {**self.config.trainparams.__dict__})
        self.visualizer = EmbeddingVisualizer(VisualizationConfig(self.config.modelconfig, self.config.modelstorageconfig, self.config.track_features_location, None), self.model)
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.recloss_fn = torch.nn.MSELoss()

    def train(self):
        self.logger.watch_model(self.model)
        self.model.to(device="cuda")

        trajectory_plot = self.visualizer.plot_whole_track_trajectory()
        self.logger.log_figure_as_img("trajectory_init", trajectory_plot)

        dataprovider = DataProvider(self.tf, self.config.trainparams.batch_size, self.config.trainparams.prediction_seq_length)

        for e in range(self.config.trainparams.n_epochs):
            print(f"training in epoch '{e}'")
            for input_tensor in dataprovider:
                rec_pred = self.model(input_tensor)
                rec_loss = self.recloss_fn(rec_pred, input_tensor)

                input_seq = dataprovider.get_next_prediction_seq()
                if input_seq is not None and e > 15:
                    seq_shape = input_seq.shape
                    embedded_seq = self.model.embed_track_window(input_seq.view((seq_shape[0]*seq_shape[1], seq_shape[2], seq_shape[3])))
                    embedded_seq = embedded_seq.view((seq_shape[0], seq_shape[1], 3))
                    seq_pred = self.model.seq_prediction_forward(embedded_seq[:, :-1])
                    seq_gt = embedded_seq[:, -1]
                    seq_loss = self.recloss_fn(seq_pred, seq_gt)
                    self.logger.log({"seq_loss": seq_loss.detach().cpu().numpy()}, commit=False)
                    loss = rec_loss + 0.01 * seq_loss
                else:
                    loss = rec_loss

                loss.backward()
                self.optimizer.step()
                rec_loss_np = rec_loss.detach().cpu().numpy()
                print(f" rec_loss {rec_loss_np}")
                self.logger.log({"rec_loss": rec_loss_np}, commit=False)

            trajectory_plot = self.visualizer.plot_whole_track_trajectory()
            self.logger.log_figure_as_img("trajectory", trajectory_plot)
            plt.close()

        print("\n\nfinished training ... \n\n")
        self.storageHandler.save_model_state_to_file(self.model)
        self.logger.log_figure_as_img("trajectory_finished", trajectory_plot)
