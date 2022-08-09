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
        self.config.modelconfig.encoderconfig.features_in_dim = self.tf.frame_height
        self.config.modelconfig.encoderconfig.frame_width_in = self.tf.frame_width
        self.config.modelconfig.decoderconfig.output_dim = self.tf.frame_height
        self.config.modelconfig.decoderconfig.output_length = self.tf.frame_width
        self.model = AudioModel(self.config.modelconfig)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.trainparams.learning_rate)
        self.storageHandler = ModelFileHandler(self.config.modelstorageconfig)
        self.logger = WandbLogger(self.config.wandbconfig, {**self.config.trainparams.__dict__})
        self.visualizer = EmbeddingVisualizer(VisualizationConfig(self.config.modelconfig, self.config.modelstorageconfig, self.config.track_features_location, None), self.model)
        
        if self.config.trainparams.rec_loss == "bce":
            self.rec_loss_fn = torch.nn.BCEWithLogitsLoss()
        elif self.config.trainparams.rec_loss == "mse":
            self.rec_loss_fn = torch.nn.MSELoss()
        
        if self.config.trainparams.seq_loss == "mse":
            self.seq_loss_fn =  torch.nn.MSELoss()
        elif self.config.trainparams.seq_loss == "cosine":
            cosineloss = torch.nn.CosineEmbeddingLoss()
            one = torch.ones(1).to(device = "cuda")
            def loss_fn(x : torch.Tensor,y : torch.Tensor) -> torch.Tensor:
                return cosineloss(x,y, one)

            self.seq_loss_fn = loss_fn

    def train(self):
        self.logger.watch_model(self.model)
        self.model.to(device="cuda")

        # trajectory_plot = self.visualizer.plot_whole_track_trajectory()
        # self.logger.log_figure_as_img("trajectory_init", trajectory_plot)

        dataprovider = DataProvider(self.tf, self.config.trainparams.batch_size, self.config.trainparams.prediction_seq_length)

        for e in range(self.config.trainparams.n_epochs):
            print(f"training in epoch '{e}'")
            for input_tensor in dataprovider:
                rec_pred = self.model(input_tensor)
                rec_loss = self.rec_loss_fn(rec_pred, input_tensor)

                input_seq = dataprovider.get_next_prediction_seq()
                if input_seq is not None and self.config.modelconfig.enable_prediction_head and e > self.config.trainparams.seq_prediction_start_epoch:
                    seq_shape = input_seq.shape
                    embedded_seq = self.model.embed_track_window(input_seq.view((seq_shape[0]*seq_shape[1], seq_shape[2], seq_shape[3])))
                    embedded_seq = embedded_seq.view((seq_shape[0], seq_shape[1], self.config.modelconfig.seqpredictorconfig.latent_dim))
                    seq_pred = self.model.seq_prediction_forward(embedded_seq[:, :-1])
                    seq_gt = embedded_seq[:, -1]
                    seq_loss = self.seq_loss_fn(seq_pred, seq_gt)
                    self.logger.log({"seq_loss": seq_loss.detach().cpu().numpy()}, commit=False)
                    loss = rec_loss + 0.01 * seq_loss
                else:
                    loss = rec_loss

                loss.backward()
                self.optimizer.step()
                rec_loss_np = rec_loss.detach().cpu().numpy()
                print(f" rec_loss {rec_loss_np}")
                self.logger.log({"rec_loss": rec_loss_np}, commit=False)

            print(f"rec_pred mean {rec_pred.detach().cpu().numpy().mean()}")
            trajectory_plot = self.visualizer.plot_whole_track_trajectory()
            self.logger.log_figure_as_img("trajectory", trajectory_plot)
            plt.close("all")

        print("\n\nfinished training ... \n\n")
        self.storageHandler.save_model_state_to_file(self.model)
        self.logger.log_figure_as_img("trajectory_finished", trajectory_plot)
