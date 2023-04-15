from pathlib import Path
from matplotlib import pyplot as plt
import torch
from src.datastructures.visualizationconfig import VisualizationConfig
from src.io.trackfeaturesfilehandler import TrackFeaturesFileHandler
from src.datastructures.trainconfig import TrainConfig
from src.train.Trainer import Trainer
from src.train.Loss import absolute_pitch_diff_loss, relative_pitch_diff_loss
from src.train.pitchdataprovider import PitchDataProvider
from src.visualization.embeddingvisualizer import EmbeddingVisualizer


class PtichEstimationTrainer(Trainer):

    def __init__(self, config: TrainConfig) -> None:
        super().__init__(config)

    def _init_visualizer(self):
        vis_config = VisualizationConfig(self.config.modelconfig, self.config.modelstorageconfig, self.config.track_features_location / Path("Kodaline-Brother_ps0.00.wav_0.5s"), None, None, 1000)
        vis_config.multi_pitch_feature_location = self.config.track_features_location
        self.visualizer = EmbeddingVisualizer(vis_config, self.model)

    def _init_data(self):
        self.tf_dict = TrackFeaturesFileHandler().load_pitch_shifted_track_features(self.config.track_features_location)
        assert len(self.tf_dict) > 0
        tf = self.tf_dict[list(self.tf_dict.keys())[0]]
        self.frame_height = tf.frame_height
        self.frame_width = tf.frame_width

    def _init_loss(self):
        if self.config.trainparams.rec_loss == "bce":
            self.rec_loss_fn = torch.nn.BCEWithLogitsLoss()
        elif self.config.trainparams.rec_loss == "mse":
            self.rec_loss_fn = torch.nn.MSELoss()

        self.pitch_loss_fn = absolute_pitch_diff_loss

    def _init_model(self):
        self.config.modelconfig.encoderconfig.final_activation_fn = torch.nn.Identity()
        super()._init_model()
        

    def train(self):
        self.logger.watch_model(self.model)
        self.model.to(device="cuda")

        pitch_plot = self.visualizer.plot_multi_pitch_prediction_over_time(0.05)
        self.logger.log_figure_as_img("pitch_init", pitch_plot, commit=False)

        dataprovider = PitchDataProvider(self.tf_dict, self.config.trainparams.batch_size, 2)

        best_epoch_loss = 1000

        for e in range(self.config.trainparams.n_epochs):
            epoch_loss = 0
            print(f"training in epoch '{e}'")
            for input, y_pitch_levels in dataprovider:
                input_reshaped = input.reshape((input.shape[0]*input.shape[1], input.shape[-2],input.shape[-1]))
                rec_pred, embedded_pred = self.model(input_reshaped)
                rec_loss = self.rec_loss_fn(rec_pred, input_reshaped)
                self._log_loss("rec_loss", rec_loss)

                embedded_pred_reshaped = embedded_pred.reshape((input.shape[0], input.shape[1], embedded_pred.shape[-1]))
                pitch_loss = self.pitch_loss_fn(embedded_pred_reshaped, y_pitch_levels)
                self._log_loss("pitch_loss", pitch_loss)
                loss = rec_loss + pitch_loss
                self.logger.log({"overall_loss": loss}, commit=False)
                epoch_loss += loss.detach().cpu().numpy()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                # print(f" rec_loss {rec_loss_np}")

            # print(f"rec_pred mean {rec_pred.detach().cpu().numpy().mean()}")
            self.logger.log({"epoch":e, "epoch_loss": epoch_loss}, commit=True)
            

            if e % 4 == 0:
                pitch_plot = self.visualizer.plot_multi_pitch_prediction_over_time(0.05)
                self.logger.log_figure_as_img("pitch", pitch_plot, commit=False)
                plt.close("all")
                trajectory_plot = self.visualizer.plot_whole_track_trajectory()
                self.logger.log_figure_as_img("trajectory", trajectory_plot, commit=False)

            if e > self.config.trainparams.n_epochs / 4:                
                if epoch_loss < best_epoch_loss:
                    print(f"new best loss reached saving model.. epoch_loss {epoch_loss:.4f} best_epoch_loss {best_epoch_loss:.4f}")
                    best_epoch_loss = epoch_loss
                    self.storageHandler.save_model_state_to_file(self.model)
               

        print("\n\nfinished training ... \n\n")
        # self.logger.log_figure_as_img("trajectory_finished", trajectory_plot)
