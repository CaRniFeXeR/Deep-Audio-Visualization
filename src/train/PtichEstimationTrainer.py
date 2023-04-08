from matplotlib import pyplot as plt
import torch
from src.io.trackfeaturesfilehandler import TrackFeaturesFileHandler
from src.datastructures.trainconfig import TrainConfig
from src.train.Trainer import Trainer
from src.train.Loss import absolute_pitch_diff_loss
from src.train.pitchdataprovider import PitchDataProvider


class PtichEstimationTrainer(Trainer):

    def __init__(self, config: TrainConfig) -> None:
        super().__init__(config)

    def _init_visualizer(self):
        pass

    def _init_data(self):
        self.tf_dict = TrackFeaturesFileHandler().load_pitch_shifted_track_features(self.config.track_features_location)
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
        super()._init_model()

    def train(self):
        self.logger.watch_model(self.model)
        self.model.to(device="cuda")

        # trajectory_plot = self.visualizer.plot_whole_track_trajectory()
        # self.logger.log_figure_as_img("trajectory_init", trajectory_plot)

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

                overall_loss_np = loss.detach().cpu().numpy()
                epoch_loss += overall_loss_np
                self.logger.log({"overall_loss": overall_loss_np}, commit=False)
                loss.backward()
                self.optimizer.step()
                # print(f" rec_loss {rec_loss_np}")

            # print(f"rec_pred mean {rec_pred.detach().cpu().numpy().mean()}")
            # trajectory_plot = self.visualizer.plot_whole_track_trajectory()
            # self.logger.log_figure_as_img("trajectory", trajectory_plot)
            # plt.close("all")

            if e > self.config.trainparams.n_epochs / 4 and epoch_loss < best_epoch_loss:
                print(f"new best loss reached saving model.. epoch_loss {epoch_loss:.4f} best_epoch_loss {best_epoch_loss:.4f}")
                best_epoch_loss = epoch_loss
                self.storageHandler.save_model_state_to_file(self.model)

        print("\n\nfinished training ... \n\n")
        # self.logger.log_figure_as_img("trajectory_finished", trajectory_plot)
