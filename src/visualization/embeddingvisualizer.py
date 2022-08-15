from pathlib import Path
from matplotlib import pyplot as plt

from .moviewriter import MovieWriter

plt.switch_backend('agg')

import torch

from ..io.modelfilehandler import ModelFileHandler
from ..datastructures.visualizationconfig import VisualizationConfig
from ..io.trackfeaturesfilehandler import TrackFeaturesFileHandler
from ..model.audiomodel import AudioModel
from .videowriter import VideoWriter
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np


class EmbeddingVisualizer:

    def __init__(self, config: VisualizationConfig, model : AudioModel = None) -> None:
        self.config = config
        self.tf = TrackFeaturesFileHandler().load_track_features(self.config.track_features_location / Path("vars.npz"))
        self.config.modelconfig.encoderconfig.features_in_dim = self.tf.frame_height
        self.config.modelconfig.encoderconfig.frame_width_in = self.tf.frame_width
        self.config.modelconfig.decoderconfig.output_dim = self.tf.frame_height
        self.config.modelconfig.decoderconfig.output_length = self.tf.frame_width
        self.model = model
        if self.model == None:
            self.model = AudioModel(self.config.modelconfig)
            self.model = ModelFileHandler(self.config.modelstorageconfig).load_state_from_file(self.model, True)

    def embed_track(self) -> np.ndarray:
        self.model.to(device="cuda")
        S_mag_norm = self.tf.get_normalized_magnitudes()
        embedded_points = []
        for w_start in range(S_mag_norm.shape[1] - self.tf.frame_width):
            w_end = w_start + self.tf.frame_width
            input_tensor = torch.from_numpy(S_mag_norm[:, w_start:w_end]).to(device="cuda").unsqueeze(dim=0)
            encoded = self.model.embed_track_window(input_tensor)
            embedded_points.append(encoded.detach().cpu().numpy())

        embeded_track = np.concatenate(embedded_points)
        return embeded_track

    def plot_whole_track_trajectory(self):
        embedded = self.embed_track()
        a = 7
        fig = plt.figure(figsize=(1.7778 * a, a))  # e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig
        ax = fig.gca(projection='3d')

        # ax.set_xlabel('$X$', fontsize=20)
        # ax.set_ylabel('$Y$')
        # disable auto rotation
        # ax.zaxis.set_rotate_label(False)
        # ax.set_zlabel('$\gamma$', fontsize=10, rotation = 0)
        ax.plot3D(embedded[:, 0], embedded[:, 1], embedded[:, 2], 'k-', markerfacecolor='black', markersize=1, linewidth=0.2, label='Z')
        return fig

    def render_video_track_trajectory(self):
        self.config.movie_out_location.mkdir(exist_ok=True, parents=True)
        embedded = self.embed_track()

        l = 45
        # determine real time start and end of the embedding in seconds
        t_start = int(self.tf.dt*(l - 1))  # first time bin index (in S_mag) in the valid range
        t_end = int(len(self.tf.T) - self.tf.frame_width - self.tf.dt - 1)  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
        T_start = self.tf.T[t_start]
        T_end = self.tf.T[t_end]
        samples = np.floor((t_end - t_start + 1)/self.tf.dt)

        print(f"t_start {t_start} (bins), t_end {t_end} (bins)")
        print(f"T_start {T_start} (s), T_end {T_end} (s)")
        print(f" seconds per Spectrogram time bin: {1/self.tf.time_resolution}")
        # print(f"30fps T_start | seconds:frame | {np.floor(T_start)}:{np.round((T_start-np.floor(T_start))*30)}")
        # print(f"30fps T_end | seconds:frame |  {np.floor(T_end)}:{np.round((T_end-np.floor(T_end))*30)}")
        print(f"bins spanning input tensor: {(l-1)*self.tf.dt}")
        print(f"time spanning input tensor: {(l-1)*self.tf.dt/self.tf.time_resolution} (s)")  # should equal T_start
        print(f"Total frames to be rendered: {samples}")

        tail_points = 12
        print(f"T_start including tail points {T_start+(tail_points-1)/self.tf.time_resolution} (s), T_end {T_end} (s)")

        a = 8
        fig2 = plt.figure(figsize=(1.7778*a, a))  #  e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig
        dtheta = 0.13/2.1  # 0.02  #rotation rate deg
        k = 0
        dk = 0  # 2*np.pi/360*0.05
        theta = 1
        # phi_0 = 15
        phi = 35  # elevation angle
        render_interval = 1
        n_frames = int(len(embedded)) - tail_points
        # n_frames = 200
        # for t in range(tail_points, int(len(embedded)), render_interval):
        def frame_fnc(given_t : float):
            t = int(given_t + tail_points)
            fig2.clear(keep_observers=True)
            ax = fig2.add_subplot(projection='3d')
            ax.plot3D(embedded[:t, 0], embedded[:t, 1], embedded[:t, 2], '-', markerfacecolor='black', markersize=1, linewidth=1, color='black', label='Z')
            ax.plot3D(embedded[t-tail_points:t, 0], embedded[t-tail_points:t, 1], embedded[t-tail_points:t, 2], '-o', markerfacecolor='orange', mec='darkblue', markersize=12, linewidth=2, label='Z(t)')

            # phi = 20*np.sin(k) + phi_0
            # phi = 0
            # print(phi)
            ax.view_init(phi, dtheta * t)  #view_init(elev=None, azim=None)
            # ax.axis('off')  # for saving transparent gifs
            ax.dist = 8
            plt.draw()
            plt.pause(.01)
            # fig2.savefig(self.config.movie_out_location / Path(f'frame_{t:03}.png'), transparent=False, dpi='figure', bbox_inches=None)

            # k += dk
            # print(t)
            return mplfig_to_npimage(fig2)

        # movieWriter = VideoWriter(self.config.movie_out_location, "movie.avi", fps = float(self.tf.time_resolution))
        movieWriter = MovieWriter(frame_fnc, self.config.movie_out_location,"movie.mp4", n_frames, float(self.tf.time_resolution), self.config.track_audio_location)
        movieWriter.write_video_file()