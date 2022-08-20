import math
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..preprocessing.sequence_smoother import smooth_sequence

from ..visualization.equalizervishandler import EqualizerVisHandler
from .moviewriter import MovieWriter

# plt.switch_backend('agg')

import torch

from ..io.modelfilehandler import ModelFileHandler
from ..datastructures.visualizationconfig import VisualizationConfig
from ..io.trackfeaturesfilehandler import TrackFeaturesFileHandler
from ..model.audiomodel import AudioModel
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np


class EmbeddingVisualizer:

    def __init__(self, config: VisualizationConfig, model: AudioModel = None) -> None:
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
        a = 8
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

        n_tail_points = 15
        print(f"T_start including tail points {T_start+(n_tail_points-1)/self.tf.time_resolution} (s), T_end {T_end} (s)")

        line_color = "white" if self.config.dark_mode else "black"
        a = 6
        fig = plt.figure(figsize=(1.7778*a, a))  # e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig
        if self.config.dark_mode == True:
            fig.set_facecolor("black")
        dtheta = 0.13/2.1  # 0.02  #rotation rate deg
        phi = 35  # elevation angle
        n_frames = int(len(embedded))  # - n_tail_points
        line_interval = 250
        n_frames = 3000
        smooth_factor = 4
        embedded = smooth_sequence(embedded, smooth_factor)
        eqVis = EqualizerVisHandler(embedded, smooth_sequence(self.tf.get_normalized_magnitudes(), 8), pooling_kernel_size=12)
        n_bins = eqVis.s_mag_reduced.shape[0]
        angle_step = 360 / n_bins / 180 * np.pi

        def frame_fnc(given_t: float):
            t = int(given_t + n_tail_points)
            fig.clear(keep_observers=True)
            ax = fig.add_subplot(projection='3d')
            eqVis.set_axis_scale(ax)
            if self.config.plot_bins:
                eqVis.render_equalizer_bar(t, ax)

            if self.config.dark_mode == True:
                # ax.grid(False)
                ax.set_facecolor("black")
                ax.w_xaxis.pane.fill = False
                ax.w_yaxis.pane.fill = False
                ax.w_zaxis.pane.fill = False

            if t > line_interval:
                sp = t - line_interval
            else:
                sp = 0
            # todo fadeout
            n_points = t - sp
            # lc = Line3DCollection(embedded[sp:t], linewidths = np.arange(n_points) / n_points * 1.1, color=line_color)
            # ax.add_collection(lc)
            ax.plot3D(embedded[sp:t, 0], embedded[sp:t, 1], embedded[sp:t, 2], '-', markerfacecolor=line_color, markersize=1, linewidth=1, color=line_color, alpha = 0.3, label='Z')
            if t > n_tail_points:
                tail_s = t-n_tail_points
                # ax.plot3D(embedded[tail_s:t, 0], embedded[tail_s:t, 1], embedded[tail_s:t, 2], '-o', markerfacecolor='orange', mec='darkblue', markersize=12* (eqVis.s_mag_reduced[0,t]*10 + 0.5), linewidth=2, label='Z(t)')
                
                angle = 0
                for fbin in range(0, n_bins):
                    feqbin_factor = eqVis.s_mag_reduced[fbin, tail_s:t] * 300

                    ax.plot3D(embedded[tail_s:t, 0], embedded[tail_s:t, 1] + feqbin_factor * math.cos(angle), embedded[tail_s:t, 2] + feqbin_factor * math.sin(angle), '-', markerfacecolor='darkblue', mec='darkblue', markersize=1.0, linewidth=1.0, label='Z(t)')
                    angle += angle_step

                # feqbin_factor = eqVis.s_mag_reduced[0,tail_s:t] * 500
                # ax.plot3D(embedded[tail_s:t, 0], embedded[tail_s:t, 1], embedded[tail_s:t, 2] + feqbin_factor, '-', markerfacecolor='darkblue', mec='darkblue', markersize=1, linewidth=1, label='Z(t)')
                # feqbin_factor = eqVis.s_mag_reduced[1,tail_s:t] * 500
                # ax.plot3D(embedded[tail_s:t, 0], embedded[tail_s:t, 1] + feqbin_factor, embedded[tail_s:t, 2], '-', markerfacecolor='darkblue', mec='darkblue', markersize=1, linewidth=1, label='Z(t)')
                # feqbin_factor = eqVis.s_mag_reduced[2,tail_s:t] * 500
                # ax.plot3D(embedded[tail_s:t, 0] + feqbin_factor, embedded[tail_s:t, 1], embedded[tail_s:t, 2], '-', markerfacecolor='darkblue', mec='darkblue', markersize=1, linewidth=1, label='Z(t)')
                # feqbin_factor = eqVis.s_mag_reduced[3,tail_s:t] * 500
                # ax.plot3D(embedded[tail_s:t, 0], embedded[tail_s:t, 1], embedded[tail_s:t, 2] - feqbin_factor, '-', markerfacecolor='darkblue', mec='darkblue', markersize=1, linewidth=1, label='Z(t)')

            ax.view_init(phi, dtheta * t)  # view_init(elev=None, azim=None)
            # ax.axis('off')  # for saving transparent gifs
            ax.dist = 8

            plt.draw()
            return mplfig_to_npimage(fig)

        movieWriter = MovieWriter(frame_fnc, self.config.movie_out_location, f"{self.config.track_features_location.name}_sm{smooth_factor}.mp4", n_frames, float(self.tf.time_resolution), self.config.track_audio_location)
        movieWriter.write_video_file()
