import math
from pathlib import Path
from matplotlib import pyplot as plt

from ..utils.sequenceindexinghandler import cut_offset_start_end, extract_sequence
from ..utils.interpolationhandler import spine_interpolate
from ..utils.sequence_smoother import smooth_sequence

from ..visualization.equalizervishandler import EqualizerVisHandler
from .moviewriter import MovieWriter

import sys


gettrace = getattr(sys, 'gettrace', None)

if gettrace is None:
    plt.switch_backend('agg') #switch backend if debugger is not enabled
#plt.switch_backend('agg') #switch backend if debugger is not enabled

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
    
    def plot_track_pitch_over_time(self, percentage_length : 0.2, percentage_offset : 0.1, embedded : np.ndarray = None):
        if embedded is None:
            embedded = self.embed_track()
        pitch = embedded[:,0]
        # pitch = cut_offset_start_end(pitch, percentage_offset, percentage_offset)
        pitch = extract_sequence(pitch, 0, percentage_length)
        fig = plt.figure(figsize=(40,5))
        time = np.arange(len(pitch))

        plt.plot(time, pitch)

        return fig



    def plot_whole_track_trajectory(self):
        embedded = self.embed_track()
        a = 10.8
        fig = plt.figure(figsize=(1.7778 * a, a), constrained_layout=True)  # e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig
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

        n_tail_points = 50
        window_start = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.55, 0.6, 0.7, 0.8, 0.9]
        opening_window = window_start + (n_tail_points - len(window_start)) * [1.0]
        print(f"T_start including tail points {T_start+(n_tail_points-1)/self.tf.time_resolution} (s), T_end {T_end} (s)")

        line_color = "white" if self.config.dark_mode else "black"
        a = 10.9
        # a = 8
        plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
        fig = plt.figure(figsize=(1.7778*a, a))  # e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig
        if self.config.dark_mode == True:
            fig.set_facecolor("black")
        dtheta = 0.13/2.1  # 0.02  #rotation rate deg
        phi = 35  # elevation angle
        n_max_frames = int(len(embedded))  # - n_tail_points
        n_frames = self.config.n_frames
        assert n_frames <= n_max_frames
        
        line_interval = 500
        smooth_factor = self.config.embed_seq_smooth_window_size
        pooling_kernel_size = self.config.pooling_kernel_size
        embedded = smooth_sequence(embedded, smooth_factor)
        eqVis = EqualizerVisHandler(embedded, smooth_sequence(self.tf.get_normalized_magnitudes(), 2), pooling_kernel_size=pooling_kernel_size)
        embedded_s = spine_interpolate([embedded[20:-40,0], embedded[20:-40, 1], embedded[20:-40, 2]])
        
        n_bins = eqVis.s_mag_reduced.shape[0]
        angle_step = 360 / n_bins / 180 * np.pi
        print("n_bins", n_bins)

        def frame_fnc(given_t: float):
            t = int(given_t * 4 + n_tail_points) 
            t_org = int(given_t + n_tail_points)      
            fig.clear(keep_observers=True)
            ax = fig.add_subplot(projection='3d')
            ax.grid(self.config.show_grid)
            eqVis.set_axis_scale(ax)
            if self.config.plot_bins:
                eqVis.render_equalizer_bar(t, ax)

            if self.config.dark_mode == True:
                # ax.grid(False)
                ax.set_facecolor("black")
                ax.w_xaxis.pane.fill = False
                ax.w_yaxis.pane.fill = False
                ax.w_zaxis.pane.fill = False

                # Transparent spines
                ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

                # Transparent panes
                ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.set_axis_off()

            if t > line_interval:
                sp = t - line_interval
                sp_org = t_org - line_interval
            else:
                sp = 0
                sp_org = 0
            n_points = t - sp
            if t > 20:
                t_half = sp + int(np.round((t-sp) / 2))
                ax.plot3D(embedded_s[0][sp:t_half], embedded_s[1][sp:t_half], embedded_s[2][sp:t_half], '-', markerfacecolor=line_color, markersize=1.9, linewidth=1.9, color=line_color, alpha=0.25, label='Z')
                ax.plot3D(embedded_s[0][t_half:t], embedded_s[1][t_half:t], embedded_s[2][t_half:t], '-', markerfacecolor=line_color, markersize=2, linewidth=2, color=line_color, alpha=0.35, label='Z')
                #ax.plot3D(embedded[sp_org+20:t_org+20,0], embedded[sp_org+20:t_org+20,1], embedded[sp_org+20:t_org+20,2], '-', markerfacecolor="red", markersize=2, linewidth=1, color="red", alpha=1.0, label='Z')
            if t > n_tail_points:
                tail_s = t-n_tail_points
                t_s_half = t - int(n_tail_points / 2)
                # ax.plot3D(embedded[tail_s:t, 0], embedded[tail_s:t, 1], embedded[tail_s:t, 2], '-o', markerfacecolor='orange', mec='darkblue', markersize=12* (eqVis.s_mag_reduced[0,t]*10 + 0.5), linewidth=2, label='Z(t)')

                angle = 0
                for fbin in range(0, n_bins):
                    feqbin_factor = eqVis.s_mag_reduced[fbin, tail_s:t] * opening_window * self.config.feqbin_offset_intensity
                    mean_feqbin_intensity = eqVis.s_mag_reduced[fbin, t_s_half:t].mean()
                    size = 2 + mean_feqbin_intensity * self.config.feqbin_linewidth_intensity
                    alpha = 0.35 + 0.65 * mean_feqbin_intensity
                    # adjusted_line = [embedded[tail_s:t, 0] - feqbin_factor * math.cos(angle), embedded[tail_s:t, 1] + feqbin_factor * math.cos(angle), embedded[tail_s:t, 2] + feqbin_factor * math.sin(angle)]
                    adjusted_line = [embedded_s[0][tail_s:t] - feqbin_factor * math.cos(angle), embedded_s[1][tail_s:t] + feqbin_factor * math.cos(angle), embedded_s[2][tail_s:t] + feqbin_factor * math.sin(angle)]
                    # line_smoothed = spine_interpolate(adjusted_line)
                    # ax.plot3D(line_smoothed[0], line_smoothed[1], line_smoothed[2], '-', markersize=size, linewidth=size, alpha=alpha, label='Z(t)')
                    ax.plot3D(adjusted_line[0], adjusted_line[1], adjusted_line[2], '-', markersize=size, linewidth=size, alpha=alpha, label='Z(t)')
                    angle += angle_step

            ax.view_init(phi, dtheta * t)  # view_init(elev=None, azim=None)
            # ax.axis('off')  # for saving transparent gifs
            ax.dist = 8.4
            plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
            plt.draw()
            return mplfig_to_npimage(fig)

        movieWriter = MovieWriter(frame_fnc, self.config.movie_out_location, f"{self.config.track_features_location.name}_sm{smooth_factor}p{pooling_kernel_size}.mp4", n_frames, float(self.tf.time_resolution), self.config.track_audio_location)
        movieWriter.write_video_file()

    def render_video_pitch(self):
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

        n_tail_points = 50
        print(f"T_start including tail points {T_start+(n_tail_points-1)/self.tf.time_resolution} (s), T_end {T_end} (s)")

        line_color = "white" if self.config.dark_mode else "black"
        a = 10.9
        # a = 8
        plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
        fig, ax = plt.subplots(figsize=(1.7778*a, a)) # e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig
        if self.config.dark_mode == True:
            fig.set_facecolor("black")
            ax.set_facecolor("black")
        n_max_frames = int(len(embedded))  # - n_tail_points
        n_frames = self.config.n_frames
        assert n_frames <= n_max_frames
        
        line_interval = 250
        smooth_factor = self.config.embed_seq_smooth_window_size
        pooling_kernel_size = self.config.pooling_kernel_size
        # embedded = smooth_sequence(embedded, smooth_factor)
        # eqVis = EqualizerVisHandler(embedded, smooth_sequence(self.tf.get_normalized_magnitudes(), 2), pooling_kernel_size=pooling_kernel_size)
        # embedded_s = spine_interpolate([embedded[20:-40,0], embedded[20:-40, 1], embedded[20:-40, 2]])
        min_pitch = embedded[:,0].min()
        max_pitch = embedded[:,0].max()
        def frame_fnc(given_t: float):
            t = int(given_t * 4 + n_tail_points) 
            ax.clear()
            ax.grid(self.config.show_grid)
     
     

            if t > line_interval:
                sp = t - line_interval
            else:
                sp = 0
            # n_points = t - sp
            if t > 20 and t < len(embedded):
                t_half = sp + int(np.round((t-sp) / 2))
                ax.plot(list(range(sp,t_half)),embedded[sp:t_half,0], '-', markerfacecolor=line_color, markersize=1.9, linewidth=1.8, color=line_color, alpha=0.6, label='Z')
                ax.plot(list(range(t_half-1,t)),embedded[t_half-1:t,0], '-', markerfacecolor=line_color, markersize=2, linewidth=2, color=line_color, alpha=0.7, label='Z')
                ax.set_ylim(min_pitch, max_pitch)

            # plt.draw()
            # fig.show()
            return mplfig_to_npimage(fig)
        # frame_fnc(100)
        # frame_fnc(101)
        # plt.show()

        movieWriter = MovieWriter(frame_fnc, self.config.movie_out_location, f"pitch_{self.config.track_features_location.name}_sm{smooth_factor}p{pooling_kernel_size}.mp4", n_frames, float(self.tf.time_resolution), self.config.track_audio_location, audio_offset_percent= 0.1)
        movieWriter.write_video_file()