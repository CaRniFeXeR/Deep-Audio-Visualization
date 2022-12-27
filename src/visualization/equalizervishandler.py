import numpy as np
from skimage.measure import block_reduce


class EqualizerVisHandler():

    def __init__(self, embedded: np.ndarray, s_mag_norm: np.ndarray, pooling_kernel_size: int = 6) -> None:
        self.max_embedded = embedded[50:-50].max(axis=0)  # ignoring the first and last frames often outside the actual position
        self.min_embedded = embedded[50:-50].min(axis=0)
        self.embeded_range = self.max_embedded - self.min_embedded
        print(self.embeded_range)
        s_mag_extend = []
        # interplot 3 points per step
        for i in range(s_mag_norm.shape[-1] - 1):
            s_mag_extend.append(s_mag_norm[:, i])
            s_mag_extend.append(3/4 * s_mag_norm[:, i] + 1/4 * s_mag_norm[:, i+1])
            s_mag_extend.append(2/4 * s_mag_norm[:, i] + 2/4 * s_mag_norm[:, i+1])
            s_mag_extend.append(1/4 * s_mag_norm[:, i] + 3/4 * s_mag_norm[:, i+1])
        s_mag_norm = np.array(s_mag_extend).transpose()
        s_mag_reduced_max = block_reduce(s_mag_norm, block_size=(pooling_kernel_size, 1), func=np.max)
        s_mag_reduced_avg = block_reduce(s_mag_norm, block_size=(pooling_kernel_size, 1), func=np.mean)
        self.s_mag_reduced = (s_mag_reduced_max + s_mag_reduced_avg) / 2
        #norm per bin
        #max_value = np.max(self.s_mag_reduced)* 0.5
        max_per_bin = np.max(self.s_mag_reduced, axis=1)[:, None]  #* 0.8
        self.s_mag_reduced = self.s_mag_reduced / max_per_bin # * max_value
        self.n_bins = self.s_mag_reduced.shape[0]

        self.s_mag_norm = self.s_mag_reduced * self.embeded_range[2]


    def set_axis_scale(self, ax):
        ax.set_xlim(self.min_embedded[0], self.max_embedded[0])
        ax.set_ylim(self.min_embedded[1], self.max_embedded[1])
        ax.set_zlim(self.min_embedded[2], self.max_embedded[2])

    def render_equalizer_bar(self, t: int, ax):

        bin_pos = self.min_embedded[0] + np.arange(self.n_bins) / self.n_bins * self.embeded_range[0]
        bin_height = self.s_mag_norm[:, t]
        bin_width = self.embeded_range[0] / (self.n_bins + 1)
        bottom = self.min_embedded[2]
        ax.bar(bin_pos, bin_height, width=bin_width, bottom=bottom, zs=self.min_embedded[1] + 1, zdir="y", color="green")
