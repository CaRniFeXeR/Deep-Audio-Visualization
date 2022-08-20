import numpy as np
from skimage.measure import block_reduce

class EqualizerVisHandler():

    def __init__(self, embedded: np.ndarray, s_mag_norm: np.ndarray, n_bins : int) -> None:
        self.max_embedded = embedded.max(axis=0)
        self.min_embedded = embedded.min(axis=0)
        self.embeded_range = self.max_embedded - self.min_embedded
        print(self.embeded_range)
        self.s_mag_norm = s_mag_norm * self.embeded_range[2] * 1.5
        self.s_mag_norm = block_reduce(self.s_mag_norm,block_size=(6,1), func = np.average)
        self.n_bins = self.s_mag_norm.shape[0]

    def set_axis_scale(self, ax):
        ax.set_xlim(self.min_embedded[0], self.max_embedded[0])
        ax.set_ylim(self.min_embedded[1], self.max_embedded[1])
        ax.set_zlim(self.min_embedded[2], self.max_embedded[2])

    def render_equalizer_bar(self, t: int, ax):

        bin_pos = self.min_embedded[0] + np.arange(self.n_bins) / self.n_bins * self.embeded_range[0]
        bin_height = self.s_mag_norm[:, t]
        bin_width = self.embeded_range[0] / (self.n_bins + 1)
        bottom = self.min_embedded[2]
        ax.bar(bin_pos, bin_height, width = bin_width, bottom = bottom, zs=self.min_embedded[1] + 1, zdir="y", color="green")
