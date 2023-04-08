from pathlib import Path

import numpy as np
from src.preprocessing.pitchshifter import PitchShifter
from src.datastructures.featureextractionconfig import FeatureExtractionConfig
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from src.datastructures.trackfeatures import TrackFeatures
from src.io.trackfeaturesfilehandler import TrackFeaturesFileHandler
import librosa
import librosa.display

class FeatureExtractor:
    """
    extracts the needed features from a given track and saves them in a specified folder
    """

    def __init__(self, config: FeatureExtractionConfig) -> None:
        self.config = config

    def generate_centroid(self, file_path : str) -> np.ndarray:
        samples, sample_rate = librosa.load(file_path)
        centroids = librosa.feature.spectral_centroid(y= samples, sr=sample_rate)[0]

        return centroids

    def generate_pitchshifted_features(self, wavefile_path : Path):
        assert self.config.pitchshifting is not None

        pitchsshifter = PitchShifter(self.config.pitchshifting)
        pitch_shifted_files = pitchsshifter.range_pitch(wavefile_path)
        for pitch_shifted_file in pitch_shifted_files:
            self.extract_features(pitch_shifted_file)



    def extract_features(self, wavfile_path: Path):
        sample_rate, track_data = wavfile.read(wavfile_path)

        assert sample_rate == 44100, "wave file should be rendered at samplerate=44100 Hz"
        assert len(track_data.shape) == 1 or (len(track_data.shape) == 2 and track_data.shape[1] == 2), "wave file is not mono or stero sound"

        if len(track_data.shape) > 1 and track_data.shape[1] == 2:
            track_data[:, 0] += track_data[:, 1]
            track_data[:, 0] = track_data[:, 0] / 2

        # track_data = track_data[:, 0]
        duration_s = track_data.shape[0] / sample_rate
        N_overlap = np.ceil(self.config.window_size*self.config.percent_overlap)

        window = signal.windows.tukey(self.config.window_size, alpha=0.4)
        F, T, S_mag = signal.spectrogram(track_data, sample_rate, scaling="spectrum", mode="magnitude", window=window, noverlap=N_overlap)

        if self.config.use_mel_spec:
            # S_mag = convert_spectogram_to_mel(F, len(T), S_mag)
            # S_mag = convert_spectogram_to_mel(S_mag_org, given_power_spec = False)
            # mel_signal = librosa.feature.melspectrogram(y=track_data, sr=sample_rate, hop_length=N_overlap)
            # spectrogram = np.abs(mel_signal)
            samples, sample_rate = librosa.load(wavfile_path, sr=44100)
            S_mag = librosa.feature.melspectrogram(samples, sr = sample_rate, win_length =self.config.window_size, window= window)

        if self.config.use_db_scale:
            S_mag = librosa.power_to_db(S_mag, ref=np.max)

        time_resolution = np.floor(len(T)/duration_s)  # Spectrogram time bins per sec
        seconds_per_window = self.config.window_size/sample_rate
        print(f"seconds per window: {seconds_per_window}")
        print(f"spectrogram time bins per sec: {time_resolution}")
        print(f"full spectrogram shape : {S_mag.shape}")

        dt = 1  # number of spectrogram time bins between consecutive slices of the input tensor. finest possible interval is 1
        dt_secs = dt/time_resolution  # time interval between consecutive slices in the input tensor
        print(dt)

        time_resolution = np.floor(len(T)/duration_s)  # Spectrogram time bins per sec
        seconds_per_window = self.config.window_size/sample_rate

        frame_width = np.argmin(np.abs(self.config.secs_per_spectrum-T))+1  # number of spectrogram time bins

        if not self.config.use_mel_spec:
            # for cropping out lower frequencies like drums
            f_max = 4000  # max frequency (Hz) to include in spectrogram. dont change
            f_min = 0  # set to 0 if not cropping lower freq
            f_max_idx = np.argmin(np.abs(f_max-F))+1  # number of frequency bins
            f_min_idx = np.argmin(np.abs(f_min-F))  # go inspect fig to check if this works out

            S_mag_save = S_mag[f_min_idx:f_max_idx+f_min_idx, :]  # crop high and low freqs
        else:
            S_mag_save = S_mag[:94,:]
            f_min_idx = 0
            f_max_idx = 94
        
        F = F[f_min_idx:f_max_idx+f_min_idx]
        frame_height = S_mag_save.shape[0]

            

        # better if dimensions are even for Decoder network
        if frame_width % 2 == 1:
            frame_width += 1

        frame_size = (frame_width, frame_height)
        print(f"spectrogram frame shape (time bins, freq bins): {frame_size}")
        # assert frame_height == 94, "Input image dimensions will cause shape error in decoder"
        mel_indicator = "_mel" if self.config.use_mel_spec else ""
        db_indicator = "_db" if self.config.use_db_scale else ""
        outfolder = self.config.outputlocation / Path(wavfile_path.name + f"_{self.config.secs_per_spectrum}s{mel_indicator}{db_indicator}")
        outfolder.mkdir(exist_ok=True, parents=True)
        track_features = TrackFeatures(F=F, T=T, S_mag=S_mag_save, dt=dt, frame_width=frame_width, frame_height=frame_height, time_resolution=time_resolution)

        if self.config.calculate_centroids:
            track_features.centroids = self.generate_centroid(wavfile_path)

        TrackFeaturesFileHandler().save_track_features(outfolder / Path('vars.npz'), track_features)

     
    def _plot_spectogram(self, T : np.ndarray, S : np.ndarray, F : np.ndarray, frame_width: int, frame_height: int, outfolder: Path):
        c = 26 * 5
        plt.figure(figsize=(20, 7))
        ax = plt.axes()
        # plt.pcolormesh(T[:frame_width], F[:frame_height], S_mag[:frame_height, :frame_width], shading='nearest', cmap='inferno')
        plt.pcolormesh(T[:frame_width*c], F[:frame_height], S[:frame_height, :frame_width*c], shading='nearest', cmap='inferno')

        ax.set_yscale('linear')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        cbar = plt.colorbar(ax=ax)
        cbar.set_label('Amplitude (dB)')
        plt.savefig(outfolder / Path('spectrogram.png'))
        plt.show()
