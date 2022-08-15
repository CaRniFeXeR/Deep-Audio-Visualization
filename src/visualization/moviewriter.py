from pathlib import Path
from typing import Callable
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.VideoClip import DataVideoClip
from moviepy.editor import AudioFileClip

class MovieWriter:

    def __init__(self, frame_fnc: Callable[[int], None], outdir: Path, video_name: str, n_frames: int, fps: float, audio_file_path : Path = None) -> None:
        self.video_name = video_name
        self.outdir = outdir
        self.frame_fnc = frame_fnc
        self.n_frames = n_frames
        self.fps = fps
        self.audio_file_path = audio_file_path
        print(fps)

    def write_video_file(self):
        video = DataVideoClip(list(range(12, self.n_frames)), self.frame_fnc, fps=self.fps)
        if self.audio_file_path is not None:
            print(f"adding audio track  {self.audio_file_path} to the video")
            audio_clip = AudioFileClip(str(self.audio_file_path))
            audio_clip = audio_clip.subclip(12/self.fps, video.end) #amount of seconds until audio starts
            video = video.set_audio(audio_clip)

        video.write_videofile(str(self.outdir / Path(self.video_name)), fps=self.fps)
        print(f"sucsessfully wrote video to folder {self.outdir}")
