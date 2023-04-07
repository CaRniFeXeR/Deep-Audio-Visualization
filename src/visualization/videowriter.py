from pathlib import Path
import cv2


class VideoWriter:

    def __init__(self, img_folder: Path, video_name: str, fps: float) -> None:
        self.img_folder = img_folder
        self.video_name = video_name
        self.fps = fps
        print(fps)

    def write_video_file(self):
        video = None
        for img_file in self.img_folder.iterdir(): #todo ensure order
            if img_file.is_file() and img_file.name.endswith("png"):
                frame = cv2.imread(str(img_file))
                if video is None:
                    video = cv2.VideoWriter(str(self.img_folder / Path(self.video_name)), 0, self.fps, (frame.shape[1], frame.shape[0]))  # width height switched
                else:
                    video.write(cv2.imread(str(img_file)))

        cv2.destroyAllWindows()
        video.release()
        print(f"sucsessfully wrote video to folder {self.img_folder}")
