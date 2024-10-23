import json
import os
from typing import *

from .util import TempFile

from moviepy.video.io.VideoFileClip import VideoFileClip, VideoClip
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm
import click
import ffmpeg


class DeltaResult:
    def __init__(self, delta: int, index: int):
        self.delta = delta
        self.index = index


def posterise(r: int, g: int, b: int, bit_depth: int = 8) -> int:
    mask_shift = (bit_depth - 5)
    mask = 0b11111 << mask_shift
    return (
            (((mask & r) >> mask_shift) << (16 - 5))
            | (((mask & g) >> mask_shift) << (16 - 10))
            | (((mask & b) >> mask_shift) << (16 - 15))
    )


def posterise_frame(frame: NDArray) -> NDArray:
    bit_depth = frame.dtype.itemsize * 8

    posterised = np.empty(frame.shape[:2], dtype=np.int16)
    for row in range(frame.shape[0]):
        for col in range(frame.shape[1]):
            r, g, b = frame[row, col]
            posterised[row, col] = posterise(r, g, b, bit_depth)

    return posterised


def histogram(frame: NDArray) -> Dict[np.int16, np.int64]:
    unique, counts = np.unique(posterise_frame(frame), return_counts=True)

    hist = {}

    for i in range(unique.shape[0]):
        hist[unique[i]] = counts[i]

    return hist


def delta(last: Dict[np.int16, np.int64], current: Dict[np.int16, np.int64]):
    # last_hist = np.unique(last, return_counts=True)
    # current_hist = np.unique(last, return_counts=True)
    # diff = 0
    # return
    diff = 0
    for k in current:
        try:
            diff += abs(last[k] - current[k])
        except KeyError:
            diff += current[k]
    return diff
    # return np.sum(np.abs(last - current))


def get_keyframes(clip: VideoClip, total_frames: int = None, threshold: float = 1) -> List[DeltaResult]:
    if total_frames is None and isinstance(clip, VideoFileClip):
        total_frames = clip.reader.nframes

    key_frames: List[DeltaResult] = []
    last = None
    last_histogram: Union[Dict[np.int16, np.int64], None] = None
    i = 0
    for frame in tqdm(clip.iter_frames(), total=total_frames, unit="frame", miniters=30):
        hist = histogram(frame)

        if last is None:
            key_frames.append(DeltaResult(-1, 0))  # first frame is always a scene start
        else:
            key_frames.append(DeltaResult(delta(last_histogram, hist) / (frame.shape[0] * frame.shape[1]), i))

        last = frame
        last_histogram = hist
        i += 1

    return [frame for frame in key_frames if frame.delta < 0 or frame.delta > threshold]


def try_parse_vector(s: str) -> Union[Tuple[int, int], None]:
    try:
        x_str, y_str = s.split(":")

        x = -1 if x_str == "" else int(x_str)
        y = -1 if y_str == "" else int(y_str)

        if x == -1 and y == -1:
            return None

        return int(x), int(y)
    except ValueError:
        return None


def resize(source_path: str, x: int, y: int, dest_path: str, overwrite: bool = False, quiet: bool = True) -> str:
    if not overwrite and os.path.exists(dest_path):
        raise FileExistsError(f"File {dest_path} already exists.")

    (
        ffmpeg
        .input(source_path)
        .filter("scale", x, y)
        .output(dest_path, loglevel="error" if quiet else "verbose")
        .run(overwrite_output=True)
    )

    return dest_path


@click.command()
@click.argument("output")
@click.option("-i", "--in-video",
              prompt="Input video", help="Path to the video to run scene detection on.",
              type=click.Path(exists=True, dir_okay=False),
              required=True)
@click.option("-r", "--resize", "resize_shape",
              help="The dimensions to resize the video to for scene detection. Lower is faster, but has reduced accuracy.",
              type=int, nargs=2,
              default=None)
@click.option("-s", "--source-video",
              help="Path to the video to extract scene start frames from. Defaults to the scene detection video.",
              type=click.Path(exists=True, dir_okay=False),
              default=None)
@click.option("-k", "--emit-keyframes",
              help="Emit scene detection information, including scene detection heuristic score, frame index and time.",
              type=click.Choice(["yes", "no", "file"]),
              default="no")
@click.option("-v", "--verbose",
              help="Set to produce detailed logs.",
              is_flag=True,
              default=False)
@click.option("-t", "--threshold",
              help="Scene detection threshold. Lower values detect more scenes. 0.65-0.75 is a good starting range.",
              type=float,
              default=0.7)
def scene_detect(output, in_video, resize_shape, source_video, emit_keyframes, verbose, threshold):
    """
    Run scene detection on a video and extract the scene start frames into an OUTPUT folder.
    """
    if source_video is None:
        source_video = in_video

    # resize_shape = try_parse_vector(resize_shape_arg)

    with TempFile(extension=".mp4") as resized_video_path:
        if resize_shape is not None:
            try:
                print("Resizing...", end="", flush=True)
                in_video = resize(in_video,
                                  resize_shape[0], resize_shape[1],
                                  resized_video_path,
                                  overwrite=False, quiet=not verbose)
                print("\x1b[92mDONE\x1b[0m")
            except ffmpeg.Error as e:
                print("\x1b[31mffmpeg failed to resize. Aborting!\x1b[0m")
                exit(1)

        # scene detection
        with VideoFileClip(in_video) as clip:
            in_frame_count = clip.reader.nframes
            in_fps = clip.fps

            print("Finding keyframes..")
            keyframes = get_keyframes(clip, in_frame_count, threshold)
            print("\x1b[92mDONE\x1b[0m")

            # key_frames.sort(key=lambda x: x.delta, reverse=True)

            # print([f"{x.delta} {x.index} {x.index / clip.fps}" for x in key_frames[:10]])

        if emit_keyframes == "file":
            with open("keyframes.json", "w") as f:
                json.dump({"keyframes": keyframes}, f)
        elif emit_keyframes == "yes":
            print("KEYFRAMES\n----------------")
            print("SCORE FRAME TIME")
            for f in keyframes:
                print(f"{f.delta} {f.index} {f.index / in_fps:.1f}s")

        # extract scene start frames
        with VideoFileClip(source_video) as clip:
            if clip.reader.nframes != in_frame_count:
                print("\x1b[33mInput video and source video frame count do not match!\x1b[0m")
            if clip.reader.fps != in_fps:
                print("\x1b[31mInput video and source video fps do not match! Extraction will probably fail!\x1b[0m")

            os.makedirs(output, exist_ok=True)

            if len(os.listdir(output)) != 0:
                print("\x1b[33mOutput directory not empty!\x1b[0m")

            for i in range(len(keyframes)):
                frame = keyframes[i]
                s = frame.index / clip.fps
                Image.fromarray(clip.get_frame(s)).save(os.path.join(output, f"{i}_{s:.1f}.png"))


if __name__ == '__main__':
    scene_detect()
