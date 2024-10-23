# scene-detect

Tiny commandline tool to perform scene detection on a video
and automatically extract the first frame of every scene.

# Installation

Grab the [latest release](https://github.com/soup-dog/scene-detect/releases/latest) as a wheel.

```commandline
pip install scene_detect-X.X.X-py3-none-any.whl
```

# Usage

Basic usage:

```commandline
scene-detect -i video.mp4 ./frames
```

Downscale video for faster processing:

```commandline
scene-detect -i video.mp4 -r 180 90 ./frames
```

Perform scene detection on one video, but extract 
frames from another (e.g. if you have a video 
optimised for scene detection):

```commandline
scene-detect -i scene.mp4 -s source.mp4 ./frames
```

Emit scene detection data (heuristic score, frame index, time):

```commandline
scene-detect -i video.mp4 -k ./frames
```
