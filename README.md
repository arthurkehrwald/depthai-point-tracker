# DepthAI Point Tracker

Tracks a single IR LED in 3D using a Luxonis Oak-D stereo camera.

## Notes
- Using DepthAI v2 instead of v3, because v3 had higher latency with cropped 720p stereo images at 100 FPS.
- At least in v2, the ImageManip node stretches the image horizontally by one pixel when no transformations are applied. This can lead to issues with triangulation because the images are no longer rectified correctly.
## TODO
- Prevent user from crashing app with invalid settings like blob max size < min size