# DepthAI Point Tracker

Tracks a single IR LED in 3D using a Luxonis Oak-D stereo camera.

## Coordinate spaces

Calculations are done in OpenCV space from the point of view of the left camera lens (X goes right, Y goes down, Z goes
forward). The Y axis is flipped to face up for visualization and output.

## Why cropping didn't work out

The idea was to save bandwidth by cropping the image on the camera to the region around the LED detected in previous
frames before transmitting it to the computer. This allows for higher framerates (100 instead of 75) and lower latency
(14 ms instead of 20). Unfortunately, it is not practical:

- Need to use DepthAI v2 instead of v3, because v3 has higher latency with cropped 720p stereo images at 100 FPS,
  negating the savings from cropping.
- It is a bad idea to set the cropping region every frame, because it increases latency.
- When a cropped frame arrives, it is difficult to determine exactly where it was cropped from. If the position of the
  cropping window constantly changes, there is no way to tell if the latest window was applied or one of the previous
  ones.
- To solve these two problems, I implemented a fixed number of static tracking regions, each with a unique, slightly
  different size. This allows the origin of a cropped frame in the larger frame to be identified purely based on its
  width and height, but the code got much more complicated than it should have been.
- When tracking is lost, the whole image must be searched for the LED. For that, cropping must be turned off. When the
  camera is set to run at 720p@100fps, removing the crop causes the actual framerate to drop to ~55 and the latency to
  increase to ~47 ms. The target framerate and resolution can't be changed at runtime to remedy this. Additionally, at
  least in v2, the ImageManip node used for cropping stretches the image horizontally by one pixel when no
  transformations are applied. This leads to issues with triangulation because the images are no longer rectified
  correctly. So whenever tracking is lost, you get lag-spikes and the tracked position jumps around erratically.
- If the tracking system accidentally locks onto static background reflection, it will crop the frame to that area and
  completely ignore the real LED, even if it is right in front of the camera.