import collections
import math
import sys
import json
import socket
import numpy as np
import cv2
import depthai as dai
import typing
from pathlib import Path
from dataclasses import dataclass
from types import TracebackType

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

CONFIG_FILE = "config.json"
CAMERA_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_720_P
RESOLUTION_MAP = {
    dai.MonoCameraProperties.SensorResolution.THE_800_P: (1280, 800),
    dai.MonoCameraProperties.SensorResolution.THE_720_P: (1280, 720),
    dai.MonoCameraProperties.SensorResolution.THE_400_P: (640, 400)
}
CAMERA_RESOLUTION_NUMERIC = RESOLUTION_MAP[CAMERA_RESOLUTION]
CAMERA_FPS = 100
# How far ahead (in seconds) the crop position is predicted, to compensate for
# the time it takes for a crop change to take effect.
CROP_PREDICTION_LOOKAHEAD_S = 0
# Above this speed (in cm/s) a 3D position change is considered too fast to
# plausibly be the head of a person, and lowers the temporal confidence. The
# speed is estimated via a smoothed (least-squares) fit over the whole window
# rather than raw frame-to-frame differences, so it is not dominated by
# per-frame measurement noise.
MAX_PLAUSIBLE_HUMAN_SPEED_CM_S = 600.0
# Above this RMS deviation (in cm) of the actual positions from the smoothed
# (least-squares fit) trajectory, the motion is considered too erratic
# (jumpy/inconsistent) to plausibly be a person's head, as opposed to e.g. a
# mismatched detection jumping between unrelated candidates.
MAX_PLAUSIBLE_HUMAN_JERK_CM = 6.0
# Below this standard deviation (in cm) of the 3D position over the sliding
# window, the detection is considered to be a static background element (e.g.
# a reflection) rather than a tracked, moving LED, and lowers the temporal
# confidence. It is small enough that the natural sway/jitter of a person
# standing still is not mistaken for a static background element.
MIN_PLAUSIBLE_HUMAN_JITTER_CM = 0.04
# The number of consecutive frames without a detection after which the
# temporal confidence drops to zero.
MAX_CONSECUTIVE_MISSES = 100
STEREO_MATCH_CONF_THRESHOLD = .5


def load_config():
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "exposure": 200,
        "iso": 100,
        "threshold": 0.9,
        "blob_min_threshold": 80,
        "blob_max_threshold": 255,
        "blob_min_area": 20,
        "blob_max_area": 1000,
        "blob_min_circularity": 0.7,
        "blob_min_convexity": 0.9,
        "blob_min_inertia": 0.6
    }


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


@dataclass(frozen=True)
class CameraSocketParams:
    projection: np.ndarray
    rectify_map_x: np.ndarray
    rectify_map_y: np.ndarray


@dataclass(frozen=True)
class Frame:
    cropped: np.ndarray
    uncropped: np.ndarray
    frame_time: float
    time_of_capture: float
    time_of_arrival: float


@dataclass(frozen=True)
class CropRect:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class DetectionRecord:
    """A single matched detection, as produced by ``match_candidates``.

    ``point_l``/``point_r`` are the 2D detection positions (in full-frame image
    pixels, origin at top-left) in the left/right camera images, ``timestamp``
    is the capture time of the frame pair, ``pos_3d`` is the triangulated 3D
    position, and ``confidence`` is the match confidence.
    """
    point_l: typing.Tuple[float, float]
    point_r: typing.Tuple[float, float]
    timestamp: float
    pos_3d: np.ndarray
    confidence: float


class DetectionHistory:
    """Keeps a sliding window of recent detections and predicts where the
    tracked point will be in each camera image a short time ahead.

    The prediction is a linear extrapolation (least-squares fit of position
    over time) of the recent 2D detections in each camera, which reacts faster
    to a moving target than simply using the latest detected position.
    """

    def __init__(self, max_len: int = 20):
        self.records: typing.Deque[DetectionRecord] = collections.deque(maxlen=max_len)
        self.num_consecutive_misses = 0

    def add(self, record: DetectionRecord) -> None:
        self.records.append(record)
        self.num_consecutive_misses = 0

    def record_miss(self) -> None:
        self.num_consecutive_misses += 1

    def clear(self) -> None:
        self.records.clear()
        self.num_consecutive_misses = 0

    def predict(
            self, lookahead: float
    ) -> typing.Optional[typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]:
        """Predict ``(point_l, point_r)`` ``lookahead`` seconds after the most
        recent detection, based on the recorded history. Returns ``None`` if
        there is no history yet.
        """
        if not self.records or self.num_consecutive_misses > 20:
            return None
        if len(self.records) == 1 or CROP_PREDICTION_LOOKAHEAD_S == 0.0:
            record = self.records[-1]
            return record.point_l, record.point_r

        times = np.array([r.timestamp for r in self.records])
        target_time = times[-1] + lookahead

        def fit_predict(values: np.ndarray) -> float:
            # Linear least-squares fit: values = a * t + b
            a, b = np.polyfit(times, values, 1)
            return float(a * target_time + b)

        xs_l = np.array([r.point_l[0] for r in self.records])
        ys_l = np.array([r.point_l[1] for r in self.records])
        xs_r = np.array([r.point_r[0] for r in self.records])
        ys_r = np.array([r.point_r[1] for r in self.records])

        point_l = (fit_predict(xs_l), fit_predict(ys_l))
        point_r = (fit_predict(xs_r), fit_predict(ys_r))
        return point_l, point_r

    def compute_temporal_confidence(self) -> float:
        """Estimate, as a value in ``[0, 1]``, how likely it is that the LED
        attached to the tracked person's head is currently being tracked
        correctly, based on the sliding window of recent detections.

        This combines three signals:
          * The match confidence of each record, biased towards more recent
            detections (an exponentially weighted average).
          * The plausibility of the implied 3D motion: since the LED is worn
            on a person's head, both moving faster than a person's head
            plausibly can, and moving erratically/inconsistently (as opposed
            to a smooth trajectory), indicate a wrong match rather than a
            real, tracked person. The motion is estimated with a smoothed
            (least-squares) fit over the whole window rather than raw
            frame-to-frame differences, so that ordinary per-frame
            measurement noise is not mistaken for implausible motion.
          * The temporal variance of the 3D position: a real, tracked person
            (even standing still) exhibits some jitter/sway, while a nearly
            perfectly static position over the whole window is a sign that a
            static background element (e.g. a reflection) is being tracked
            instead.
        """
        if not self.records:
            return 0.0

        n = len(self.records)
        confidences = np.array([r.confidence for r in self.records])

        # Exponentially weighted average, biased towards the most recent
        # detections.
        weights = np.array([2.0 ** i for i in range(n)])
        weights /= weights.sum()
        weighted_confidence = float(np.dot(weights, confidences))

        if n < 3:
            # Not enough history to reliably judge motion plausibility or
            # temporal variance; fall back to the confidence-only estimate,
            # still penalized by recent misses.
            miss_penalty = logistic_interpolation(
                self.num_consecutive_misses, ideal=0, cutoff=MAX_CONSECUTIVE_MISSES
            )
            return max(0.0, min(1.0, weighted_confidence * miss_penalty))

        positions = np.array([r.pos_3d for r in self.records])
        times = np.array([r.timestamp for r in self.records])

        # Fit a smooth (constant-velocity) trajectory to the window, per axis,
        # so that speed/erraticness are not dominated by noise between any two
        # individual frames.
        fitted = np.empty_like(positions)
        velocity = np.empty(3)
        for axis in range(3):
            a, b = np.polyfit(times, positions[:, axis], 1)
            velocity[axis] = a
            fitted[:, axis] = a * times + b

        # Motion plausibility: the smoothed speed should stay within what a
        # person's head can plausibly achieve.
        speed = float(np.linalg.norm(velocity))
        speed_plausibility = logistic_interpolation(
            speed, ideal=0.0, cutoff=MAX_PLAUSIBLE_HUMAN_SPEED_CM_S
        )

        # Erraticness: large deviations of the actual positions from the
        # smoothed trajectory indicate jumpy, inconsistent motion (e.g. a
        # detection jumping between unrelated candidates) rather than a
        # person's continuous movement.
        jerk_rms = float(np.sqrt(np.mean(np.sum((positions - fitted) ** 2, axis=1))))
        smoothness_plausibility = logistic_interpolation(
            jerk_rms, ideal=0.0, cutoff=MAX_PLAUSIBLE_HUMAN_JERK_CM
        )

        # Static-background detection: very little variance around the mean
        # position over the window suggests a static element rather than a
        # (possibly still, but never perfectly static) tracked person.
        mean_pos = positions.mean(axis=0)
        jitter_std = float(np.std(np.linalg.norm(positions - mean_pos, axis=1)))
        not_static = logistic_interpolation(
            jitter_std, ideal=MIN_PLAUSIBLE_HUMAN_JITTER_CM, cutoff=0
        )

        miss_penalty = logistic_interpolation(
            self.num_consecutive_misses, ideal=0, cutoff=MAX_CONSECUTIVE_MISSES
        )

        temporal_confidence = (
            weighted_confidence *
            speed_plausibility *
            smoothness_plausibility *
            not_static *
            miss_penalty
        )
        return max(0.0, min(1.0, temporal_confidence))


class Cropper:
    """Produces a finite set of fixed crop regions and reverses the crop.

    ``get_crop_rect`` picks, for a detected point, the crop region whose center
    is closest to that point. ``uncrop`` takes an image that the camera cropped
    with one of these regions and places it back where it was captured, padding
    the rest with black pixels so the result has the full frame resolution.

    The core difficulty is that the region has to be recovered from *only* the
    width and height of the returned image, because the crop instruction that
    the camera actually applied is not known when the frame arrives. This is
    solved by giving every region a unique size, so a returned image's size
    identifies the region that produced it.

    The regions are laid out on an equally spaced grid. Because the tracked
    object moves continuously across the frame, adjacent regions overlap (by
    ``overlap`` of their extent, one third by default) so the object is never
    cut in half between two neighbours. The camera does not pad a crop that
    extends past the image edges (which would yield an unexpected size), so the
    regions are kept inside the frame; the outermost ones may stop a few pixels
    short because of their slightly different sizes.
    """

    def __init__(self, resolution: typing.Tuple[int, int], cols: int = 8, rows: int = 6,
                 overlap: float = 1.0 / 2.0, size_step: int = 4, margin: int = 4):
        if cols < 1 or rows < 1:
            raise ValueError("cols and rows must be >= 1")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        self.width, self.height = resolution
        self.cols = cols
        self.rows = rows
        self.overlap = overlap
        # Ensure size_step is a multiple of 4 so that all generated region
        # widths and heights are also multiples of 4.
        self.size_step = max(4, (size_step + 3) // 4 * 4)
        self.margin = margin

        # The largest region along each axis spans the full usable extent; every
        # other region is smaller by a multiple of ``size_step`` so that all
        # widths (per column) and heights (per row) - and therefore all
        # (width, height) pairs - are distinct.
        nominal_w = int((self.width - margin) / (1 + (cols - 1) * (1 - overlap))) // 4 * 4
        nominal_h = int((self.height - margin) / (1 + (rows - 1) * (1 - overlap))) // 4 * 4
        base_w = nominal_w - (cols - 1) * size_step
        base_h = nominal_h - (rows - 1) * size_step
        if base_w <= 0 or base_h <= 0:
            raise ValueError("Too many regions (or size_step too large) for this resolution")

        # Equal spacing between region centers, derived from the nominal size so
        # neighbouring regions overlap by ``overlap`` of that size.
        step_x = nominal_w * (1 - overlap)
        step_y = nominal_h * (1 - overlap)
        start_x = margin / 2.0 + nominal_w / 2.0
        start_y = margin / 2.0 + nominal_h / 2.0

        self.regions: typing.List[CropRect] = []
        self.centers: typing.List[typing.Tuple[float, float]] = []
        self._by_size: typing.Dict[typing.Tuple[int, int], CropRect] = {}
        for j in range(rows):
            for i in range(cols):
                w = base_w + i * size_step
                h = base_h + j * size_step
                cx = start_x + i * step_x
                cy = start_y + j * step_y
                x_min = int(round(cx - w / 2.0))
                y_min = int(round(cy - h / 2.0))
                x_max = x_min + w
                y_max = y_min + h
                # Defensive clamp: never let a region exceed the frame, keeping
                # its size (and thus its identity) intact by shifting inwards.
                if x_max > self.width:
                    x_min -= x_max - self.width
                    x_max = self.width
                if y_max > self.height:
                    y_min -= y_max - self.height
                    y_max = self.height
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                rect = CropRect(x_min, y_min, x_max, y_max)
                self.regions.append(rect)
                self.centers.append((cx, cy))
                self._by_size[(w, h)] = rect

    def get_crop_rect(self, x: float, y: float) -> CropRect:
        """Return the crop region whose center is closest to ``(x, y)``.

        Coordinates use the image convention: the origin is at the top-left,
        ``x`` grows to the right and ``y`` downward, in pixels of the full frame.
        """
        best_index = min(
            range(len(self.centers)),
            key=lambda k: (self.centers[k][0] - x) ** 2 + (self.centers[k][1] - y) ** 2,
        )
        return self.regions[best_index]

    def uncrop(self, cropped: np.ndarray) -> np.ndarray:
        """Place a cropped image back where it was captured, padding with black.

        The originating region is recovered solely from the width and height of
        ``cropped``. The returned image has the full frame resolution.
        """
        uncropped = np.zeros((self.height, self.width), dtype=cropped.dtype)
        h, w = cropped.shape[:2]
        rect = self._region_for_size(w, h)
        # Anchor at the recovered region's top-left, or the frame's top-left when
        # the size is unknown (e.g. a full/unset frame).
        x0 = rect.x_min if rect is not None else 0
        y0 = rect.y_min if rect is not None else 0
        # Clip to the frame so a size that differs slightly (camera rounding)
        # from the region cannot overflow the destination.
        y1 = min(y0 + h, self.height)
        x1 = min(x0 + w, self.width)
        uncropped[y0:y1, x0:x1] = cropped[0:y1 - y0, 0:x1 - x0]
        return uncropped

    def _region_for_size(self, w: int, h: int) -> typing.Optional[CropRect]:
        rect = self._by_size.get((w, h))
        if rect is not None:
            return rect
        # A full (crop unset) frame is not one of the regions.
        if w >= self.width and h >= self.height:
            return None
        # Tolerate small rounding differences from the camera by matching the
        # closest known size, but only when it is clearly the best match.
        best_rect = None
        best_dist = None
        for (rw, rh), candidate in self._by_size.items():
            dist = (rw - w) ** 2 + (rh - h) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_rect = candidate
        if best_dist is not None and best_dist < (self.size_step ** 2):
            return best_rect
        return None


class MonoCamera:
    def __init__(self, pipeline: dai.Pipeline, socket: dai.CameraBoardSocket, name: str, sync: dai.node.Sync,
                 resolution: dai.MonoCameraProperties.SensorResolution, fps: int):
        self.resolution = resolution
        self.numeric_resolution = RESOLUTION_MAP[resolution]
        self.prev_frame_arrival_time = -1.0
        self.cropper = Cropper(self.numeric_resolution)
        self.name = name

        cam = pipeline.create(dai.node.MonoCamera)
        assert isinstance(cam, dai.node.MonoCamera)
        cam.setBoardSocket(socket)
        cam.setFps(fps)
        cam.setResolution(resolution)

        manip = pipeline.create(dai.node.ImageManip)
        assert isinstance(manip, dai.node.ImageManip)
        manip.inputImage.setBlocking(False)
        manip.inputImage.setQueueSize(1)
        manip.inputConfig.setBlocking(False)
        manip.inputConfig.setQueueSize(1)
        # Allocate an output buffer large enough for the biggest possible output
        # (the full frame, produced when the crop is unset).
        manip.setMaxOutputFrameSize(self.numeric_resolution[0] * self.numeric_resolution[1])
        self.manip_ctrl = pipeline.create(dai.node.XLinkIn)
        assert isinstance(self.manip_ctrl, dai.node.XLinkIn)
        self.manip_ctrl_q_name = f"{name}_manip_control"
        self.manip_ctrl.setStreamName(self.manip_ctrl_q_name)
        self.manip_ctrl.out.link(manip.inputConfig)
        cam.out.link(manip.inputImage)

        manip.out.link(sync.inputs[name])
        sync.inputs[name].setBlocking(False)
        sync.inputs[name].setQueueSize(1)

        self.cam_ctrl = pipeline.create(dai.node.XLinkIn)
        assert isinstance(self.cam_ctrl, dai.node.XLinkIn)
        self.cam_ctrl_q_name = f"{name}_cam_control"
        self.cam_ctrl.setStreamName(self.cam_ctrl_q_name)
        self.cam_ctrl.out.link(cam.inputControl)

        self.current_crop = CropRect(0, 0, self.numeric_resolution[0], self.numeric_resolution[1])
        self.rect_map_x = None
        self.rect_map_y = None

    def set_rectification_maps(self, rect_map_x: np.ndarray, rect_map_y: np.ndarray):
        self.rect_map_x = rect_map_x
        self.rect_map_y = rect_map_y

    def process_frame(self, frame: dai.ImgFrame, arrival_time: float):
        frame_time_ms = (arrival_time - self.prev_frame_arrival_time) * 1000
        self.prev_frame_arrival_time = arrival_time
        capture_time = frame.getTimestamp().total_seconds()
        cv_frame = frame.getCvFrame()
        uncropped = self.uncrop(frame.getCvFrame())
        if self.rect_map_x is not None and self.rect_map_y is not None:
            rect_uncropped = cv2.remap(uncropped, self.rect_map_x, self.rect_map_y,
                                       cv2.INTER_LINEAR)
        else:
            print("WARNING: Rectification map not set. Will not rectify.")
            rect_uncropped = uncropped
        return Frame(cv_frame, rect_uncropped, frame_time_ms, capture_time, arrival_time)

    def uncrop(self, cropped: np.ndarray) -> np.ndarray:
        return self.cropper.uncrop(cropped)

    def set_crop_to_point(self, device: dai.Device, x: float, y: float):
        self.set_crop(device, self.cropper.get_crop_rect(x, y))

    def set_crop(self, device: dai.Device, rect: CropRect):
        if rect == self.current_crop:
            return
        self.current_crop = rect
        msg = dai.ImageManipConfig()
        x_min = rect.x_min / self.numeric_resolution[0]
        y_min = rect.y_min / self.numeric_resolution[1]
        x_max = rect.x_max / self.numeric_resolution[0]
        y_max = rect.y_max / self.numeric_resolution[1]
        msg.setCropRect(x_min, y_min, x_max, y_max)
        device.getInputQueue(self.manip_ctrl_q_name).send(msg)

    def unset_crop(self, device: dai.Device):
        self.set_crop(device, CropRect(0, 0, self.numeric_resolution[0], self.numeric_resolution[1]))

    def set_exposure(self, device: dai.Device, exp_time: int, sens_iso: int) -> None:
        msg = dai.CameraControl()
        msg.setManualExposure(exp_time, sens_iso)
        device.getInputQueue(self.cam_ctrl_q_name).send(msg)


class StereoCamera:
    def __init__(
            self, resolution: dai.MonoCameraProperties.SensorResolution, fps: int
    ):
        self.resolution = resolution
        self.numeric_resolution = RESOLUTION_MAP[resolution]
        self.fps = fps

    def __enter__(self) -> "StereoCamera":
        self.pipeline = dai.Pipeline()
        sync = self.pipeline.create(dai.node.Sync)
        assert isinstance(sync, dai.node.Sync)
        self.cam_l = MonoCamera(self.pipeline, dai.CameraBoardSocket.CAM_B, "left", sync, self.resolution, self.fps)
        self.cam_r = MonoCamera(self.pipeline, dai.CameraBoardSocket.CAM_C, "right", sync, self.resolution, self.fps)
        x_out_sync = self.pipeline.create(dai.node.XLinkOut)
        assert isinstance(x_out_sync, dai.node.XLinkOut)
        self.x_out_stream_name = "x_out"
        x_out_sync.setStreamName(self.x_out_stream_name)
        sync.out.link(x_out_sync.input)
        x_out_sync.input.setBlocking(False)
        x_out_sync.input.setQueueSize(1)
        self.device = dai.Device(self.pipeline)
        self.cam_params_l, self.cam_params_r = self.compute_stereo_rectification()
        self.cam_l.set_rectification_maps(self.cam_params_l.rectify_map_x, self.cam_params_l.rectify_map_y)
        self.cam_r.set_rectification_maps(self.cam_params_r.rectify_map_x, self.cam_params_r.rectify_map_y)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None,
                 exc_tb: TracebackType | None):
        self.device.close()

    def compute_stereo_rectification(self) -> typing.Tuple[CameraSocketParams, CameraSocketParams]:
        calibration = self.device.readCalibration()

        intrinsics_l = np.array(
            calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_B, self.numeric_resolution[0], self.numeric_resolution[1]
            ),
        )
        intrinsics_r = np.array(
            calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_C, self.numeric_resolution[0], self.numeric_resolution[1]
            ),
        )

        distortion_l = np.array(
            calibration.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B),
        )
        distortion_r = np.array(
            calibration.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C),
        )

        l_to_r_transformation = np.array(
            calibration.getCameraExtrinsics(
                dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
            )
        )
        l_to_r_rotation = l_to_r_transformation[:3, :3]
        l_to_r_translation = l_to_r_transformation[:3, 3:4]

        rotation_l, rotation_r, projection_l, projection_r, _, _, _ = cv2.stereoRectify(
            intrinsics_l, distortion_l.flatten(),
            intrinsics_r, distortion_r.flatten(),
            imageSize=self.numeric_resolution,
            R=l_to_r_rotation,
            T=l_to_r_translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        rectify_map_l_x, rectify_map_l_y = cv2.initUndistortRectifyMap(
            intrinsics_l, distortion_l,
            rotation_l, projection_l,
            self.numeric_resolution,
            cv2.CV_16SC2
        )
        rectify_map_r_x, rectify_map_r_y = cv2.initUndistortRectifyMap(
            intrinsics_r, distortion_r,
            rotation_r, projection_r,
            self.numeric_resolution,
            cv2.CV_16SC2
        )

        return (CameraSocketParams(projection_l, rectify_map_l_x, rectify_map_l_y),
                CameraSocketParams(projection_r, rectify_map_r_x, rectify_map_r_y))

    def get_stereo_frames(self) -> typing.Tuple[Frame, Frame]:
        message_group = self.device.getOutputQueue(self.x_out_stream_name).get()
        arrival_time = dai.Clock.now().total_seconds()
        assert isinstance(message_group, dai.MessageGroup)
        raw_frame_l = message_group["left"]
        raw_frame_r = message_group["right"]
        assert isinstance(raw_frame_l, dai.ImgFrame) and isinstance(raw_frame_r, dai.ImgFrame)
        frame_l = self.cam_l.process_frame(raw_frame_l, arrival_time)
        frame_r = self.cam_r.process_frame(raw_frame_r, arrival_time)
        return frame_l, frame_r

    def triangulate(
            self,
            point_l: typing.Tuple[float, float],
            point_r: typing.Tuple[float, float],
    ) -> np.ndarray:
        # cv.triangulatePoints operates on 2xN arrays of points
        points_l = np.array(point_l).reshape(2, 1)
        points_r = np.array(point_r).reshape(2, 1)
        points4d: np.ndarray = cv2.triangulatePoints(self.cam_params_l.projection, self.cam_params_r.projection,
                                                     points_l, points_r)
        first = points4d[:, 0]
        first = first[:3] / first[3]  # homogenous -> cartesian
        return first

    def set_exposure(self, exp_time: int, sens_iso: int) -> None:
        self.cam_l.set_exposure(self.device, exp_time, sens_iso)
        self.cam_r.set_exposure(self.device, exp_time, sens_iso)

    def set_crop(self, rect_l: CropRect, rect_r: CropRect):
        self.cam_l.set_crop(self.device, rect_l)
        self.cam_r.set_crop(self.device, rect_r)

    def unset_crop(self):
        self.cam_l.unset_crop(self.device)
        self.cam_r.unset_crop(self.device)

    def track_crop(self, found_l: bool, x_l: float, y_l: float,
                   found_r: bool, x_r: float, y_r: float):
        # Coordinates are in full-frame image pixels (origin at top-left). When a
        # point is detected the crop follows it; otherwise the crop is released
        # to the full frame so the target can be re-acquired.
        if found_l:
            self.cam_l.set_crop_to_point(self.device, x_l, y_l)
        else:
            self.cam_l.unset_crop(self.device)
        if found_r:
            self.cam_r.set_crop_to_point(self.device, x_r, y_r)
        else:
            self.cam_r.unset_crop(self.device)


class BlobDetector:
    def __init__(self, config):
        self.detector = None
        self.update_params(config)

    def update_params(self, config):
        params = cv2.SimpleBlobDetector.Params()
        params.minThreshold = config.get("blob_min_threshold", 80)
        params.maxThreshold = config.get("blob_max_threshold", 255)
        params.maxThreshold = max(params.maxThreshold, params.minThreshold)
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = config.get("blob_min_area", 20)
        params.maxArea = config.get("blob_max_area", 1000)
        params.maxArea = max(params.maxArea, params.minArea)
        params.filterByCircularity = True
        params.minCircularity = max(0.01, config.get("blob_min_circularity", 0.7))
        params.filterByConvexity = True
        params.minConvexity = max(0.01, config.get("blob_min_convexity", 0.9))
        params.filterByInertia = True
        params.minInertiaRatio = max(0.01, config.get("blob_min_inertia", 0.6))
        self.detector = cv2.SimpleBlobDetector.create(params)

    def detect_candidates(self, img: cv2.typing.MatLike) -> typing.List[typing.Tuple[float, float, float]]:
        keypoints = self.detector.detect(img)
        keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)[:10]
        return [(kp.pt[0], kp.pt[1], kp.size) for kp in keypoints]


def logistic_interpolation(
        val: float,
        ideal: float,
        cutoff: float,
        cutoff2: float | None = None,
        k: float = 6.0
) -> float:
    """
    Interpolates a float value smoothly between 1.0 (ideal) and 0.0 (cutoff).

    If cutoff2 is provided, the function creates a two-sided curve that transitions
    to 0.0 on both sides of the ideal value.
    """
    # If a second cutoff is given, pick which cutoff applies to the input value
    if cutoff2 is not None:
        # Determine which side of ideal the value falls on
        # Side A: cutoff, Side B: cutoff2
        if (ideal <= cutoff and val >= ideal) or (ideal >= cutoff and val <= ideal):
            active_cutoff = cutoff
        else:
            active_cutoff = cutoff2
    else:
        active_cutoff = cutoff

    if ideal == active_cutoff:
        raise ValueError("Ideal and active cutoff values cannot be equal.")

    # Relative normalized position: 0.0 at ideal, 1.0 at active_cutoff
    t = (val - ideal) / (active_cutoff - ideal)

    # Beyond boundary conditions
    if t <= 0.0:
        return 1.0
    if t >= 1.0:
        return 0.0

    # Map t in (0, 1) to sigmoid domain [-k, k]
    z = k * (1.0 - 2.0 * t)

    # Standard sigmoid using math module
    sigmoid = 1.0 / (1.0 + math.exp(-z))

    # Rescale to ensure clean 1.0 and 0.0 endpoints
    sig_k = 1.0 / (1.0 + math.exp(-k))
    sig_minus_k = 1.0 / (1.0 + math.exp(k))

    scaled_val = (sigmoid - sig_minus_k) / (sig_k - sig_minus_k)

    return max(0.0, min(1.0, scaled_val))


def match_candidates(candidates_l, candidates_r, stereo_cam: StereoCamera):
    best_match = None
    highest_confidence = 0

    for x_l, y_l, s_l in candidates_l:
        for x_r, y_r, s_r in candidates_r:
            # 1. Epipolar constraint: Corresponding points must have very similar Y-coordinates in image space
            y_pixel_diff = abs(y_l - y_r)

            # 2. Plausibility of 3D position
            pos_3d = stereo_cam.triangulate((x_l, y_l), (x_r, y_r))

            # 3. Also consider blob size similarity
            size_ratio = min(s_l, s_r) / max(s_l, s_r, 1e-6)

            # 4. And absolute size
            size = s_l + s_r

            weights = [10, 1, 1, 1, 1, 1]

            # Combined score using logistic interpolation
            conf = np.average([logistic_interpolation(y_pixel_diff, ideal=0, cutoff=3),
                               logistic_interpolation(pos_3d[0], ideal=0, cutoff=-200, cutoff2=200),
                               logistic_interpolation(pos_3d[1], ideal=0, cutoff=-100, cutoff2=100),
                               logistic_interpolation(pos_3d[2], ideal=100, cutoff=0, cutoff2=300),
                               logistic_interpolation(size_ratio, ideal=1, cutoff=0),
                               logistic_interpolation(size, ideal=20, cutoff=5)], weights=weights)

            if conf > highest_confidence and conf > STEREO_MATCH_CONF_THRESHOLD:
                highest_confidence = conf
                best_match = {
                    'left': (x_l, y_l, s_l),
                    'right': (x_r, y_r, s_r),
                    'pos_3d': pos_3d,
                    'confidence': highest_confidence
                }
    return best_match


class Worker(QtCore.QThread):
    frame_ready = QtCore.Signal(np.ndarray, np.ndarray)
    centroid_ready = QtCore.Signal(float, float, bool, float, float, bool, float, float, float, float, bool)
    position_ready = QtCore.Signal(np.ndarray)
    stats_ready = QtCore.Signal(float, float, float, float, float)
    confidence_ready = QtCore.Signal(float)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = True
        self.exposure = int(config.get('exposure', 200))
        self.iso = int(config.get('iso', 100))
        self.blob_params = {
            "blob_min_threshold": config.get("blob_min_threshold", 80),
            "blob_max_threshold": config.get("blob_max_threshold", 255),
            "blob_min_area": config.get("blob_min_area", 20),
            "blob_max_area": config.get("blob_max_area", 1000),
            "blob_min_circularity": config.get("blob_min_circularity", 0.7),
            "blob_min_convexity": config.get("blob_min_convexity", 0.9),
            "blob_min_inertia": config.get("blob_min_inertia", 0.6)
        }
        self.settings_changed = True
        self.blob_settings_changed = True
        self.last_pos = None
        self.last_pos_time = None
        self.detection_history = DetectionHistory()

    def run(self):
        with StereoCamera(CAMERA_RESOLUTION, CAMERA_FPS) as stereo_cam:
            blob_detector = BlobDetector(self.blob_params)

            IP = "127.0.0.1"
            PORT = 4241
            sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            stereo_cam.set_exposure(self.exposure, self.iso)

            while self.running:
                if self.settings_changed:
                    stereo_cam.set_exposure(self.exposure, self.iso)
                    self.settings_changed = False

                if self.blob_settings_changed:
                    blob_detector.update_params(self.blob_params)
                    self.blob_settings_changed = False

                frame_l, frame_r = stereo_cam.get_stereo_frames()

                candidates_l = blob_detector.detect_candidates(frame_l.uncropped)
                candidates_r = blob_detector.detect_candidates(frame_r.uncropped)

                match = match_candidates(candidates_l, candidates_r, stereo_cam)

                if match:
                    cX_l, cY_l_top, _ = match['left']
                    cX_r, cY_r_top, _ = match['right']
                    tracked_pos = match['pos_3d']
                    self.last_pos = tracked_pos
                    self.last_pos_time = frame_l.time_of_capture
                    s_l = s_r = True

                    frame_h = CAMERA_RESOLUTION_NUMERIC[1]
                    # For UI display on vertically flipped images
                    cY_l = frame_h - cY_l_top
                    cY_r = frame_h - cY_r_top

                    self.detection_history.add(DetectionRecord(
                        point_l=(cX_l, cY_l_top),
                        point_r=(cX_r, cY_r_top),
                        timestamp=frame_l.time_of_capture,
                        pos_3d=tracked_pos,
                        confidence=match['confidence'],
                    ))
                else:
                    self.detection_history.record_miss()
                    s_l = s_r = False
                    cX_l = cY_l = cX_r = cY_r = -1.0
                    cY_l_top = cY_r_top = -1.0

                # Steer each camera's crop towards where the point is predicted to
                # be a short time ahead, using the recent detection history, since
                # simply following the latest detection is too slow for fast
                # moving objects.
                prediction = self.detection_history.predict(CROP_PREDICTION_LOOKAHEAD_S)
                if prediction is not None:
                    (pX_l, pY_l_top), (pX_r, pY_r_top) = prediction
                    has_p = True
                else:
                    (pX_l, pY_l_top), (pX_r, pY_r_top) = (cX_l, cY_l_top), (cX_r, cY_r_top)
                    has_p = False

                frame_h = CAMERA_RESOLUTION_NUMERIC[1]
                pY_l = frame_h - pY_l_top if has_p or s_l else -1.0
                pY_r = frame_h - pY_r_top if has_p or s_r else -1.0

                self.frame_ready.emit(frame_l.uncropped.copy(), frame_r.uncropped.copy())
                self.centroid_ready.emit(cX_l, cY_l, s_l, cX_r, cY_r, s_r, pX_l, pY_l, pX_r, pY_r, has_p)

                temporal_confidence = self.detection_history.compute_temporal_confidence()
                self.confidence_ready.emit(temporal_confidence)

                stereo_cam.track_crop(
                    has_p, pX_l, pY_l_top,
                    has_p, pX_r, pY_r_top,
                )

                latency_arrival = (frame_l.time_of_arrival - frame_l.time_of_capture) * 1000
                ts_diff_us = abs(frame_l.time_of_capture - frame_r.time_of_capture) * 1_000_000
                latency_calc = -1.0
                latency_total = -1.0

                if s_l and s_r:
                    t_3d_finished = dai.Clock.now().total_seconds()
                    latency_calc = (t_3d_finished - frame_l.time_of_arrival) * 1000
                    latency_total = (t_3d_finished - frame_l.time_of_capture) * 1000
                    self.position_ready.emit(tracked_pos)

                    tracked_pos_with_empty_rotation = np.zeros(6)
                    tracked_pos_with_empty_rotation[:3] = tracked_pos
                    sock.sendto(tracked_pos_with_empty_rotation.tobytes(), (IP, PORT))

                self.stats_ready.emit(frame_l.frame_time, latency_arrival, latency_calc, latency_total, ts_diff_us)

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DepthAI Point Tracker")
        self.resize(1200, 900)

        self.config = load_config()

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left Panel: Controls
        controls_panel = QtWidgets.QWidget()
        controls_panel.setFixedWidth(300)
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        main_layout.addWidget(controls_panel)

        # Camera Settings
        controls_layout.addWidget(QtWidgets.QLabel("<b>Camera Settings</b>"))

        # Exposure
        controls_layout.addWidget(QtWidgets.QLabel("Exposure (\u03bcs):"))
        self.exp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exp_slider.setRange(1, 33000)
        self.exp_slider.setValue(self.config['exposure'])
        self.exp_spin = QtWidgets.QSpinBox()
        self.exp_spin.setRange(1, 33000)
        self.exp_spin.setValue(self.config['exposure'])
        exp_h = QtWidgets.QHBoxLayout()
        exp_h.addWidget(self.exp_slider)
        exp_h.addWidget(self.exp_spin)
        controls_layout.addLayout(exp_h)

        # ISO
        controls_layout.addWidget(QtWidgets.QLabel("ISO:"))
        self.iso_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iso_slider.setRange(100, 1600)
        self.iso_slider.setValue(self.config['iso'])
        self.iso_spin = QtWidgets.QSpinBox()
        self.iso_spin.setRange(100, 1600)
        self.iso_spin.setValue(self.config['iso'])
        iso_h = QtWidgets.QHBoxLayout()
        iso_h.addWidget(self.iso_slider)
        iso_h.addWidget(self.iso_spin)
        controls_layout.addLayout(iso_h)

        # Blob Detector Settings
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QtWidgets.QLabel("<b>Blob Detector Settings</b>"))

        # Blob Min/Max Threshold
        controls_layout.addWidget(QtWidgets.QLabel("Blob Threshold (Min/Max):"))
        self.blob_min_thresh_spin = QtWidgets.QSpinBox()
        self.blob_min_thresh_spin.setRange(0, 255)
        self.blob_min_thresh_spin.setValue(self.config['blob_min_threshold'])
        self.blob_max_thresh_spin = QtWidgets.QSpinBox()
        self.blob_max_thresh_spin.setRange(0, 255)
        self.blob_max_thresh_spin.setValue(self.config['blob_max_threshold'])
        blob_thresh_h = QtWidgets.QHBoxLayout()
        blob_thresh_h.addWidget(self.blob_min_thresh_spin)
        blob_thresh_h.addWidget(self.blob_max_thresh_spin)
        controls_layout.addLayout(blob_thresh_h)

        # Blob Min/Max Area
        controls_layout.addWidget(QtWidgets.QLabel("Blob Area (Min/Max):"))
        self.blob_min_area_spin = QtWidgets.QSpinBox()
        self.blob_min_area_spin.setRange(1, 10000)
        self.blob_min_area_spin.setValue(self.config['blob_min_area'])
        self.blob_max_area_spin = QtWidgets.QSpinBox()
        self.blob_max_area_spin.setRange(1, 10000)
        self.blob_max_area_spin.setValue(self.config['blob_max_area'])
        blob_area_h = QtWidgets.QHBoxLayout()
        blob_area_h.addWidget(self.blob_min_area_spin)
        blob_area_h.addWidget(self.blob_max_area_spin)
        controls_layout.addLayout(blob_area_h)

        # Blob Circularity
        controls_layout.addWidget(QtWidgets.QLabel("Min Circularity:"))
        self.blob_circ_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blob_circ_slider.setRange(0, 100)
        self.blob_circ_slider.setValue(int(self.config['blob_min_circularity'] * 100))
        self.blob_circ_spin = QtWidgets.QDoubleSpinBox()
        self.blob_circ_spin.setRange(0.0, 1.0)
        self.blob_circ_spin.setSingleStep(0.05)
        self.blob_circ_spin.setValue(self.config['blob_min_circularity'])
        blob_circ_h = QtWidgets.QHBoxLayout()
        blob_circ_h.addWidget(self.blob_circ_slider)
        blob_circ_h.addWidget(self.blob_circ_spin)
        controls_layout.addLayout(blob_circ_h)

        # Blob Convexity
        controls_layout.addWidget(QtWidgets.QLabel("Min Convexity:"))
        self.blob_conv_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blob_conv_slider.setRange(0, 100)
        self.blob_conv_slider.setValue(int(self.config['blob_min_convexity'] * 100))
        self.blob_conv_spin = QtWidgets.QDoubleSpinBox()
        self.blob_conv_spin.setRange(0.0, 1.0)
        self.blob_conv_spin.setSingleStep(0.05)
        self.blob_conv_spin.setValue(self.config['blob_min_convexity'])
        blob_conv_h = QtWidgets.QHBoxLayout()
        blob_conv_h.addWidget(self.blob_conv_slider)
        blob_conv_h.addWidget(self.blob_conv_spin)
        controls_layout.addLayout(blob_conv_h)

        # Blob Inertia
        controls_layout.addWidget(QtWidgets.QLabel("Min Inertia Ratio:"))
        self.blob_inert_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blob_inert_slider.setRange(0, 100)
        self.blob_inert_slider.setValue(int(self.config['blob_min_inertia'] * 100))
        self.blob_inert_spin = QtWidgets.QDoubleSpinBox()
        self.blob_inert_spin.setRange(0.0, 1.0)
        self.blob_inert_spin.setSingleStep(0.05)
        self.blob_inert_spin.setValue(self.config['blob_min_inertia'])
        blob_inert_h = QtWidgets.QHBoxLayout()
        blob_inert_h.addWidget(self.blob_inert_slider)
        blob_inert_h.addWidget(self.blob_inert_spin)
        controls_layout.addLayout(blob_inert_h)

        # Info
        controls_layout.addStretch()
        controls_layout.addWidget(QtWidgets.QLabel("<b>Tracking Info</b>"))
        self.centroid_label = QtWidgets.QLabel("Centroid: N/A")
        controls_layout.addWidget(self.centroid_label)
        self.pos_label = QtWidgets.QLabel("XYZ: N/A")
        controls_layout.addWidget(self.pos_label)

        controls_layout.addWidget(QtWidgets.QLabel("Tracking confidence:"))
        self.confidence_plot = pg.PlotWidget()
        self.confidence_plot.setFixedHeight(80)
        self.confidence_plot.setMouseEnabled(x=False, y=False)
        self.confidence_plot.hideAxis('bottom')
        self.confidence_plot.setYRange(0.0, 1.0, padding=0)
        self.confidence_plot.setXRange(-1.0, 1.0, padding=0)
        self.confidence_bar = pg.BarGraphItem(x=[0], height=[0.0], width=1.2, brush='g')
        self.confidence_plot.addItem(self.confidence_bar)
        controls_layout.addWidget(self.confidence_plot)

        controls_layout.addSpacing(10)
        controls_layout.addWidget(QtWidgets.QLabel("<b>Timing Info</b>"))
        self.frame_time_label = QtWidgets.QLabel("Frame time: N/A")
        controls_layout.addWidget(self.frame_time_label)
        self.capture_latency_label = QtWidgets.QLabel("Capture latency: N/A")
        controls_layout.addWidget(self.capture_latency_label)
        self.processing_latency_label = QtWidgets.QLabel("Processing latency: N/A")
        controls_layout.addWidget(self.processing_latency_label)
        self.total_latency_label = QtWidgets.QLabel("Total latency: N/A")
        controls_layout.addWidget(self.total_latency_label)
        self.lr_diff_label = QtWidgets.QLabel("L-R frame diff: N/A")
        controls_layout.addWidget(self.lr_diff_label)

        # Right Panel: Visuals
        visuals_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(visuals_layout)
        top_visuals = QtWidgets.QHBoxLayout()

        # Top Visuals: Image Preview and 3D
        visuals_layout.addLayout(top_visuals, stretch=1)

        # Image Previews
        self.image_view_l = pg.GraphicsLayoutWidget()
        self.image_view_l.setBackground('gray')
        self.image_view_l.setMinimumSize(300, 400)
        self.image_view_l.addLabel("<span style='color: #00FF00; font-weight: bold;'>Left Camera</span>", row=0, col=0)
        self.vb_l = self.image_view_l.addViewBox(row=1, col=0)
        self.vb_l.setAspectLocked(True)
        self.img_item_l = pg.ImageItem()
        self.vb_l.addItem(self.img_item_l)
        self.crosshair_v_l = pg.InfiniteLine(angle=90, movable=False, pen='r')
        self.crosshair_h_l = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.vb_l.addItem(self.crosshair_v_l)
        self.vb_l.addItem(self.crosshair_h_l)
        self.crosshair_v_l.hide()
        self.crosshair_h_l.hide()

        self.pred_marker_l = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush('g'))
        self.vb_l.addItem(self.pred_marker_l)

        self.image_view_r = pg.GraphicsLayoutWidget()
        self.image_view_r.setBackground('gray')
        self.image_view_r.setMinimumSize(300, 400)
        self.image_view_r.addLabel("<span style='color: #00FF00; font-weight: bold;'>Right Camera</span>", row=0, col=0)
        self.vb_r = self.image_view_r.addViewBox(row=1, col=0)
        self.vb_r.setAspectLocked(True)
        self.img_item_r = pg.ImageItem()
        self.vb_r.addItem(self.img_item_r)
        self.crosshair_v_r = pg.InfiniteLine(angle=90, movable=False, pen='r')
        self.crosshair_h_r = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.vb_r.addItem(self.crosshair_v_r)
        self.vb_r.addItem(self.crosshair_h_r)
        self.crosshair_v_r.hide()
        self.crosshair_h_r.hide()

        self.pred_marker_r = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush('g'))
        self.vb_r.addItem(self.pred_marker_r)

        # Pre-fill both ImageItems with a blank frame of identical size so the
        # left and right previews have the same shape/aspect ratio right from
        # startup (before any real frame has been received from the worker),
        # and lock their views together so they always stay in sync.
        blank_w, blank_h = CAMERA_RESOLUTION_NUMERIC
        blank_frame = np.zeros((blank_w, blank_h), dtype=np.uint8)
        # Disable pyqtgraph's automatic level (brightness/contrast) scaling so the
        # preview reflects the camera's actual exposure instead of being
        # auto-stretched to the min/max of each frame, which otherwise makes
        # dark frames appear artificially lightened.
        self.img_item_l.setImage(blank_frame, autoLevels=False, levels=(0, 255))
        self.img_item_r.setImage(blank_frame, autoLevels=False, levels=(0, 255))
        self.vb_r.setXLink(self.vb_l)
        self.vb_r.setYLink(self.vb_l)

        top_visuals.addWidget(self.image_view_l, stretch=5)
        top_visuals.addWidget(self.image_view_r, stretch=5)

        # 3D View
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setMinimumSize(400, 300)
        grid = gl.GLGridItem(size=QtGui.QVector3D(100, 100, 1))
        grid.setSpacing(10, 10, 10, )
        self.gl_view.addItem(grid)
        self.pos_marker = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
        self.gl_view.addItem(self.pos_marker)
        top_visuals.addWidget(self.gl_view, stretch=6)

        # Plot
        self.plot_widget = pg.PlotWidget(title="XYZ over Time")
        self.plot_widget.setMinimumSize(400, 300)
        self.plot_widget.addLegend()
        self.curve_x = self.plot_widget.plot(pen='r', name='X')
        self.curve_y = self.plot_widget.plot(pen='g', name='Y')
        self.curve_z = self.plot_widget.plot(pen='b', name='Z')
        self.data_x = []
        self.data_y = []
        self.data_z = []
        self.max_points = 100
        visuals_layout.addWidget(self.plot_widget)

        # Connect signals
        self.exp_slider.valueChanged.connect(self.exp_spin.setValue)
        self.exp_spin.valueChanged.connect(self.update_exposure)
        self.iso_slider.valueChanged.connect(self.iso_spin.setValue)
        self.iso_spin.valueChanged.connect(self.update_iso)

        self.blob_min_thresh_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_threshold", v))
        self.blob_max_thresh_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_max_threshold", v))
        self.blob_min_area_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_area", v))
        self.blob_max_area_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_max_area", v))
        self.blob_circ_slider.valueChanged.connect(lambda v: self.blob_circ_spin.setValue(v / 100.0))
        self.blob_circ_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_circularity", v))
        self.blob_conv_slider.valueChanged.connect(lambda v: self.blob_conv_spin.setValue(v / 100.0))
        self.blob_conv_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_convexity", v))
        self.blob_inert_slider.valueChanged.connect(lambda v: self.blob_inert_spin.setValue(v / 100.0))
        self.blob_inert_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_inertia", v))

        # Worker thread
        self.worker = Worker(self.config)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.centroid_ready.connect(self.on_centroid)
        self.worker.position_ready.connect(self.on_position)
        self.worker.stats_ready.connect(self.on_stats)
        self.worker.confidence_ready.connect(self.on_confidence)
        self.worker.start()

    def update_exposure(self, val):
        self.worker.exposure = val
        self.worker.settings_changed = True
        self.exp_slider.blockSignals(True)
        self.exp_slider.setValue(val)
        self.exp_slider.blockSignals(False)

    def update_iso(self, val):
        self.worker.iso = val
        self.worker.settings_changed = True
        self.iso_slider.blockSignals(True)
        self.iso_slider.setValue(val)
        self.iso_slider.blockSignals(False)

    def update_blob_param(self, name, val):
        self.worker.blob_params[name] = val
        self.worker.blob_settings_changed = True
        if name == "blob_min_circularity":
            self.blob_circ_slider.blockSignals(True)
            self.blob_circ_slider.setValue(int(val * 100))
            self.blob_circ_slider.blockSignals(False)
        elif name == "blob_min_convexity":
            self.blob_conv_slider.blockSignals(True)
            self.blob_conv_slider.setValue(int(val * 100))
            self.blob_conv_slider.blockSignals(False)
        elif name == "blob_min_inertia":
            self.blob_inert_slider.blockSignals(True)
            self.blob_inert_slider.setValue(int(val * 100))
            self.blob_inert_slider.blockSignals(False)

    @QtCore.Slot(np.ndarray, np.ndarray)
    def on_frame(self, frame_l: np.ndarray, frame_r: np.ndarray):
        for frame, img_item in [(frame_l, self.img_item_l), (frame_r, self.img_item_r)]:
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
            mirrored = cv2.flip(rotated_frame, 1)
            img_item.setImage(mirrored.T, autoLevels=False, levels=(0, 255))

    @QtCore.Slot(float, float, bool, float, float, bool, float, float, float, float, bool)
    def on_centroid(self, x_l, y_l, found_l, x_r, y_r, found_r, px_l, py_l, px_r, py_r, has_p):
        # Update Left
        if found_l:
            self.crosshair_v_l.setPos(x_l)
            self.crosshair_h_l.setPos(y_l)
            self.crosshair_v_l.show()
            self.crosshair_h_l.show()
        else:
            self.crosshair_v_l.hide()
            self.crosshair_h_l.hide()

        if has_p:
            self.pred_marker_l.setData(pos=[(px_l, py_l)])
            self.pred_marker_l.show()
        else:
            self.pred_marker_l.hide()

        # Update Right
        if found_r:
            self.crosshair_v_r.setPos(x_r)
            self.crosshair_h_r.setPos(y_r)
            self.crosshair_v_r.show()
            self.crosshair_h_r.show()
        else:
            self.crosshair_v_r.hide()
            self.crosshair_h_r.hide()

        if has_p:
            self.pred_marker_r.setData(pos=[(px_r, py_r)])
            self.pred_marker_r.show()
        else:
            self.pred_marker_r.hide()

        text_l = f"L: {x_l:.1f}, {y_l:.1f}" if found_l else "L: N/A"
        text_r = f"R: {x_r:.1f}, {y_r:.1f}" if found_r else "R: N/A"
        self.centroid_label.setText(f"Centroid: {text_l} | {text_r}")

    @QtCore.Slot(np.ndarray)
    def on_position(self, pos):
        self.pos_label.setText(f"XYZ: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}")
        # OpenCV Y is down, so negate it for GL view Z (up)
        self.pos_marker.setData(pos=np.array([[pos[0], pos[2], -pos[1]]]))
        self.data_x.append(pos[0])
        self.data_y.append(pos[1])
        self.data_z.append(pos[2])
        if len(self.data_x) > self.max_points:
            self.data_x.pop(0)
            self.data_y.pop(0)
            self.data_z.pop(0)
        self.curve_x.setData(self.data_x)
        self.curve_y.setData(self.data_y)
        self.curve_z.setData(self.data_z)

    @QtCore.Slot(float, float, float, float, float)
    def on_stats(self, frame_time, l_arrival, l_processing, l_total, ts_diff):
        fps = float(1000.0 / frame_time)
        self.frame_time_label.setText(f"Frame time: {frame_time:.1f}ms ({fps:.1f} FPS)")
        self.capture_latency_label.setText(f"Capture latency: {l_arrival:.1f}ms")
        if l_processing >= 0:
            self.processing_latency_label.setText(f"Processing latency: {l_processing:.1f}ms")
            self.total_latency_label.setText(f"Total latency: {l_total:.1f}ms")
        else:
            self.processing_latency_label.setText("Processing latency: N/A")
            self.total_latency_label.setText("Total latency: N/A")
        self.lr_diff_label.setText(f"L-R frame diff: {ts_diff:.1f}µs")

    @QtCore.Slot(float)
    def on_confidence(self, confidence):
        confidence = max(0.0, min(1.0, confidence))
        self.confidence_bar.setOpts(height=[confidence])
        # Color the bar from red (low confidence) to green (high confidence).
        color = pg.mkColor(int(255 * (1.0 - confidence)), int(255 * confidence), 0)
        self.confidence_bar.setOpts(brush=color)

    def closeEvent(self, event):
        self.worker.stop()
        config_to_save = {
            'exposure': self.worker.exposure,
            'iso': self.worker.iso,
        }
        config_to_save.update(self.worker.blob_params)
        save_config(config_to_save)
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
