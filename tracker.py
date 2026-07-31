import math
import typing
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

import blob_detector
from camera import StereoCamera, StereoFrame
import config

class TrackerConfigKeys(StrEnum):
    stereo_conf_threshold = 'stereo_conf_threshold'

DEFAULT_CONFIG = {
    TrackerConfigKeys.stereo_conf_threshold: 0.5
}

@dataclass
class Detection3D:
    frame: StereoFrame
    pos_2d_raw_l: typing.Tuple[float, float]
    pos_2d_raw_r: typing.Tuple[float, float]
    pos_2d_rect_l: typing.Tuple[float, float]
    pos_2d_rect_r: typing.Tuple[float, float]
    pos_3d: typing.Tuple[float, float, float]
    time_of_processing_finished: float
    confidence_01: float


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


def map_open_cv_to_output_coords(pos: typing.Tuple[float, float, float]) -> typing.Tuple[float, float, float]:
    return pos[0], -pos[1], pos[2]


class Tracker:
    def __init__(self, config: config.Config):
        super().__init__()
        self.config = config
        self.blob_detector = blob_detector.BlobDetector(self.config)

    def __enter__(self):
        self.stereo_cam = StereoCamera(self.config)
        self.stereo_cam.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stereo_cam.stop()

    def detect(self):
        stereo_frame = self.stereo_cam.get_stereo_frame()

        detections_l = self.blob_detector.detect_candidates(stereo_frame.left_frame)
        detections_r = self.blob_detector.detect_candidates(stereo_frame.right_frame)
        best_match = Detection3D(stereo_frame, (0, 0), (0, 0), (0, 0), (0, 0), (0, 0, 0), -1, 0)
        if not detections_l or not detections_r:
            return best_match
        rect_coords_l = self.stereo_cam.rectify_points([(d.x, d.y) for d in detections_l], is_left=True)
        rect_coords_r = self.stereo_cam.rectify_points([(d.x, d.y) for d in detections_r], is_left=False)

        for det_l, rect_l in zip(detections_l, rect_coords_l):
            for det_r, rect_r in zip(detections_r, rect_coords_r):
                # 1. Epipolar constraint: Corresponding points must have very similar Y-coordinates in image space
                y_pixel_diff = abs(rect_l[1] - rect_r[1])

                # 2. Plausibility of 3D position
                pos_3d = self.stereo_cam.triangulate((rect_l[0], rect_l[1]), (rect_r[0], rect_r[1]))

                # 3. Also consider blob size similarity
                size_ratio = min(det_l.size, det_r.size) / max(det_l.size, det_r.size, 1e-6)

                # 4. And absolute size
                size = det_l.size + det_r.size

                weights = [10, 1, 1, 1, 1, 1]

                # Combined score using logistic interpolation
                conf = np.average([logistic_interpolation(y_pixel_diff, ideal=0, cutoff=3),
                                   logistic_interpolation(pos_3d[0], ideal=0, cutoff=-200, cutoff2=200),
                                   logistic_interpolation(pos_3d[1], ideal=0, cutoff=-100, cutoff2=100),
                                   logistic_interpolation(pos_3d[2], ideal=100, cutoff=0, cutoff2=300),
                                   logistic_interpolation(size_ratio, ideal=1, cutoff=0),
                                   logistic_interpolation(size, ideal=20, cutoff=5)], weights=weights)

                if conf > best_match.confidence_01:
                    best_match.pos_2d_raw_l = det_l.x, det_l.y
                    best_match.pos_2d_raw_r = det_r.x, det_r.y
                    best_match.pos_3d = map_open_cv_to_output_coords(pos_3d)
                    best_match.pos_2d_rect_l = rect_l
                    best_match.pos_2d_rect_r = rect_r
                    best_match.time_of_processing_finished = self.stereo_cam.get_time()
                    best_match.confidence_01 = float(conf)

        return best_match
