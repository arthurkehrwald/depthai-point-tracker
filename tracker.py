import math
import typing
from dataclasses import dataclass

import numpy as np

from blob_detector import Detection2D
from camera import StereoCamera


@dataclass
class Detection3D:
    pos_2d_raw_l: typing.Tuple[float, float]
    pos_2d_raw_r: typing.Tuple[float, float]
    pos_2d_rect_l: typing.Tuple[float, float]
    pos_2d_rect_r: typing.Tuple[float, float]
    pos_3d: typing.Tuple[float, float, float]
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


def detect3d(detections_l: typing.List[Detection2D], detections_r: typing.List[Detection2D],
             stereo_cam: StereoCamera) -> typing.List[Detection3D]:
    if not detections_l or not detections_r:
        return []

    rect_coords_l = stereo_cam.rectify_points([d.pos for d in detections_l], is_left=True)
    rect_coords_r = stereo_cam.rectify_points([d.pos for d in detections_r], is_left=False)

    detections = []
    for det_l, rect_l in zip(detections_l, rect_coords_l):
        for det_r, rect_r in zip(detections_r, rect_coords_r):
            # 1. Epipolar constraint: Corresponding points must have very similar Y-coordinates in image space
            y_pixel_diff = abs(rect_l[1] - rect_r[1])

            # 2. Plausibility of 3D position
            pos_3d = stereo_cam.triangulate((rect_l[0], rect_l[1]), (rect_r[0], rect_r[1]))

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

            detections.append(Detection3D(det_l.pos, det_r.pos, rect_l, rect_r, pos_3d, conf))

    detections.sort(key=lambda d: d.confidence_01, reverse=True)
    return detections
