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


def normalize(val: float, min_ideal: float, max_ideal: float, min_cutoff: float, max_cutoff: float) -> float:
    if val < min_cutoff:
        return 0.0
    if val > max_cutoff:
        return 0.0
    if val < min_ideal:
        return (val - min_cutoff) / (min_ideal - min_cutoff)
    if val > max_ideal:
        return 1 - (val - max_ideal) / (max_cutoff - max_ideal)

    return 1.0


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
            pos_3d = map_open_cv_to_output_coords(pos_3d)

            # 3. Also consider blob size similarity
            size_ratio = min(det_l.size, det_r.size) / max(det_l.size, det_r.size, 1e-6)

            conf = normalize(y_pixel_diff, min_ideal=0, max_ideal=3.5, min_cutoff=-1, max_cutoff=15)
            conf *= normalize(size_ratio, min_ideal=1, max_ideal=1, min_cutoff=0, max_cutoff=1.1)
            conf *= normalize(pos_3d[2], min_ideal=10, max_ideal=150, min_cutoff=0, max_cutoff=300)

            detections.append(Detection3D(det_l.pos, det_r.pos, rect_l, rect_r, pos_3d, conf))

    detections.sort(key=lambda d: d.confidence_01, reverse=True)
    return detections
