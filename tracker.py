import math
import socket
import typing
from dataclasses import dataclass

import numpy as np

from udp_sender import UdpSender
from blob_detector import BlobDetector, Detection
from camera import StereoCamera, StereoFrame
from config import Config

@dataclass
class TrackerFrameInfo:
    frame: StereoFrame
    pos_2d_raw_l: typing.Tuple[float, float]
    pos_2d_raw_r: typing.Tuple[float, float]
    pos_2d_rect_l : typing.Tuple[float, float]
    pos_2d_rect_r : typing.Tuple[float, float]
    pos_3d: typing.Tuple[float, float, float]
    pos_3d_found_time: float
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


def match_candidates(candidates_l: typing.List[Detection], candidates_r: typing.List[Detection], stereo_cam: StereoCamera, conf_threshold: float):
    best_match = None
    highest_confidence = 0

    for candidate_l in candidates_l:
        x_l_raw, y_l_raw, s_l = candidate_l['raw']
        x_l_rect, y_l_rect = candidate_l['rect']
        for candidate_r in candidates_r:
            x_r_raw, y_r_raw, s_r = candidate_r['raw']
            x_r_rect, y_r_rect = candidate_r['rect']

            # 1. Epipolar constraint: Corresponding points must have very similar Y-coordinates in image space
            y_pixel_diff = abs(y_l_rect - y_r_rect)

            # 2. Plausibility of 3D position
            pos_3d = stereo_cam.triangulate((x_l_rect, y_l_rect), (x_r_rect, y_r_rect))

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

            if conf > highest_confidence and conf > conf_threshold:
                highest_confidence = conf
                best_match = {
                    'left_raw': (x_l_raw, y_l_raw, s_l),
                    'right_raw': (x_r_raw, y_r_raw, s_r),
                    'pos_3d': pos_3d,
                    'confidence': highest_confidence
                }
    return best_match


def map_open_cv_to_output_coords(pos: np.ndarray) -> np.ndarray:
    return np.array([pos[0], -pos[1], pos[2]])


class Tracker:
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.running = True
        self.stereo_conf_threshold = float(config.get('stereo_conf_threshold'))
        self.blob_params = {
            "blob_min_threshold": config.get("blob_min_threshold", 80),
            "blob_max_threshold": config.get("blob_max_threshold", 255),
            "blob_threshold_step": config.get("blob_threshold_step", 10),
            "blob_min_area": config.get("blob_min_area", 20),
            "blob_max_area": config.get("blob_max_area", 1000),
            "blob_min_circularity": config.get("blob_min_circularity", 0.7),
            "blob_min_convexity": config.get("blob_min_convexity", 0.9),
            "blob_min_inertia": config.get("blob_min_inertia", 0.6)
        }
        self.settings_changed = True
        self.blob_settings_changed = True

    def run(self):
        with StereoCamera(self.config) as stereo_cam:
            blob_detector = BlobDetector(self.config)
            udp_sender = UdpSender("127.0.0.1", 4241)

            while self.running:
                stereo_frame = stereo_cam.get_stereo_frames()

                raw_candidates_l = blob_detector.detect_candidates(stereo_frame.left_frame)
                raw_candidates_r = blob_detector.detect_candidates(stereo_frame.right_frame)

                candidates_l = []
                for detection in raw_candidates_l:
                    candidates_l.append({
                        'raw': (detection.x, detection.y, s),
                        'rect': stereo_cam.cam_l.rectify_point(x, y)
                    })
                candidates_r = []
                for x, y, s in raw_candidates_r:
                    candidates_r.append({
                        'raw': (x, y, s),
                        'rect': stereo_cam.cam_r.rectify_point(x, y)
                    })

                match = match_candidates(candidates_l, candidates_r, stereo_cam, self.stereo_conf_threshold)
                found_correspondence = match is not None

                if found_correspondence:
                    cX_l, cY_l, _ = match['left_raw']
                    cX_r, cY_r, _ = match['right_raw']
                    tracked_pos = match['pos_3d']
                else:
                    cX_l = cY_l = cX_r = cY_r = -1.0

                self.frame_ready.emit(frame_l.frame.copy(), frame_r.frame.copy())
                self.centroid_ready.emit(found_correspondence, cX_l, cY_l, cX_r, cY_r)

                latency_arrival = (frame_l.time_of_arrival - frame_l.time_of_capture) * 1000
                ts_diff_us = abs(frame_l.time_of_capture - frame_r.time_of_capture) * 1_000_000
                latency_calc = -1.0
                latency_total = -1.0

                if found_correspondence:
                    t_3d_finished = dai.Clock.now().total_seconds()
                    latency_calc = (t_3d_finished - frame_l.time_of_arrival) * 1000
                    latency_total = (t_3d_finished - frame_l.time_of_capture) * 1000
                    self.position_ready.emit(tracked_pos)

                    udp_sender.send(tracked_pos)

                self.stats_ready.emit(frame_l.frame_time, latency_arrival, latency_calc, latency_total, ts_diff_us)

    def stop(self):
        self.running = False
