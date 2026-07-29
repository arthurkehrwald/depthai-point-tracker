import typing
from dataclasses import dataclass

import cv2

from config import Config, ConfigValue


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    size: float


class BlobDetector:
    def __init__(self, config: Config):
        self.detector = None
        self.config = config
        self.update_params()
        self.config.add_callback()

    def update_params(self):
        params = cv2.SimpleBlobDetector.Params()
        params.minThreshold = self.config.get_default("blob_min_threshold", 80)
        params.maxThreshold = self.config.get_default("blob_max_threshold", 255)
        params.thresholdStep = self.config.get_default("blob_threshold_step", 10)
        params.maxThreshold = max(params.maxThreshold, params.minThreshold)
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = self.config.get_default("blob_min_area", 20)
        params.maxArea = self.config.get_default("blob_max_area", 1000)
        params.maxArea = max(params.maxArea, params.minArea)
        params.filterByCircularity = True
        params.minCircularity = max(0.01, self.config.get_default("blob_min_circularity", 0.7))
        params.filterByConvexity = True
        params.minConvexity = max(0.01, self.config.get_default("blob_min_convexity", 0.9))
        params.filterByInertia = True
        params.minInertiaRatio = max(0.01, self.config.get_default("blob_min_inertia", 0.6))
        self.detector = cv2.SimpleBlobDetector.create(params)

    def detect_candidates(self, img: cv2.typing.MatLike) -> typing.List[Detection]:
        keypoints = self.detector.detect(img)
        keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)[:10]
        return [Detection(kp.pt[0], kp.pt[1], kp.size) for kp in keypoints]

    def on_config_changed(self, name: str, _: ConfigValue):
        if name.startswith("blob"):
            self.update_params()
