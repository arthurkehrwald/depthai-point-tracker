import typing
from dataclasses import dataclass
from enum import StrEnum

import cv2

from config import Config

@dataclass(frozen=True)
class Detection2D:
    pos: typing.Tuple[float, float]
    size: float


class BlobDetectorConfigKeys(StrEnum):
    min_threshold = "blob_min_threshold"
    max_threshold = "blob_max_threshold"
    threshold_step = "blob_threshold_step"
    min_area = "blob_min_area"
    max_area = "blob_max_area"
    min_circularity = "blob_min_circularity"
    min_convexity = "blob_min_convexity"
    min_inertia = "blob_min_inertia"


DEFAULT_CONFIG = {
    BlobDetectorConfigKeys.min_threshold: 80,
    BlobDetectorConfigKeys.max_threshold: 255,
    BlobDetectorConfigKeys.threshold_step: 10,
    BlobDetectorConfigKeys.min_area: 20,
    BlobDetectorConfigKeys.max_area: 1000,
    BlobDetectorConfigKeys.min_circularity: 0.1,
    BlobDetectorConfigKeys.min_convexity: 0.1,
    BlobDetectorConfigKeys.min_inertia: 0.1,
}


class BlobDetector:
    def __init__(self, config: Config):
        self.detector = None
        self.config = config
        self.update_params()
        self.config.add_callback(self.on_config_changed)

    def update_params(self):
        params = cv2.SimpleBlobDetector.Params()
        params.minThreshold = self.config.get(BlobDetectorConfigKeys.min_threshold)
        params.maxThreshold = self.config.get(BlobDetectorConfigKeys.max_threshold)
        params.thresholdStep = self.config.get(BlobDetectorConfigKeys.threshold_step)
        params.maxThreshold = max(params.maxThreshold, params.minThreshold)
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = self.config.get(BlobDetectorConfigKeys.min_area)
        params.maxArea = self.config.get(BlobDetectorConfigKeys.max_area)
        params.maxArea = max(params.maxArea, params.minArea)
        params.filterByCircularity = True
        params.minCircularity = max(0.01, self.config.get(BlobDetectorConfigKeys.min_circularity))
        params.filterByConvexity = True
        params.minConvexity = max(0.01, self.config.get(BlobDetectorConfigKeys.min_convexity))
        params.filterByInertia = True
        params.minInertiaRatio = max(0.01, self.config.get(BlobDetectorConfigKeys.min_inertia))
        params.minRepeatability = 1
        self.detector = cv2.SimpleBlobDetector.create(params)

    def detect_candidates(self, img: cv2.typing.MatLike) -> typing.List[Detection2D]:
        keypoints = self.detector.detect(img)
        keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)[:10]
        return [Detection2D((kp.pt[0], kp.pt[1]), kp.size) for kp in keypoints]

    def on_config_changed(self, changed: typing.List[str]):
        if any(name.startswith("blob") for name in changed):
            self.update_params()
