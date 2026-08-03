import sys
from enum import StrEnum

from dataclasses import dataclass, field
import typing

import numpy as np
import cv2
from PySide6 import QtCore, QtWidgets

from blob_detector import BlobDetector, Detection2D
from camera import StereoCamera, StereoFrame
import camera
from config import Config
from tracker import detect3d, Detection3D
import ui


class TrackerConfigKeys(StrEnum):
    stereo_conf_threshold = 'stereo_conf_threshold'


@dataclass
class TrackingData:
    stereo_frame: StereoFrame
    detections_l: typing.List[Detection2D]
    detections_r: typing.List[Detection2D]
    detection_3d: typing.Optional[Detection3D] = None
    timestamp: float = field(default_factory=camera.get_time)


class Worker(QtCore.QThread):
    result_ready = QtCore.Signal(TrackingData)
    recording_active = False
    recorder = None

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.running = True

    def run(self):
        blobdetector = BlobDetector(self.config)
        with StereoCamera(self.config) as cam:
            while self.running:
                stereo_frame = cam.get_stereo_frame()

                detections_l = blobdetector.detect_candidates(stereo_frame.left_frame)
                detections_r = blobdetector.detect_candidates(stereo_frame.right_frame)
                detections_3d = detect3d(detections_l, detections_r, cam)

                detection = None
                conf_thresh = self.config.get(TrackerConfigKeys.stereo_conf_threshold)
                if detections_3d and detections_3d[0].confidence_01 >= conf_thresh:
                    detection = detections_3d[0]
                    if self.recording_active and self.recorder:
                        self.recorder.record(detection.pos_3d[0], detection.pos_3d[1], detection.pos_3d[2],
                                             stereo_frame.left_time_of_capture, detection.confidence_01)

                result = TrackingData(
                    stereo_frame=stereo_frame,
                    detections_l=detections_l,
                    detections_r=detections_r,
                    detection_3d=detection,
                    timestamp=camera.get_time()
                )
                self.result_ready.emit(result)

                self.config.do_callbacks()

    def stop(self):
        self.running = False
        self.wait()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ui.MainWindow()
    window.show()
    sys.exit(app.exec())
