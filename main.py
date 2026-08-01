import sys
from enum import StrEnum

import numpy as np
from PySide6 import QtCore, QtWidgets

from blob_detector import BlobDetector
from camera import StereoCamera
from config import Config
from tracker import detect3d
import ui


class TrackerConfigKeys(StrEnum):
    stereo_conf_threshold = 'stereo_conf_threshold'


class Worker(QtCore.QThread):
    frame_ready = QtCore.Signal(np.ndarray, np.ndarray)
    centroid_ready = QtCore.Signal(bool, float, float, float, float)
    candidates_ready = QtCore.Signal(list, list)
    position_ready = QtCore.Signal(bool, float, float, float, float)
    stats_ready = QtCore.Signal(float, float, float, float, float)
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

                self.frame_ready.emit(stereo_frame.left_frame.copy(), stereo_frame.right_frame.copy())
                self.candidates_ready.emit(detections_l, detections_r)
                self.stats_ready.emit(stereo_frame.frame_time_ms, stereo_frame.left_time_of_capture,
                                      stereo_frame.right_time_of_capture, stereo_frame.time_of_arrival,
                                      cam.get_time())
                conf_thresh = self.config.get(TrackerConfigKeys.stereo_conf_threshold)
                if detections_3d and detections_3d[0].confidence_01 >= conf_thresh:
                    detection = detections_3d[0]
                    self.centroid_ready.emit(True, detection.pos_2d_raw_l[0], detection.pos_2d_raw_l[1],
                                             detection.pos_2d_raw_r[0],
                                             detection.pos_2d_raw_r[1])
                    self.position_ready.emit(True, detection.pos_3d[0], detection.pos_3d[1], detection.pos_3d[2],
                                             detection.confidence_01)
                    if self.recording_active and self.recorder:
                        self.recorder.record(detection.pos_3d[0], detection.pos_3d[1], detection.pos_3d[2],
                                             stereo_frame.left_time_of_capture, detection.confidence_01)
                else:
                    self.centroid_ready.emit(False, 0, 0, 0, 0)
                    self.position_ready.emit(False, 0, 0, 0, 0)

                self.config.do_callbacks()

    def stop(self):
        self.running = False
        self.wait()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ui.MainWindow()
    window.show()
    sys.exit(app.exec())
