import sys
import time
from enum import StrEnum

import cv2
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

                #detections_l = blobdetector.detect_candidates(stereo_frame.left_frame)
                #detections_r = blobdetector.detect_candidates(stereo_frame.right_frame)
                detections_l = None
                detections_r = None
                detections_3d = detect3d(detections_l, detections_r, cam)

                left_frame = cv2.cvtColor(stereo_frame.left_frame, cv2.COLOR_GRAY2BGR)
                right_frame = cv2.cvtColor(stereo_frame.right_frame, cv2.COLOR_GRAY2BGR)


#                for det in detections_l:
#                    cv2.circle(left_frame, (int(det.pos[0]), int(det.pos[1])), int(det.size), (0, 255, 0), 2)
#                for det in detections_r:
#                    cv2.circle(right_frame, (int(det.pos[0]), int(det.pos[1])), int(det.size), (0, 255, 0), 2)

                conf_thresh = self.config.get(TrackerConfigKeys.stereo_conf_threshold)
                best_det = None
                if detections_3d and detections_3d[0].confidence_01 >= conf_thresh:
                    best_det = detections_3d[0]
                    for frame, pos in [(left_frame, best_det.pos_2d_raw_l), (right_frame, best_det.pos_2d_raw_r)]:
                        x, y = int(pos[0]), int(pos[1])
                        cv2.line(frame, (x - 10, y), (x + 10, y), (255, 0, 0), 1)
                        cv2.line(frame, (x, y - 10), (x, y + 10), (255, 0, ), 1)

                    self.position_ready.emit(True, best_det.pos_3d[0], best_det.pos_3d[1], best_det.pos_3d[2],
                                             best_det.confidence_01)
                    if self.recording_active and self.recorder:
                        self.recorder.record(best_det.pos_3d[0], best_det.pos_3d[1], best_det.pos_3d[2],
                                             stereo_frame.left_time_of_capture, best_det.confidence_01)
                else:
                    self.position_ready.emit(False, 0, 0, 0, 0)

                self.frame_ready.emit(left_frame, right_frame)
                self.stats_ready.emit(stereo_frame.frame_time_ms, stereo_frame.left_time_of_capture,
                                      stereo_frame.right_time_of_capture, stereo_frame.time_of_arrival,
                                      cam.get_time())

                self.config.do_callbacks()

    def stop(self):
        self.running = False
        self.wait()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ui.MainWindow()
    window.show()
    sys.exit(app.exec())
