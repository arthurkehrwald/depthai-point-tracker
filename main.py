import sys
from enum import StrEnum

import numpy as np
import cv2
from PySide6 import QtCore, QtWidgets

from blob_detector import BlobDetector
from camera import StereoCamera
import camera
from config import Config
from tracker import detect3d
import ui


class TrackerConfigKeys(StrEnum):
    stereo_conf_threshold = 'stereo_conf_threshold'


class Worker(QtCore.QThread):
    frame_ready = QtCore.Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    centroid_ready = QtCore.Signal(bool, float, float, float, float)
    candidates_ready = QtCore.Signal(list, list)
    position_ready = QtCore.Signal(bool, float, float, float, float, float)
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

                h, w = stereo_frame.left_frame.shape
                oh, ow = h // 2, w // 2
                overlay_l = np.zeros((oh, ow), dtype=np.uint8)
                overlay_r = np.zeros((oh, ow), dtype=np.uint8)

                for det in detections_l:
                    cv2.circle(overlay_l, (int(det.pos[0] / 2), int(det.pos[1] / 2)), int(det.size / 4), 255, 2)
                for det in detections_r:
                    cv2.circle(overlay_r, (int(det.pos[0] / 2), int(det.pos[1] / 2)), int(det.size / 4), 255, 2)

                conf_thresh = self.config.get(TrackerConfigKeys.stereo_conf_threshold)
                if detections_3d and detections_3d[0].confidence_01 >= conf_thresh:
                    detection = detections_3d[0]
                    # Draw crosshair
                    for img, pos in [(overlay_l, detection.pos_2d_raw_l), (overlay_r, detection.pos_2d_raw_r)]:
                        x, y = int(pos[0] / 2), int(pos[1] / 2)
                        line_length = 1000
                        cv2.line(img, (x - line_length, y), (x + line_length, y), 255, 2)
                        cv2.line(img, (x, y - line_length), (x, y + line_length), 255, 2)

                    self.centroid_ready.emit(True, detection.pos_2d_raw_l[0], detection.pos_2d_raw_l[1],
                                             detection.pos_2d_raw_r[0],
                                             detection.pos_2d_raw_r[1])
                    self.position_ready.emit(True, detection.pos_3d[0], detection.pos_3d[1], detection.pos_3d[2],
                                             detection.confidence_01, stereo_frame.left_time_of_capture)
                    if self.recording_active and self.recorder:
                        self.recorder.record(detection.pos_3d[0], detection.pos_3d[1], detection.pos_3d[2],
                                             stereo_frame.left_time_of_capture, detection.confidence_01)
                else:
                    self.centroid_ready.emit(False, 0, 0, 0, 0)
                    self.position_ready.emit(False, 0, 0, 0, 0, 0)

                self.frame_ready.emit(stereo_frame.left_frame.copy(), stereo_frame.right_frame.copy(),
                                      overlay_l, overlay_r)
                self.candidates_ready.emit(detections_l, detections_r)
                self.stats_ready.emit(stereo_frame.frame_time_ms, stereo_frame.left_time_of_capture,
                                      stereo_frame.right_time_of_capture, stereo_frame.time_of_arrival,
                                      camera.get_time())

                self.config.do_callbacks()

    def stop(self):
        self.running = False
        self.wait()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ui.MainWindow()
    window.show()
    sys.exit(app.exec())
