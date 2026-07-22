import sys
import json
import socket
import numpy as np
import cv2
import depthai as dai
import typing
import time
from pathlib import Path
from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

CONFIG_FILE = "config.json"
CAMERA_RESOLUTION = (1280, 720)

def load_config():
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "exposure": 200,
        "iso": 100,
        "threshold": 0.9,
        "blob_min_threshold": 80,
        "blob_max_threshold": 255,
        "blob_min_area": 20,
        "blob_max_area": 1000,
        "blob_min_circularity": 0.7,
        "blob_min_convexity": 0.9,
        "blob_min_inertia": 0.6
    }

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

@dataclass(frozen=True)
class CameraSocketParams:
    projection: np.ndarray
    rectify_map_x: np.ndarray
    rectify_map_y: np.ndarray


class StereoCamera:
    def __init__(
            self, pipeline: dai.Pipeline, resolution: typing.Tuple[int, int]
    ) -> None:
        self.pipeline = pipeline
        self.resolution = resolution

        self.cam_params_l, self.cam_params_r = self.compute_stereo_rectification()

        sync = pipeline.create(dai.node.Sync)
        sync.setRunOnHost(True)
        cam_l = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        cam_l.requestOutput(self.resolution).link(sync.inputs["left"])
        cam_r = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        cam_r.requestOutput(self.resolution).link(sync.inputs["right"])
        self.synced_q = sync.out.createOutputQueue()

        control_in_l = cam_l.inputControl
        self.ctrl_q_l = control_in_l.createInputQueue()
        control_in_r = cam_r.inputControl
        self.ctrl_q_r = control_in_r.createInputQueue()

    def compute_stereo_rectification(self) -> typing.Tuple[CameraSocketParams, CameraSocketParams]:
        calibration = self.pipeline.getDefaultDevice().readCalibration()
        intrinsics_l = np.array(
            calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_B, self.resolution[0], self.resolution[1]
            ),
        )
        intrinsics_r = np.array(
            calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_C, self.resolution[0], self.resolution[1]
            ),
        )

        distortion_l = np.array(
            calibration.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B),
        )
        distortion_r = np.array(
            calibration.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C),
        )

        l_to_r_transformation = np.array(
            calibration.getCameraExtrinsics(
                dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
            )
        )
        l_to_r_rotation = l_to_r_transformation[:3, :3]
        l_to_r_translation = l_to_r_transformation[:3, 3:4]

        rotation_l, rotation_r, projection_l, projection_r, _, _, _ = cv2.stereoRectify(
            intrinsics_l, distortion_l.flatten(),
            intrinsics_r, distortion_r.flatten(),
            imageSize=self.resolution,
            R=l_to_r_rotation,
            T=l_to_r_translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        rectify_map_l_x, rectify_map_l_y = cv2.initUndistortRectifyMap(
            intrinsics_l, distortion_l,
            rotation_l, projection_l,
            self.resolution,
            cv2.CV_16SC2
        )
        rectify_map_r_x, rectify_map_r_y = cv2.initUndistortRectifyMap(
            intrinsics_r, distortion_r,
            rotation_r, projection_r,
            self.resolution,
            cv2.CV_16SC2
        )

        return (CameraSocketParams(projection_l, rectify_map_l_x, rectify_map_l_y),
                CameraSocketParams(projection_r, rectify_map_r_x, rectify_map_r_y))

    def try_get_stereo_frames(self) -> typing.Tuple[bool, np.ndarray | None, np.ndarray | None]:
        message_group = self.synced_q.get()
        if isinstance(message_group, dai.MessageGroup):
            left = message_group["left"]
            right = message_group["right"]
            if isinstance(left, dai.ImgFrame) and isinstance(right, dai.ImgFrame):
                rect_l = cv2.remap(left.getCvFrame(), self.cam_params_l.rectify_map_x, self.cam_params_l.rectify_map_y, cv2.INTER_LINEAR)
                rect_r = cv2.remap(right.getCvFrame(), self.cam_params_r.rectify_map_x, self.cam_params_r.rectify_map_y, cv2.INTER_LINEAR)
                return True, rect_l, rect_r
        return False, None, None
    
    def triangulate(
            self,
            point_l: typing.Tuple[float, float],
            point_r: typing.Tuple[float, float],
    ) -> np.ndarray:
        # cv.triangulatePoints operates on 2xN arrays of points
        points_l = np.array(point_l).reshape(2, 1)
        points_r = np.array(point_r).reshape(2, 1)
        points4d: np.ndarray = cv2.triangulatePoints(self.cam_params_l.projection, self.cam_params_r.projection, points_l, points_r)
        first = points4d[:, 0]
        first = first[:3] / first[3]  # homogenous -> cartesian
        # OpenCV's camera coordinate convention defines +Y as pointing DOWN, but who tf does that.
        first[1] *= -1
        return first

    def set_exposure(self, exp_time: int, sens_iso: int) -> None:
        msg = dai.CameraControl()
        msg.setManualExposure(exp_time, sens_iso)
        self.ctrl_q_l.send(msg)
        self.ctrl_q_r.send(msg)

class BlobDetector:
    def __init__(self, config):
        self.detector = None
        self.update_params(config)

    def update_params(self, config):
        params = cv2.SimpleBlobDetector.Params()
        params.minThreshold = config.get("blob_min_threshold", 80)
        params.maxThreshold = config.get("blob_max_threshold", 255)
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = config.get("blob_min_area", 20)
        params.maxArea = config.get("blob_max_area", 1000)
        params.filterByCircularity = True
        params.minCircularity = config.get("blob_min_circularity", 0.7)
        params.filterByConvexity = True
        params.minConvexity = config.get("blob_min_convexity", 0.9)
        params.filterByInertia = True
        params.minInertiaRatio = config.get("blob_min_inertia", 0.6)
        self.detector = cv2.SimpleBlobDetector.create(params)

    def detect(self, img: cv2.typing.MatLike) -> typing.Tuple[bool, float, float]:
        keypoints = self.detector.detect(img)
        if keypoints:
            biggest = max(keypoints, key=lambda kp: kp.size).pt
            return True, biggest[0], biggest[1]
        return False, -1, -1

class Worker(QtCore.QThread):
    frame_ready = QtCore.Signal(np.ndarray, np.ndarray)
    centroid_ready = QtCore.Signal(float, float, bool, float, float, bool)
    position_ready = QtCore.Signal(np.ndarray)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = True
        self.exposure = int(config.get('exposure', 200))
        self.iso = int(config.get('iso', 100))
        self.blob_params = {
            "blob_min_threshold": config.get("blob_min_threshold", 80),
            "blob_max_threshold": config.get("blob_max_threshold", 255),
            "blob_min_area": config.get("blob_min_area", 20),
            "blob_max_area": config.get("blob_max_area", 1000),
            "blob_min_circularity": config.get("blob_min_circularity", 0.7),
            "blob_min_convexity": config.get("blob_min_convexity", 0.9),
            "blob_min_inertia": config.get("blob_min_inertia", 0.6)
        }
        self.settings_changed = True
        self.blob_settings_changed = True

    def run(self):
        with dai.Pipeline() as pipeline:
            stereo_cam = StereoCamera(pipeline, CAMERA_RESOLUTION)
            blob_detector = BlobDetector(self.blob_params)

            IP = "127.0.0.1"
            PORT = 4241
            sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            pipeline.start()

            while self.running and pipeline.isRunning():
                if self.settings_changed:
                    stereo_cam.set_exposure(self.exposure, self.iso)
                    self.settings_changed = False

                if self.blob_settings_changed:
                    blob_detector.update_params(self.blob_params)
                    self.blob_settings_changed = False

                success, img_l, img_r = stereo_cam.try_get_stereo_frames()

                if success and img_l is not None and img_r is not None:
                    s_l, cX_l, cY_l = blob_detector.detect(img_l)
                    s_r, cX_r, cY_r = blob_detector.detect(img_r)

                    self.frame_ready.emit(img_l.copy(), img_r.copy())

                    img_height = np.shape(img_l)[0]
                    disp_cY_l = (cY_l - img_height) * -1
                    disp_cY_r = (cY_r - img_height) * -1

                    self.centroid_ready.emit(cX_l, disp_cY_l, s_l, cX_r, disp_cY_r, s_r)

                    if s_l and s_r:
                        tracked_pos = stereo_cam.triangulate((cX_l, cY_l), (cX_r, cY_r))
                        self.position_ready.emit(tracked_pos)

                        tracked_pos_with_empty_rotation = np.zeros(6)
                        tracked_pos_with_empty_rotation[:3] = tracked_pos
                        sock.sendto(tracked_pos_with_empty_rotation.tobytes(), (IP, PORT))

                time.sleep(0.001)

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DepthAI Point Tracker")
        self.resize(1200, 900)

        self.config = load_config()

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left Panel: Controls
        controls_panel = QtWidgets.QWidget()
        controls_panel.setFixedWidth(300)
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        main_layout.addWidget(controls_panel)

        # Camera Settings
        controls_layout.addWidget(QtWidgets.QLabel("<b>Camera Settings</b>"))

        # Exposure
        controls_layout.addWidget(QtWidgets.QLabel("Exposure (\u03bcs):"))
        self.exp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exp_slider.setRange(1, 33000)
        self.exp_slider.setValue(self.config['exposure'])
        self.exp_spin = QtWidgets.QSpinBox()
        self.exp_spin.setRange(1, 33000)
        self.exp_spin.setValue(self.config['exposure'])
        exp_h = QtWidgets.QHBoxLayout()
        exp_h.addWidget(self.exp_slider)
        exp_h.addWidget(self.exp_spin)
        controls_layout.addLayout(exp_h)

        # ISO
        controls_layout.addWidget(QtWidgets.QLabel("ISO:"))
        self.iso_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iso_slider.setRange(100, 1600)
        self.iso_slider.setValue(self.config['iso'])
        self.iso_spin = QtWidgets.QSpinBox()
        self.iso_spin.setRange(100, 1600)
        self.iso_spin.setValue(self.config['iso'])
        iso_h = QtWidgets.QHBoxLayout()
        iso_h.addWidget(self.iso_slider)
        iso_h.addWidget(self.iso_spin)
        controls_layout.addLayout(iso_h)

        # Blob Detector Settings
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QtWidgets.QLabel("<b>Blob Detector Settings</b>"))

        # Blob Min/Max Threshold
        controls_layout.addWidget(QtWidgets.QLabel("Blob Threshold (Min/Max):"))
        self.blob_min_thresh_spin = QtWidgets.QSpinBox()
        self.blob_min_thresh_spin.setRange(0, 255)
        self.blob_min_thresh_spin.setValue(self.config['blob_min_threshold'])
        self.blob_max_thresh_spin = QtWidgets.QSpinBox()
        self.blob_max_thresh_spin.setRange(0, 255)
        self.blob_max_thresh_spin.setValue(self.config['blob_max_threshold'])
        blob_thresh_h = QtWidgets.QHBoxLayout()
        blob_thresh_h.addWidget(self.blob_min_thresh_spin)
        blob_thresh_h.addWidget(self.blob_max_thresh_spin)
        controls_layout.addLayout(blob_thresh_h)

        # Blob Min/Max Area
        controls_layout.addWidget(QtWidgets.QLabel("Blob Area (Min/Max):"))
        self.blob_min_area_spin = QtWidgets.QSpinBox()
        self.blob_min_area_spin.setRange(1, 10000)
        self.blob_min_area_spin.setValue(self.config['blob_min_area'])
        self.blob_max_area_spin = QtWidgets.QSpinBox()
        self.blob_max_area_spin.setRange(1, 10000)
        self.blob_max_area_spin.setValue(self.config['blob_max_area'])
        blob_area_h = QtWidgets.QHBoxLayout()
        blob_area_h.addWidget(self.blob_min_area_spin)
        blob_area_h.addWidget(self.blob_max_area_spin)
        controls_layout.addLayout(blob_area_h)

        # Blob Circularity
        controls_layout.addWidget(QtWidgets.QLabel("Min Circularity:"))
        self.blob_circ_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blob_circ_slider.setRange(0, 100)
        self.blob_circ_slider.setValue(int(self.config['blob_min_circularity'] * 100))
        self.blob_circ_spin = QtWidgets.QDoubleSpinBox()
        self.blob_circ_spin.setRange(0.0, 1.0)
        self.blob_circ_spin.setSingleStep(0.05)
        self.blob_circ_spin.setValue(self.config['blob_min_circularity'])
        blob_circ_h = QtWidgets.QHBoxLayout()
        blob_circ_h.addWidget(self.blob_circ_slider)
        blob_circ_h.addWidget(self.blob_circ_spin)
        controls_layout.addLayout(blob_circ_h)

        # Blob Convexity
        controls_layout.addWidget(QtWidgets.QLabel("Min Convexity:"))
        self.blob_conv_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blob_conv_slider.setRange(0, 100)
        self.blob_conv_slider.setValue(int(self.config['blob_min_convexity'] * 100))
        self.blob_conv_spin = QtWidgets.QDoubleSpinBox()
        self.blob_conv_spin.setRange(0.0, 1.0)
        self.blob_conv_spin.setSingleStep(0.05)
        self.blob_conv_spin.setValue(self.config['blob_min_convexity'])
        blob_conv_h = QtWidgets.QHBoxLayout()
        blob_conv_h.addWidget(self.blob_conv_slider)
        blob_conv_h.addWidget(self.blob_conv_spin)
        controls_layout.addLayout(blob_conv_h)

        # Blob Inertia
        controls_layout.addWidget(QtWidgets.QLabel("Min Inertia Ratio:"))
        self.blob_inert_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blob_inert_slider.setRange(0, 100)
        self.blob_inert_slider.setValue(int(self.config['blob_min_inertia'] * 100))
        self.blob_inert_spin = QtWidgets.QDoubleSpinBox()
        self.blob_inert_spin.setRange(0.0, 1.0)
        self.blob_inert_spin.setSingleStep(0.05)
        self.blob_inert_spin.setValue(self.config['blob_min_inertia'])
        blob_inert_h = QtWidgets.QHBoxLayout()
        blob_inert_h.addWidget(self.blob_inert_slider)
        blob_inert_h.addWidget(self.blob_inert_spin)
        controls_layout.addLayout(blob_inert_h)

        # Info
        controls_layout.addStretch()
        controls_layout.addWidget(QtWidgets.QLabel("<b>Tracking Info</b>"))
        self.centroid_label = QtWidgets.QLabel("Centroid: N/A")
        controls_layout.addWidget(self.centroid_label)
        self.pos_label = QtWidgets.QLabel("XYZ: N/A")
        controls_layout.addWidget(self.pos_label)

        # Right Panel: Visuals
        visuals_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(visuals_layout)
        top_visuals = QtWidgets.QHBoxLayout()

        # Top Visuals: Image Preview and 3D
        visuals_layout.addLayout(top_visuals, stretch=1)

        # Image Previews
        self.image_view_l = pg.GraphicsLayoutWidget()
        self.image_view_l.setBackground('gray')
        self.image_view_l.setMinimumSize(300, 400)
        self.image_view_l.addLabel("<span style='color: #00FF00; font-weight: bold;'>Left Camera</span>", row=0, col=0)
        self.vb_l = self.image_view_l.addViewBox(row=1, col=0)
        self.vb_l.setAspectLocked(True)
        self.img_item_l = pg.ImageItem()
        self.vb_l.addItem(self.img_item_l)
        self.crosshair_v_l = pg.InfiniteLine(angle=90, movable=False, pen='r')
        self.crosshair_h_l = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.vb_l.addItem(self.crosshair_v_l)
        self.vb_l.addItem(self.crosshair_h_l)
        self.crosshair_v_l.hide()
        self.crosshair_h_l.hide()

        self.image_view_r = pg.GraphicsLayoutWidget()
        self.image_view_r.setBackground('gray')
        self.image_view_r.setMinimumSize(300, 400)
        self.image_view_r.addLabel("<span style='color: #00FF00; font-weight: bold;'>Right Camera</span>", row=0, col=0)
        self.vb_r = self.image_view_r.addViewBox(row=1, col=0)
        self.vb_r.setAspectLocked(True)
        self.img_item_r = pg.ImageItem()
        self.vb_r.addItem(self.img_item_r)
        self.crosshair_v_r = pg.InfiniteLine(angle=90, movable=False, pen='r')
        self.crosshair_h_r = pg.InfiniteLine(angle=0, movable=False, pen='r')
        self.vb_r.addItem(self.crosshair_v_r)
        self.vb_r.addItem(self.crosshair_h_r)
        self.crosshair_v_r.hide()
        self.crosshair_h_r.hide()

        # Pre-fill both ImageItems with a blank frame of identical size so the
        # left and right previews have the same shape/aspect ratio right from
        # startup (before any real frame has been received from the worker),
        # and lock their views together so they always stay in sync.
        blank_w, blank_h = CAMERA_RESOLUTION
        blank_frame = np.zeros((blank_w, blank_h), dtype=np.uint8)
        self.img_item_l.setImage(blank_frame)
        self.img_item_r.setImage(blank_frame)
        self.vb_r.setXLink(self.vb_l)
        self.vb_r.setYLink(self.vb_l)

        top_visuals.addWidget(self.image_view_l, stretch=5)
        top_visuals.addWidget(self.image_view_r, stretch=5)

        # 3D View
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setMinimumSize(400, 300)
        grid = gl.GLGridItem(size=QtGui.QVector3D(100,100,1))
        grid.setSpacing(10, 10, 10,)
        self.gl_view.addItem(grid)
        self.pos_marker = gl.GLScatterPlotItem(pos=np.array([[0,0,0]]), color=(1,0,0,1), size=10)
        self.gl_view.addItem(self.pos_marker)
        top_visuals.addWidget(self.gl_view, stretch=6)

        # Plot
        self.plot_widget = pg.PlotWidget(title="XYZ over Time")
        self.plot_widget.setMinimumSize(400, 300)
        self.plot_widget.addLegend()
        self.curve_x = self.plot_widget.plot(pen='r', name='X')
        self.curve_y = self.plot_widget.plot(pen='g', name='Y')
        self.curve_z = self.plot_widget.plot(pen='b', name='Z')
        self.data_x = []
        self.data_y = []
        self.data_z = []
        self.max_points = 100
        visuals_layout.addWidget(self.plot_widget)

        # Connect signals
        self.exp_slider.valueChanged.connect(self.exp_spin.setValue)
        self.exp_spin.valueChanged.connect(self.update_exposure)
        self.iso_slider.valueChanged.connect(self.iso_spin.setValue)
        self.iso_spin.valueChanged.connect(self.update_iso)

        self.blob_min_thresh_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_threshold", v))
        self.blob_max_thresh_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_max_threshold", v))
        self.blob_min_area_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_area", v))
        self.blob_max_area_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_max_area", v))
        self.blob_circ_slider.valueChanged.connect(lambda v: self.blob_circ_spin.setValue(v / 100.0))
        self.blob_circ_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_circularity", v))
        self.blob_conv_slider.valueChanged.connect(lambda v: self.blob_conv_spin.setValue(v / 100.0))
        self.blob_conv_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_convexity", v))
        self.blob_inert_slider.valueChanged.connect(lambda v: self.blob_inert_spin.setValue(v / 100.0))
        self.blob_inert_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_inertia", v))

        # Worker thread
        self.worker = Worker(self.config)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.centroid_ready.connect(self.on_centroid)
        self.worker.position_ready.connect(self.on_position)
        self.worker.start()

    def update_exposure(self, val):
        self.worker.exposure = val
        self.worker.settings_changed = True
        self.exp_slider.blockSignals(True)
        self.exp_slider.setValue(val)
        self.exp_slider.blockSignals(False)

    def update_iso(self, val):
        self.worker.iso = val
        self.worker.settings_changed = True
        self.iso_slider.blockSignals(True)
        self.iso_slider.setValue(val)
        self.iso_slider.blockSignals(False)

    def update_blob_param(self, name, val):
        self.worker.blob_params[name] = val
        self.worker.blob_settings_changed = True
        if name == "blob_min_circularity":
            self.blob_circ_slider.blockSignals(True)
            self.blob_circ_slider.setValue(int(val * 100))
            self.blob_circ_slider.blockSignals(False)
        elif name == "blob_min_convexity":
            self.blob_conv_slider.blockSignals(True)
            self.blob_conv_slider.setValue(int(val * 100))
            self.blob_conv_slider.blockSignals(False)
        elif name == "blob_min_inertia":
            self.blob_inert_slider.blockSignals(True)
            self.blob_inert_slider.setValue(int(val * 100))
            self.blob_inert_slider.blockSignals(False)

    @QtCore.Slot(np.ndarray, np.ndarray)
    def on_frame(self, frame_l: np.ndarray, frame_r: np.ndarray):
        for frame, img_item in [(frame_l, self.img_item_l), (frame_r, self.img_item_r)]:
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
            mirrored = cv2.flip(rotated_frame, 1)
            img_item.setImage(mirrored.T)

    @QtCore.Slot(float, float, bool, float, float, bool)
    def on_centroid(self, x_l, y_l, found_l, x_r, y_r, found_r):
        # Update Left
        if found_l:
            self.crosshair_v_l.setPos(x_l)
            self.crosshair_h_l.setPos(y_l)
            self.crosshair_v_l.show()
            self.crosshair_h_l.show()
        else:
            self.crosshair_v_l.hide()
            self.crosshair_h_l.hide()

        # Update Right
        if found_r:
            self.crosshair_v_r.setPos(x_r)
            self.crosshair_h_r.setPos(y_r)
            self.crosshair_v_r.show()
            self.crosshair_h_r.show()
        else:
            self.crosshair_v_r.hide()
            self.crosshair_h_r.hide()

        text_l = f"L: {x_l:.1f}, {y_l:.1f}" if found_l else "L: N/A"
        text_r = f"R: {x_r:.1f}, {y_r:.1f}" if found_r else "R: N/A"
        self.centroid_label.setText(f"Centroid: {text_l} | {text_r}")

    @QtCore.Slot(np.ndarray)
    def on_position(self, pos):
        self.pos_label.setText(f"XYZ: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}")
        self.pos_marker.setData(pos=np.array([[pos[0], pos[2], pos[1]]]))
        self.data_x.append(pos[0])
        self.data_y.append(pos[1])
        self.data_z.append(pos[2])
        if len(self.data_x) > self.max_points:
            self.data_x.pop(0)
            self.data_y.pop(0)
            self.data_z.pop(0)
        self.curve_x.setData(self.data_x)
        self.curve_y.setData(self.data_y)
        self.curve_z.setData(self.data_z)

    def closeEvent(self, event):
        self.worker.stop()
        config_to_save = {
            'exposure': self.worker.exposure,
            'iso': self.worker.iso,
        }
        config_to_save.update(self.worker.blob_params)
        save_config(config_to_save)
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
