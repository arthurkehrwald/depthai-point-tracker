import math
import sys
import socket
import typing

import numpy as np
import cv2
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from blob_detector import BlobDetector
from camera import StereoCamera, CameraConfigKeys
from config import Config
from tracker import Tracker
from udp_sender import UdpSender


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


def match_candidates(candidates_l, candidates_r, stereo_cam: StereoCamera, conf_threshold: float):
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


class Worker(QtCore.QThread):
    frame_ready = QtCore.Signal(np.ndarray, np.ndarray)
    centroid_ready = QtCore.Signal(bool, typing.Tuple[float, float], typing.Tuple[float, float], typing.Tuple[int, int])
    position_ready = QtCore.Signal(np.ndarray)
    stats_ready = QtCore.Signal(float, float, float, float, float)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = True

    def run(self):
        with Tracker(self.config) as tracker:
            while self.running:
                detection = tracker.detect()

                self.frame_ready.emit(detection.frame.left_frame.copy(), detection.frame.right_frame.copy())
                self.stats_ready.emit(detection.frame.frame_time, detection.frame.left_time_of_capture,
                                      detection.frame.right_time_of_capture, detection.frame.time_of_arrival,
                                      detection.time_of_processing_finished)
                found = detection.confidence_01 >= self.config.get_default("detection_confidence_threshold", 0.5)
                resolution = detection.frame.left_frame.shape[1], detection.frame.left_frame.shape[0]
                self.centroid_ready.emit(found, detection.pos_2d_raw_l, detection.pos_2d_raw_r, resolution)
                if found:
                    self.position_ready.emit(detection.pos_3d)

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DepthAI Point Tracker")
        self.resize(1200, 900)

        self.config = Config()

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

        # Resolution
        self.initial_resolution = self.config.get('resolution')
        controls_layout.addWidget(QtWidgets.QLabel("Resolution:"))
        self.res_combo = QtWidgets.QComboBox()
        self.res_options = [(1280, 800), (1280, 720), (640, 400)]
        for w, h in self.res_options:
            self.res_combo.addItem(f"{w}x{h}")
        current_res = tuple(self.config.get('resolution'))
        if current_res in self.res_options:
            self.res_combo.setCurrentIndex(self.res_options.index(current_res))
        controls_layout.addWidget(self.res_combo)

        # Framerate
        controls_layout.addWidget(QtWidgets.QLabel("Framerate (FPS):"))
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(30, 100)
        self.fps_spin.setValue(self.config.get('fps', DEFAULT_FPS))
        controls_layout.addWidget(self.fps_spin)

        # Restart warning
        restart_warning = QtWidgets.QLabel("Resolution and FPS changes require restart")
        restart_warning.setWordWrap(True)
        restart_warning.setStyleSheet("color: orange; font-style: italic;")
        controls_layout.addWidget(restart_warning)

        controls_layout.addSpacing(10)

        # Exposure
        controls_layout.addWidget(QtWidgets.QLabel("Exposure (\u03bcs):"))
        self.exp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exp_slider.setRange(1, 3000)
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

        # Blob Threshold Step
        controls_layout.addWidget(QtWidgets.QLabel("Blob Threshold Step:"))
        self.blob_thresh_step_spin = QtWidgets.QSpinBox()
        self.blob_thresh_step_spin.setRange(1, 100)
        self.blob_thresh_step_spin.setValue(self.config.get('blob_threshold_step', 10))
        controls_layout.addWidget(self.blob_thresh_step_spin)

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

        # Stereo Match Confidence Threshold
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QtWidgets.QLabel("<b>Stereo Matching</b>"))
        controls_layout.addWidget(QtWidgets.QLabel("Confidence Threshold:"))
        self.stereo_conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.stereo_conf_slider.setRange(0, 100)
        self.stereo_conf_slider.setValue(
            int(self.config.get('stereo_conf_threshold', DEFAULT_STEREO_CONF_THRESHOLD) * 100))
        self.stereo_conf_spin = QtWidgets.QDoubleSpinBox()
        self.stereo_conf_spin.setRange(0.0, 1.0)
        self.stereo_conf_spin.setSingleStep(0.05)
        self.stereo_conf_spin.setValue(self.config.get('stereo_conf_threshold', DEFAULT_STEREO_CONF_THRESHOLD))
        stereo_conf_h = QtWidgets.QHBoxLayout()
        stereo_conf_h.addWidget(self.stereo_conf_slider)
        stereo_conf_h.addWidget(self.stereo_conf_spin)
        controls_layout.addLayout(stereo_conf_h)

        # Info
        controls_layout.addStretch()
        controls_layout.addWidget(QtWidgets.QLabel("<b>Tracking Info</b>"))
        self.centroid_label = QtWidgets.QLabel("Centroid: N/A")
        controls_layout.addWidget(self.centroid_label)
        self.pos_label = QtWidgets.QLabel("XYZ: N/A")
        controls_layout.addWidget(self.pos_label)

        controls_layout.addSpacing(10)
        controls_layout.addWidget(QtWidgets.QLabel("<b>Timing Info</b>"))
        self.frame_time_label = QtWidgets.QLabel("Frame time: N/A")
        controls_layout.addWidget(self.frame_time_label)
        self.capture_latency_label = QtWidgets.QLabel("Capture latency: N/A")
        controls_layout.addWidget(self.capture_latency_label)
        self.processing_latency_label = QtWidgets.QLabel("Processing latency: N/A")
        controls_layout.addWidget(self.processing_latency_label)
        self.total_latency_label = QtWidgets.QLabel("Total latency: N/A")
        controls_layout.addWidget(self.total_latency_label)
        self.lr_diff_label = QtWidgets.QLabel("L-R frame diff: N/A")
        controls_layout.addWidget(self.lr_diff_label)

        # Right Panel: Visuals
        visuals_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(visuals_layout)
        top_visuals = QtWidgets.QHBoxLayout()

        # Top Visuals: Image Preview and 3D
        visuals_layout.addLayout(top_visuals, stretch=1)

        # Image Previews (Vertical)
        camera_layout = QtWidgets.QVBoxLayout()
        top_visuals.addLayout(camera_layout, stretch=5)

        self.image_view_l = pg.GraphicsLayoutWidget()
        self.image_view_l.setBackground('gray')
        self.image_view_l.setMinimumSize(300, 200)
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
        self.image_view_r.setMinimumSize(300, 200)
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

        self.pred_marker_r = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush('g'))
        self.vb_r.addItem(self.pred_marker_r)

        # Pre-fill both ImageItems with a blank frame of identical size so the
        # left and right previews have the same shape/aspect ratio right from
        # startup (before any real frame has been received from the worker),
        # and lock their views together so they always stay in sync.
        blank_w, blank_h = self.config.get('resolution', DEFAULT_RESOLUTION)
        blank_frame = np.zeros((blank_w, blank_h), dtype=np.uint8)
        # Disable pyqtgraph's automatic level (brightness/contrast) scaling so the
        # preview reflects the camera's actual exposure instead of being
        # auto-stretched to the min/max of each frame, which otherwise makes
        # dark frames appear artificially lightened.
        self.img_item_l.setImage(blank_frame, autoLevels=False, levels=(0, 255))
        self.img_item_r.setImage(blank_frame, autoLevels=False, levels=(0, 255))
        self.vb_r.setXLink(self.vb_l)
        self.vb_r.setYLink(self.vb_l)

        camera_layout.addWidget(self.image_view_l)
        camera_layout.addWidget(self.image_view_r)

        # 3D View
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setMinimumSize(400, 300)
        self.gl_view.setCameraPosition(distance=200)
        grid = gl.GLGridItem(size=QtGui.QVector3D(100, 100, 1))
        grid.setSpacing(10, 10, 10, )
        self.gl_view.addItem(grid)
        self.pos_marker = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
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
        self.blob_thresh_step_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_threshold_step", v))
        self.blob_min_area_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_area", v))
        self.blob_max_area_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_max_area", v))
        self.blob_circ_slider.valueChanged.connect(lambda v: self.blob_circ_spin.setValue(v / 100.0))
        self.blob_circ_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_circularity", v))
        self.blob_conv_slider.valueChanged.connect(lambda v: self.blob_conv_spin.setValue(v / 100.0))
        self.blob_conv_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_convexity", v))
        self.blob_inert_slider.valueChanged.connect(lambda v: self.blob_inert_spin.setValue(v / 100.0))
        self.blob_inert_spin.valueChanged.connect(lambda v: self.update_blob_param("blob_min_inertia", v))
        self.stereo_conf_slider.valueChanged.connect(lambda v: self.stereo_conf_spin.setValue(v / 100.0))
        self.stereo_conf_spin.valueChanged.connect(self.update_stereo_conf)

        # Worker thread
        self.worker = Worker(self.config)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.centroid_ready.connect(self.on_centroid)
        self.worker.position_ready.connect(self.on_position)
        self.worker.stats_ready.connect(self.on_stats)
        self.worker.start()

    def update_exposure(self, val):
        self.worker.exposure = val
        self.config.set(CameraConfigKeys.exposure, val)
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

    def update_stereo_conf(self, val):
        self.worker.stereo_conf_threshold = val
        self.stereo_conf_slider.blockSignals(True)
        self.stereo_conf_slider.setValue(int(val * 100))
        self.stereo_conf_slider.blockSignals(False)

    @QtCore.Slot(np.ndarray, np.ndarray)
    def on_frame(self, frame_l: np.ndarray, frame_r: np.ndarray):
        for frame, img_item in [(frame_l, self.img_item_l), (frame_r, self.img_item_r)]:
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
            mirrored = cv2.flip(rotated_frame, 1)
            img_item.setImage(mirrored.T, autoLevels=False, levels=(0, 255))

    @QtCore.Slot(bool, typing.Tuple[float, float], typing.Tuple[float, float], typing.Tuple[int, int])
    def on_centroid(self, is_detected: bool, pos_2d_raw_l: typing.Tuple[float, float],
                    pos_2d_raw_r: typing.Tuple[float, float], resolution: typing.Tuple[int, int]):
        # Update Left
        if is_detected:
            frame_h = self.worker.resolution[1]
            # OpenCV is Y down, UI is Y up
            y_l = frame_h - pos_2d_raw_l[1]
            y_r = frame_h - pos_2d_raw_r[1]
            self.crosshair_v_l.setPos(pos_2d_raw_l[0])
            self.crosshair_h_l.setPos(y_l)
            self.crosshair_v_l.show()
            self.crosshair_h_l.show()
            self.crosshair_v_r.setPos(pos_2d_raw_r[0])
            self.crosshair_h_r.setPos(y_r)
            self.crosshair_v_r.show()
            self.crosshair_h_r.show()
        else:
            self.crosshair_v_l.hide()
            self.crosshair_h_l.hide()
            self.crosshair_v_r.hide()
            self.crosshair_h_r.hide()

        text_l = f"L: {x_l:.1f}, {y_l:.1f}" if is_detected else "L: N/A"
        text_r = f"R: {x_r:.1f}, {y_r:.1f}" if is_detected else "R: N/A"
        self.centroid_label.setText(f"Centroid: {text_l} | {text_r}")

    @QtCore.Slot(np.ndarray)
    def on_position(self, pos):
        self.pos_label.setText(f"XYZ: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}")
        # Flip z and y to transform to the space of the UI view
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

    @QtCore.Slot(float, float, float, float, float)
    def on_stats(self, frame_time: float, time_of_capture_l: float, time_of_capture_r: float, time_of_arrival: float,
                 time_of_3d_detection: float):
        fps = float(1000.0 / frame_time)
        self.frame_time_label.setText(f"Frame time: {frame_time:.1f}ms ({fps:.1f} FPS)")
        l_arrival = (time_of_capture_l - time_of_arrival) * 1e3
        self.capture_latency_label.setText(f"Capture latency: {l_arrival:.1f}ms")
        if time_of_3d_detection >= 0:
            l_processing = (time_of_3d_detection - time_of_arrival) * 1e3
            self.processing_latency_label.setText(f"Processing latency: {l_processing:.1f}ms")
            l_total = l_arrival + l_processing
            self.total_latency_label.setText(f"Total latency: {l_total:.1f}ms")
        else:
            self.processing_latency_label.setText("Processing latency: N/A")
            self.total_latency_label.setText("Total latency: N/A")

        diff_ts = (time_of_capture_r - time_of_capture_l) * 1e6
        self.lr_diff_label.setText(f"L-R frame diff: {diff_ts:.1f}µs")

    def closeEvent(self, event):
        self.worker.stop()
        self.config.save_file()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
